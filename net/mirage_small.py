"""
MIRAGE: Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization
Author: Bin Ren
Conference: ICLR 2026
"""

import numbers
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import trunc_normal_, DropPath


# Constants
BIAS_FREE = 'BiasFree'
WITH_BIAS = 'WithBias'


# Utility Functions
def to_3d(x):
    """Reshape tensor from 4D to 3D."""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """Reshape tensor from 3D back to 4D."""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# Layer Normalization Components
class BiasFree_LayerNorm(nn.Module):
    """Layer normalization without bias."""
    
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    """Layer normalization with bias."""
    
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        
        assert len(normalized_shape) == 1
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer normalization factory that creates either biased or unbiased norm."""
    
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == BIAS_FREE:
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class GatedCNNBlock(nn.Module):
    """Dynamic Gated CNN Block based on https://arxiv.org/pdf/1612.08083"""
    def __init__(
        self, 
        dim, 
        expansion_ratio=8/3, 
        kernel_size=7, 
        conv_ratio=1.0,
        norm_layer=nn.BatchNorm2d, 
        act_layer=nn.GELU,
        drop_path=0.,
        dynamic_conv=True,
        **kwargs
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.gic1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.act = act_layer()
        
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)

        self.dynamic_conv = dynamic_conv
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size

        if dynamic_conv:
            # Dynamic kernel generator 
            self.kernel_generator = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(conv_channels, conv_channels, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(conv_channels, conv_channels * kernel_size * kernel_size, 1, bias=False)
            )
        else:
            # Normal depthwise convolution
            self.cconv = nn.Conv2d(
                conv_channels, 
                conv_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2, 
                groups=conv_channels,
                bias=False
            )

        self.gic2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        shortcut = x
        x = self.norm(x)

        gic = self.gic1(x)  # [B, 2 * hidden, H, W]
        g, i, c = torch.split(gic, self.split_indices, dim=1)

        if self.dynamic_conv:
            # --- Dynamic Depthwise Conv ---
            b, ch, h, w = c.shape
            kernel = self.kernel_generator(c)  # [B, ch*k*k, 1, 1]
            kernel = kernel.view(b * ch, 1, self.kernel_size, self.kernel_size)

            # c = c.view(1, b * ch, h, w)  # 变成group conv输入
            c = c.reshape(1, b * ch, h, w)
            c = F.conv2d(c, kernel, padding=self.kernel_size//2, groups=b*ch)
            # c = c.view(b, ch, h, w)
            c = c.reshape(b, ch, h, w)
        else:
            # --- Static Depthwise Conv ---
            c = self.cconv(c)

        # --- Gating + Output ---
        gate = self.act(g / self.temperature)
        x = self.gic2(gate * torch.cat((i, c), dim=1))
        x = self.drop_path(x)

        x = x + shortcut
        x = self.proj_out(x)

        return x


# Neural Network Building Blocks
class FeedForward(nn.Module):
    """Feed Forward Network with GELU activation and depthwise convolution."""
    
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        
        hidden_features = int(dim * ffn_expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, 
            hidden_features * 2, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=hidden_features * 2, 
            bias=bias
        )
        self.act = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    
    def forward(self, x):
        x = self.project_in(x)
        x = self.act(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dim_attn = dim // 3
        self.dim_gated = dim // 3
        self.dim_mlp = dim - self.dim_attn - self.dim_gated

        # --- for attention
        self.qkv = nn.Conv2d(self.dim_attn, self.dim_attn*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            self.dim_attn*3, 
            self.dim_attn*3, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            groups=self.dim_attn*3, 
            bias=bias
        )
        
        # --- for gated conv
        self.gatedcnn = GatedCNNBlock(
            dim=self.dim_gated,
            drop_path=0.1,
        )
        
        self.mlp = nn.Sequential(
            nn.Conv1d(self.dim_mlp, self.dim_mlp * 2, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv1d(self.dim_mlp * 2, self.dim_mlp, kernel_size=1, bias=bias),
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.alpha_att = nn.Parameter(torch.ones(1))
        self.alpha_gated = nn.Parameter(torch.ones(1))
        self.alpha_mlp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        b,c,h,w = x.shape
        x_attn, x_gated, x_mlp = torch.split(x, [self.dim_attn, self.dim_gated, self.dim_mlp], dim=1)

        # --- 1. Attention Branch 
        qkv = self.qkv_dwconv(self.qkv(x_attn))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # --- 2. Gated Branch
        out_gated = self.gatedcnn(x_gated)

        # --- 3. MLP Branch
        out_mlp = x_mlp.view(b, self.dim_mlp, h*w)  # (B, C, L)
        out_mlp = self.mlp(out_mlp)
        out_mlp = out_mlp.view(b, self.dim_mlp, h, w)

        # --- 4. Fusion
        out_att_ = out_att + self.alpha_att * torch.sigmoid(out_gated + out_mlp)
        out_gated_ = out_gated + self.alpha_gated * torch.sigmoid(out_att + out_mlp)
        out_mlp_ = out_mlp + self.alpha_mlp * torch.sigmoid(out_att + out_gated)

        out = torch.cat((out_att_, out_gated_, out_mlp_), dim=1)  # [B, dim, H, W]

        out = self.project_out(out)
        return out


class ResBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# Resizing modules
class Downsample(nn.Module):
    """Downsample spatial resolution by 2x and double channels."""
    
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Upsample spatial resolution by 2x and halve channels."""
    
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv."""
    
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

def global_pool(feat):
    return F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # [B, C]

def contrastive_loss(z1, z2, temperature=0.1):
    # NT-Xent (SimCLR Style) Contrastive Loss
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)

class MIRAGE(nn.Module):
    """
    MIRAGE: A neural network for image restoration.
    
    This model uses a UNet-like architecture with transformer blocks for feature extraction
    and processing at multiple resolutions.
    
    Args:
        inp_channels (int): Number of input channels (default: 3)
        out_channels (int): Number of output channels (default: 3)
        dim (int): Base dimension for feature maps (default: 24)
        num_blocks (list): Number of transformer blocks at each resolution (default: [3,5,5,7])
        num_refinement_blocks (int): Number of refinement blocks (default: 2)
        heads (list): Number of attention heads at each resolution (default: [1,2,4,8])
        ffn_expansion_factor (float): Expansion factor for FFN (default: 2)
        bias (bool): Whether to use bias in convolutions (default: False)
        LayerNorm_type (str): Type of layer normalization (default: 'WithBias')
    """
    
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=30,
        num_blocks=[3, 5, 5, 7],
        num_refinement_blocks=3,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type=WITH_BIAS,
    ):
        super(MIRAGE, self).__init__()

        # Initial feature extraction
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Encoder path
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Downsampling and level 2 processing
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Downsampling and level 3 processing
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Downsampling and bottleneck (latent) processing
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])
        
        # Decoder path
        # Level 4 to 3
        self.up4_3 = Upsample(int(dim*2**2))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)
        self.reduce_dim_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])
        
        # Level 3 to 2
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])
        
        # Level 2 to 1
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])
        
        # Final refinement and output
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.reduce_shallow = nn.Conv2d(30, 12, 1)
        self.reduce_latent = nn.Conv2d(120, 24, 1)

        # SPD projection after flattening
        self.spd_proj_shallow = nn.Linear(12 * 12, 64)
        self.spd_proj_latent  = nn.Linear(24 * 24, 64)


    def compute_spd_feature(self, x):
        """
        Extract SPD features from x and flatten to a vector (without projection)
        Args:
            x: Tensor [B, C, H, W]
        Returns:
            Tensor [B, C*C]
        """
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x, x.transpose(1, 2)) / (x.shape[2] - 1)

        eye = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        cov = cov + 1e-5 * eye  # Stabilization for SPD

        return cov.view(B, -1)  # [B, C*C]


    def forward(self, inp_img):
        # --- Initial feature extraction
        inp_enc_level1 = self.patch_embed(inp_img)  # [B, dim, H, W]
        
        # --- Encoder path (downsampling)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # [B, dim, H, W]
        
        inp_enc_level2 = self.down1_2(out_enc_level1)  # [B, dim*2, H/2, W/2]
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)  # [B, dim*4, H/4, W/4]
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)  # [B, dim*8, H/8, W/8]
        latent = self.latent(inp_enc_level4)
        
        # --- Prepare bottleneck features for decoder
        latent = self.reduce_dim_level3(latent)  # [B, dim*4, H/8, W/8]
        
        # --- Decoder path (upsampling with skip connections)
        # Level 4 to 3
        inp_dec_level3 = self.up4_3(latent)  # [B, dim*4, H/4, W/4]
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        # Level 3 to 2
        inp_dec_level2 = self.up3_2(out_dec_level3)  # [B, dim*2, H/2, W/2]
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        # Level 2 to 1
        inp_dec_level1 = self.up2_1(out_dec_level2)  # [B, dim*2, H, W]
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # --- Final refinement
        out_dec_level1 = self.refinement(out_dec_level1)
        
        # Output with residual connection to input
        output = self.output(out_dec_level1) + inp_img
        
        # --- SPD Contrastive loss
        shallow_feat = out_enc_level1  # [B, C(24), H, W]
        latent_feat = latent           # [B, dim*4(96), H/8, W/8]

        shallow_feat = self.reduce_shallow(shallow_feat)  # [B, 12, H, W]
        latent_feat = self.reduce_latent(latent_feat)      # [B, 24, H/8, W/8]

        # SPD feature extraction
        z_s_raw = self.compute_spd_feature(shallow_feat)    # [B, 144]
        z_l_raw = self.compute_spd_feature(latent_feat)     # [B, 576]

        # Projection
        z_s = F.normalize(self.spd_proj_shallow(z_s_raw), dim=1)  # [B, 64]
        z_l = F.normalize(self.spd_proj_latent(z_l_raw), dim=1)   # [B, 64]

        loss_ctr = contrastive_loss(z_s, z_l)  # Contrastive loss
        loss_ctr_spd = 0.01 * loss_ctr

        return output, loss_ctr_spd


# Testing code with performance metrics
if __name__ == "__main__":
    # Create model instance
    model = MIRAGE(
        num_blocks=[3, 5, 5, 7],
        dim=30,
        ffn_expansion_factor=2,
        num_refinement_blocks=3
    ).cuda()
    
    # Create test input
    x = torch.randn(1, 3, 224, 224).cuda()
    
    # Print model architecture
    print(model)
    
    # Run forward pass
    out, _ = model(x)
    
    # Print memory usage
    print('{:>16s} : {:<.3f} [M]'.format(
        'Max Memory', 
        torch.cuda.max_memory_allocated(torch.cuda.current_device())/1024**2
    ))
    print(f'Output shape: {out.shape}')
    
    # Calculate and print FLOPS and parameters
    flops = FlopCountAnalysis(model, (x))
    print(flop_count_table(flops))