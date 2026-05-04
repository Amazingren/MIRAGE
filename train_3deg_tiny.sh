# micromamba environment
source ~/.bashrc
conda activate /share_5/users/bin_ren/envs/anyir

# Project directory
cd /share_5/users/bin_ren/project/ll/MIRAGE

### ===== [2] Configurations =====
CKPT_DIR="train_ckpt/3deg_tiny"
DENOISE_DIR="/share_5/users/bin_ren/datasets/ll/Train/Denoise/"
DERAIN_DIR="/share_5/users/bin_ren/datasets/ll/Train/Derain/"
DEHAZE_DIR="/share_5/users/bin_ren/datasets/ll/Train/Dehaze/"
GOPRO_DIR="/share_5/users/bin_ren/datasets/ll/Train/Deblur/"
ENHANCE_DIR="/share_5/users/bin_ren/datasets/ll/Train/Enhance/"

DE_TYPES="denoise_15 denoise_25 denoise_50 derain dehaze"

NUM_GPUS=1
BATCH_SIZE=32
EPOCHS=130
FFT_LOSS_WEIGHT=0.1

### ===== [3] Launch Training =====
CUDA_VISIBLE_DEVICES=5 python train_tiny.py \
    --trainset AnyIR \
    --ckpt_dir "$CKPT_DIR" \
    --de_type $DE_TYPES \
    --denoise_dir "$DENOISE_DIR" \
    --derain_dir "$DERAIN_DIR" \
    --dehaze_dir "$DEHAZE_DIR" \
    --gopro_dir "$GOPRO_DIR" \
    --enhance_dir "$ENHANCE_DIR" \
    --num_gpus $NUM_GPUS \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --fft_loss_weight $FFT_LOSS_WEIGHT