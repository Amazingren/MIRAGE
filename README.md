# MIRAGE: Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization

The official PyTorch Implementation of MIRAGE for All-in-One Image Restoration

#### [Bin Ren <sup>1,2</sup>](https://amazingren.github.io/), [Yawei Li<sup>5</sup>](https://yaweili.bitbucket.io/), [Xu Zheng<sup>4</sup>](https://scholar.google.com/citations?hl=en&user=Ii1c51QAAAAJ), [Yuqian Fu<sup>5</sup>](https://scholar.google.com/citations?user=y3Bpp1IAAAAJ&hl=en&oi=ao), [Danda Pani Paudel<sup>5</sup>](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en), [Hong Liu<sup>6</sup>](https://scholar.google.com/citations?user=WLMUAjsAAAAJ&hl=en)$^\dagger$, [Ming-Hsuan Yang <sup>7</sup>](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Luc Van Gool <sup>5</sup>](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), and [Nicu Sebe <sup>2</sup>](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)


$\dagger$: Corresponding author <br>

<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence, UAE, <br>
<sup>2</sup> University of Trento, Italy, <br>
<sup>3</sup> ETH Zürich, Switzerland, <br>
<sup>4</sup> HKUST (Guangzhou), China, <br>
<sup>5</sup> INSAIT Sofia University, "St. Kliment Ohridski", Bulgaria, <br>
<sup>6</sup> Peking University, China, <br>
<sup>7</sup> University of California, Merced, USA <br>

## Latest
- [ ] Projectpage release. 
- [ ] Main visual results release. 
- [ ] Checkpoints release.
- [x] Code release.
- `01/2026`: 🍺🎉 Our MIRAGE is accepted by ICLR2026!


## Method



## Installation
### 1) Environment
```bash
conda create -n mirage python=3.9 -y
conda activate mirage

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

### 2) Dependencies
```bash
# NOTE: file in this repo is currently named "requiements.txt"
pip install -r requiements.txt
```

### 3) CUDA Setup (if needed on your machine/cluster)
Please make sure CUDA is available on your system. You can check it with:

```bash
nvidia-smi
nvcc --version

# 📝 On many clusters, CUDA can be loaded via environment modules, e.g.:
module avail cuda
module load cuda/12.9

# 📝 If your cluster does not use module, please manually add your CUDA path:
export CUDA_HOME=/path/to/your/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 💡 For example:
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Datasets Preparation:
1. 3 Degradations & 5 Degradations Settings
You are suggested to follow the setup from P


2. CDD11 (Composited/Mixed Degradations).


3. 4-task Adverse Weather Removal.



## Checkpoints Downloads:
TODO


## Visual Results Downloads:
TODO


## Citation
If you find this project useful, please cite:
```bibtex
TODO
```


## Acknowledgements
This work was partially supported by the FIS project GUIDANCE (Debugging Computer Vision Models via Controlled Cross-modal Generation) (No. FIS2023-03251).

The code base is built on top of excellent prior work, including:
- [PromptIR](https://github.com/va1shn9v/PromptIR)
- [AirNet](https://github.com/XLearning-SCU/2022-CVPR-AirNet)
