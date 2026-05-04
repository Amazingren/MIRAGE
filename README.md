# MIRAGE: 

<p align="center">
  <b>Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization</b><br/>
</p>


The official PyTorch Implementation of MIRAGE for All-in-One Image Restoration.

<p align="center">
  <a href="https://arxiv.org/pdf/2505.18679">
    <img src="https://img.shields.io/badge/arXiv-2504.14249-b31b1b.svg">
  </a>
  <a href="https://openreview.net/pdf?id=jDMAvoLsVj">
    <img src="https://img.shields.io/badge/OpenReview-ICLR%202026-blue">
  </a>
  <a href="https://github.com/Amazingren/MIRAGE">
    <img src="https://img.shields.io/badge/Project-Page-green">
  </a>
  <img src="https://img.shields.io/badge/PyTorch-Lightning-purple">
  </a>
  <img src="https://img.shields.io/badge/Task-All--in--One%20IR-blue">
</p>

<p align="center">
  <img src="assets/teaser.png" width="98%">
  <figcaption><em>Teaser: (a)-(d): Visual comparison for Denoising, Deraining, Composited Degradations (low-light, haze, and snow), and underwater image enhancement. (e): The average PSNR and SSIM comparison across 4 challenging all-in-one and 1 zero-shot settings (Please zoom in for a better view).</em></figcaption>
</p>

#### [Bin Ren <sup>1,2</sup>](https://amazingren.github.io/), [Yawei Li<sup>5</sup>](https://yaweili.bitbucket.io/), [Xu Zheng<sup>4</sup>](https://scholar.google.com/citations?hl=en&user=Ii1c51QAAAAJ), [Yuqian Fu<sup>5</sup>](https://scholar.google.com/citations?user=y3Bpp1IAAAAJ&hl=en&oi=ao), [Danda Pani Paudel<sup>5</sup>](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en), [Hong Liu<sup>6</sup>](https://scholar.google.com/citations?user=WLMUAjsAAAAJ&hl=en)$^\dagger$, [Ming-Hsuan Yang <sup>7</sup>](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Luc Van Gool <sup>5</sup>](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), and [Nicu Sebe <sup>2</sup>](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)


$\dagger$: Corresponding author <br>
<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence, UAE, <br>
<sup>2</sup> University of Trento, Italy, <br>
<sup>3</sup> ETH Zürich, Switzerland, <br>
<sup>4</sup> HKUST (Guangzhou), China, <br>
<sup>5</sup> INSAIT Sofia University, "St. Kliment Ohridski", Bulgaria, <br>
<sup>6</sup> Peking University, China, <br>
<sup>7</sup> University of California, Merced, USA <br>


## Latest & News:
- [ ] Projectpage Release. 
- [ ] Main Visual Results Release. 
- [ ] Checkpoints Release.
- [x] `05/2026`: Code Release.
- `01/2026`: 🍺 Our MIRAGE is accepted by ICLR2026!


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
pip install -r requirements.txt
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

## Dataset Preparation

### 1) 3-Degradation and 5-Degradation Settings

We follow the dataset preparation protocols from the following prior works:

* **3-Degradation setting**: [PromptIR(NeurIPS2023)](https://github.com/va1shn9v/PromptIR/blob/main/INSTALL.md)
* **5-Degradation setting**: [AdaIR(ICLR2025)](https://github.com/c-yn/AdaIR/blob/main/INSTALL.md)

#### (Optional) Preprocessed Datasets

For convenience, we provide preprocessed and well-organized datasets via Google Drive.

> ⚠️ **Important:**
> Please strictly follow the original dataset licenses and usage policies.
> The provided datasets are for **academic research purposes only**.

**Training Sets:**

| Dehaze | Derain | Denoising | Deblurring | Low-light |
| --- | --- | --- | ---| --- |
| [Download (11.2G)](https://drive.google.com/file/d/13LBouXHNsMKyL5rEpnnpXer2RaBZ1Xwq/view?usp=sharing) | [Download (103.6M)](https://drive.google.com/file/d/12ugQ-jKevGDSwbi0im5Uh6dLXZXISCou/view?usp=sharing) | [Download (3.02G)](https://drive.google.com/file/d/1O8k0hXHYn0FtIR7ABViwP1MksGtN0YGa/view?usp=sharing) | [Download (3.8G)](https://drive.google.com/file/d/1d7ga-ZE4iWTsW-CFnKpTsgWB4d6rHVru/view?usp=sharing) | [Download (322.0M)](https://drive.google.com/file/d/1P9tVjPp4G4jftG-9VhYv_0-kRpoZimlu/view?usp=sharing) |

> If you directly download the datasets from the above links, the directory structure is already organized as required as below:
```
.../datasets/Train/
├── Deblur/
│   ├── blur/
│   └── sharp/
├── Dehaze/
│   ├── train/
│   └── test/
├── Denoise/
│   ├── ...
│   └── xxx.png / xxx.jpg
├── Derain/
│   ├── gt/
│   └── rainy/
└── Enhance/
    ├── gt/
    └── low/
```

**Inference Sets:**
> Please download the preprocessed test sets via [Download]() (including both the 3-Degradation and 5-Degradation settings). The datasets are already organized as follows:
```
.../datasets/test/
├── deblur/
│   └── gopro/
│       ├── input/
│       └── target/
├── dehaze/
│   ├── input/
│   └── target/
├── denoise/
│   ├── bsd68/
│   └── urban100/
├── derain/
│   └── Rain100L/
│       ├── input/
│       └── target/
└── enhance/
    └── lol/
        ├── input/
        └── target/
```

### 2) CDD11 (Composited / Mixed Degradations)
|Train|Test|
|---|---|
|[Download(21.0G)](https://drive.google.com/file/d/1lbh4BoL9T210vv4YnW33FbIcoC7vbzk6/view?usp=sharing)|[Download(3.5G)](https://drive.google.com/file/d/1LzTq3-qy2N8cNWuriqfJvHVaYHeimKQp/view?usp=sharing)|


### 3) 4-Task Adverse Weather Removal.
|Train & Test|
|---|
|[Download(17.1G)]()|


## Visual Results Downloads:
TODO

## Checkpoints Downloads:
TODO

## Inference
### 1) 3-Degradation Setting
```bash
# 🦖 Tiny Model (6M)
sh test_3deg_tiny.sh

# 🦖 Small Model (10M)
sh test_3deg_small.sh
```

### 2) 5-Degradation Setting
```bash
# 🦖 Tiny Model (6M)
sh test_5deg_tiny.sh

# 🦖 Small Model (10M)
sh test_5deg_small.sh
```


## Training
### 1) 3-Degradation Setting
```bash
# 🦖 Tiny Model (6M)
sh train_3deg_tiny.sh

# 🦖 Small Model (10M)
sh train_3deg_small.sh
```

### 3) Composited / Mixed (CDD11) Degradation Setting
TODO

### 4) 4-Task Adverse Weather Removal.
TODO


### 2) 5-Degradation Setting
```bash
# 🦖 Tiny Model (6M)
sh train_5deg_tiny.sh

# 🦖 Small Model (10M)
sh train_5deg_small.sh
```

### 3) Composited / Mixed (CDD11) Degradation Setting
TODO

### 4) 4-Task Adverse Weather Removal.
TODO




## Cite Our Work
If you find this project useful, please cite:
```bibtex
@inproceedings{ren2026efficient,
  title={Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization}, 
  author={Bin Ren and Yawei Li and Xu Zheng and Yuqian Fu and Danda Pani Paudel and Hong Liu and Ming-Hsuan Yang and Luc Van Gool and Nicu Sebe},
  booktitle={The Fourteenth International Conference on Learning Representations(ICLR)},
  year={2026}
}
```

## Acknowledgements
This work was partially supported by the FIS project GUIDANCE (Debugging Computer Vision Models via Controlled Cross-modal Generation) (No. FIS2023-03251).

The code base is built on top of excellent prior work, including:
- [PromptIR(NeurIPS2023)](https://github.com/va1shn9v/PromptIR)
- [AirNet(CVPR2022)](https://github.com/XLearning-SCU/2022-CVPR-AirNet)