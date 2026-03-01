# MIRAGE

### Efficient Degradation-agnostic Image Restoration via Channel-Wise Functional Decomposition and Manifold Regularization

The official PyTorch Implementation of AnyIR for All-in-One Image Restoration

#### [Bin Ren <sup>1,2</sup>](https://amazingren.github.io/)$^\star$, [Yawei Li<sup>5</sup>](https://yaweili.bitbucket.io/), [Xu Zheng<sup>4</sup>](https://scholar.google.com/citations?hl=en&user=Ii1c51QAAAAJ), [Yuqian Fu<sup>5</sup>](https://scholar.google.com/citations?user=y3Bpp1IAAAAJ&hl=en&oi=ao), [Danda Pani Paudel<sup>5</sup>](https://scholar.google.com/citations?user=W43pvPkAAAAJ&hl=en), [Hong Liu<sup>6</sup>](https://scholar.google.com/citations?user=WLMUAjsAAAAJ&hl=en)$^\dagger$, [Ming-Hsuan Yang <sup>7</sup>](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en), [Luc Van Gool <sup>5</sup>](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en), and [Nicu Sebe <sup>2</sup>](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)


$\star$: This work was partially conducted during the visiting stay at INSAIT. <br>
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
- [ ] Ckpts release. 
- [ ] Main visual results release. 
- [ ] Code release.
- `01/2026`: 🍺🎉 Our MIRAGE is accepted by ICLR2026!


## Method


## Installation
### 1) Environment
```bash
micromamba create -n mirage python=3.9 -y
micromamba activate mirage
# or
conda create -n mirage python=3.9 -y
conda activate mirage
```

### 2) Dependencies
```bash
# NOTE: file in this repo is currently named "requiements.txt"
pip install -r requiements.txt
```

### 3) CUDA (if needed on your cluster)
```bash
export LD_LIBRARY_PATH=/opt/modules/nvidia-cuda-11.8.0/lib64:$LD_LIBRARY_PATH
export PATH=/opt/modules/nvidia-cuda-11.8.0/bin:$PATH
```

## Datasets Preparation:
TODO


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
