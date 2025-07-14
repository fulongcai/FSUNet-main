# FSUNet

//[![TGRS](https://img.shields.io/badge/IEEE%20TGRS-2026-blue.svg)](https://ieeexplore.ieee.org/document/10764792) [![ArXiv](https://img.shields.io/badge/ArXiv-2025-red.svg)](https://arxiv.org/abs/2406.13445)
> **What’s in the Frequency: Wavelet-Guided Semantic Understanding for Infrared Small Target Detection**  
> Wen Guo, Fulong Cai, and Wuzhou Quan

This repository contains the official implementation of the paper "**What’s in the Frequency: Wavelet-Guided Semantic Understanding for Infrared Small Target Detection**".
Besides, it is also a simple and integrated framework for infrared small target detection, which is easy to use and extend.

If our work is helpful to you, please cite it as follows:

```
todo
```

**_Thanks for your attention!_**

## Prerequisites

### Environment

```bash
python==3.9
conda install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda==11.8 -c pytorch -c nvidia

pip install einops mmengine click clearml rich scikit-image
```

For DCNv2, [lucasjinreal/DCNv2_latest](https://github.com/lucasjinreal/DCNv2_latest) is also required.

### Datasets

#### 1. File system architecture

Please ensure that the [IRSTD1K](https://github.com/RuiZhang97/ISNet), [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection), and [SIRST](https://github.com/YimianDai/open-acm) datasets are properly downloaded and organized as follows:

```
FSUNet (Root folder)
└── data
    ├── IRSTD1K
    │   ├── IRSTD1k_Img
    │   └── IRSTD1k_Label
    ├── NUAA
    │   ├── images
    │   └── masks
    └── NUDT-SIRST
        ├── images
        └── masks
```

If you need a custom folder structure, modify the `folder_arch` dictionary in the `gsettings.py` file to fit your folder architecture.
Additionally, ensure these folders have read and write permissions.

#### 2. Generate filter file

Execute `python gen_filter_files.py` to split the dataset into two mutually exclusive parts for distinguishing the training and testing sets.
You can modify the value of 0.2 to set the proportion of the test set (the default is 0.2, meaning a 4:1 split).

## Quick Start

### Training

If you want to train a model on IRSTD1k with batch size of 4, you can run the following command:

```bash
python train.py -m FSUNet -t irstd1k_train -v irstd1k_test -b 4 --max_epoches 500
```

**ClearML** is used for logging and visualization, for more details, please refer to their [official doc](https://clear.ml/docs/latest/docs/).

`--model_arch_name` or `-m` specifies the model architecture (located in the path of [`deployments/models`](deployments/models)). You can refer to the existing files to create your own model architecture (b.t.w. _it's quite simple_), or refer to the [official MMEngine config file documentation](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html) for more details.


`--train_dataset_name` or `-t` and `--val_dataset_name` or `-v` decide the training and validation datasets (located in the path of [`deployments/datasets`](deployments/datasets)), respectively.

`--batch_size` or `-b` specifies the batch size.

`--max_epoches` specifies the maximum number of training epochs.

### Testing

_Todo_

### Evaluation

_Todo_
