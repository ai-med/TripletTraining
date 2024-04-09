# TripletTraining

This repository contains a PyTorch implementation of the paper [From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data](https://openreview.net/forum?id=7iW2nuL2lS), which is published in [MIDL](https://2024.midl.io/) 2024.

<p align="center">
  <img src="img/Triplet_training.png" />
</p>


If you find this repository useful, please consider giving a star and citing the paper:

```bibtex
@inproceedings{li2024triplet,
  title={From Barlow Twins to Triplet Training: Differentiating Dementia with Limited Data},
  author={Li, Yitong and Wolf, Tom Nuno and P{\"o}lsterl, Sebastian and Yakushev, Igor and Hedderich, Dennis M and Wachinger, Christian},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## Data

We used data from [UK Biobank](https://www.ukbiobank.ac.uk/), [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/), and [the Frontotemporal Lobar Degeneration Neuroimaging Initiative (NIFD)](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp) for self-supervised learning and self-distillation. Since we are not allowed to share our data, you would need to process the data yourself. Data for training, validation, and testing should be stored in separate [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) files, using the following hierarchical format:

1. First level: A unique identifier, e.g. image ID.
2. The second level always has the following entries:
    i. A group named 'MRI/T1', containing the T1-weighted 3D MRI data.
    ii. A string attribute 'DX' containing the diagnosis labels: 'CN', 'AD' or 'FTD', if available.
    iii. A scalar attribute 'RID' with the patient ID, if available.
    iv. Additional attributes depending on the task, such as 'Gender' and 'Age', if available.

## Installation

1. Install [Miniconda3](https://conda.io/en/latest/miniconda.html#latest-miniconda-installer-links)
2. Create environment: `conda env create -n triplet --file requirements.yaml`
3. Activate environment: `conda activate triplet`
4. Install `addiagnosis` package in development mode: `pip install --no-deps -e .`

## Usage

The package uses [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai) and [Hydra](https://hydra.cc).
PyTorch Lightning is a lightweight PyTorch wrapper.
Hydra's key feature is the ability to dynamically create a hierarchical configuration by composition and override
it through config files and the command line. It allows you to conveniently manage experiments.

The Python modules are located in the `src/addiagnosis` directory,
the Hydra configuration files are in the `configs` directory, where `configs/train.yaml` is
the main config for training (self-supervised learning and self-distillation), and `configs/train_transfer.yaml` for transfer learning on the your downstream tasks. Specify the `pretrained_model` path in the config files to continue training the next step with the pretrained backbone from the previous step.

After specifying the config files, simply start training (self-supervised learning or self-distillation) by:
```bash
python train.py
```
transfer learning on your downstream task:
```bash
python transfer_learning.py
```
and testing on your downstream task:
```bash
python test.py
```

## Contacts

For any questions, please contact: Yitong Li (yi_tong.li@tum.de)

## Acknowlegements

The self-supervised learning part of the codes were adopted into 3D implementation from [Barlow Twins](https://github.com/facebookresearch/barlowtwins), [VICReg](https://github.com/facebookresearch/vicreg), [SupContrast](https://github.com/facebookresearch/vicreg), [DiRA](https://github.com/fhaghighi/DiRA). I used [rfs](https://github.com/WangYueFt/rfs) as a reference for the self-distillation implementation.

