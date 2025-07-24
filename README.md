Medical Imaging Segmentation Toolkit
===

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/mist-medical/MIST/python-publish.yml)
![Read the Docs](https://img.shields.io/readthedocs/mist-medical?style=flat)
![PyPI - Downloads](https://img.shields.io/pypi/dm/mist-medical?style=flat&logo=PyPI&label=pypi%20downloads)
![GitHub Repo stars](https://img.shields.io/github/stars/mist-medical/MIST?style=flat)
![Coverage](coverage.svg)

## About
The Medical Imaging Segmentation Toolkit (MIST) is a simple, scalable, and end-to-end 3D medical imaging segmentation 
framework. MIST allows researchers to seamlessly train, evaluate, and deploy state-of-the-art deep learning models for 3D 
medical imaging segmentation.

Please cite the following papers if you use this code for your work:

[A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)

[A. Celaya et al., "FMG-Net and W-Net: Multigrid Inspired Deep Learning Architectures For Medical Imaging Segmentation," in Proceedings of LatinX in AI (LXAI) Research Workshop @ NeurIPS 2023, doi: 10.52591/lxai202312104](https://research.latinxinai.org/papers/neurips/2023/pdf/Adrian_Celaya.pdf)

[A. Celaya et al. "MIST: A Simple and Scalable End-To-End 3D Medical Imaging Segmentation Framework," arXiv preprint arXiv:2407.21343](https://www.arxiv.org/abs/2407.21343)

## Documentation
Please see our Read the Docs page [**here**](https://mist-medical.readthedocs.io/).

## What's New
* November 2024 - MedNeXt models (small, base, medium, and large) added to MIST.
These models can be called with ```--model mednext-v1-<small, base, medium, large>```.
* October 2024 - [MIST takes 3rd place in BraTS 2024 adult glioma challenge @ MICCAI 2024!](https://www.synapse.org/Synapse:syn53708249/wiki/630150)
* August 2024 - Added clDice as an available loss function.

