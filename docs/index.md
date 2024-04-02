Medical Imaging Segmentation Toolkit
===

## About
The Medical Imaging Segmentation Toolkit (MIST) is a simple, scalable, and end-to-end 3D medical imaging segmentation 
framework. MIST allows researchers to seamlessly train, evaluate, and deploy state-of-the-art deep learning models for 3D 
medical imaging segmentation.

!!! warning ""
    MIST is licensed under [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please click [here](https://github.com/aecelaya/MIST/blob/main/LICENSE) to see the license file.

!!! note ""
    Please cite the following papers if you use this code for your work:

    1. [A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on 
    Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)

    2. [A. Celaya et al., "FMG-Net and W-Net: Multigrid Inspired Deep Learning Architectures For Medical Imaging Segmentation", in
    Proceedings of LatinX in AI (LXAI) Research Workshop @ NeurIPS 2023, doi: 10.52591/lxai202312104](https://research.latinxinai.org/papers/neurips/2023/pdf/Adrian_Celaya.pdf)

## What's New
* April 2024 - The Read the Docs page is up!
* March 2024 - Simplify and decouple postprocessing from main MIST pipeline.
* March 2024 - Support for using transfer learning with pretrained MIST models is now available.
* March 2024 - Boundary-based loss functions are now available.
* Feb. 2024 - MIST is now available as PyPI package and as a Docker image on DockerHub.
* Feb. 2024 - Major improvements to the analysis, preprocessing, and postprocessing pipelines, 
and new network architectures like UNETR added.
* Feb. 2024 - We have moved the TensorFlow version of MIST to [mist-tf](https://github.com/aecelaya/mist-tf).