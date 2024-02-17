# Medical Imaging Segmentation Toolkit

## Overview
The Medical Imaging Segmentation Toolkit (MIST) is a simple, fully automated 3D medical imaging segmentation 
framework. MIST allows researchers to quickly set up, train, and test a variety of deep learning models for 3D 
medical imaging segmentation. The following architectures are implemented on MIST:

* nnUNet
* U-Net
* Attention U-Net
* U-NetTR

The following features are supported by MIST: 

* [NVIDIA Data Loading Library (DALI)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
* [Automatic mixed precision (AMP)](https://www.tensorflow.org/guide/mixed_precision)
* [Multi-GPU training with DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

Please cite the following if you use this code for your work:

> [A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on 
> Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)


## What's New
* Feb. 2024 - Major improvements to the analysis, preprocessing, and postprocessing pipelines, 
and new network architectures like UNETR added.
* Feb. 2024 - We have moved the TensorFlow version of MIST to [mist-tf](https://github.com/aecelaya/mist-tf).

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
    * [Requirements](#requirements)
    * [Data Format](#data-format)
    * [Getting Started](#getting-started)
    * [Output](#output)
    * [Advanced Usage](#advanced-usage)
- [Inference](#inference)
    * [Overview](#overview)
    * [Advanced Usage](#advanced-usage)
- [MSD and CSV Formatted Data](#msd-and-csv-formatted-data)

 
## Setup
### Requirements
#### Conda
We include a YAML file for building the correct Conda environment to run the MIST pipeline. Please
be sure to have [Conda](https://docs.conda.io/en/latest/) installed.

#### Docker
We include a Dockerfile in this repository that builds the necessary Docker container and installs the 
dependencies. Please make sure to have the following components ready.

* [Docker](https://www.docker.com/)
* [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Data Format
The MIST pipeline assumes that your train and test data directories 
are set up in the following structure.
```
data/
│   └── patient_1
│       └── image_1.nii.gz
│       └── image_2.nii.gz
│       ...
│       └── image_n.nii.gz
|       └── mask.nii.gz
|
│   └── patient_2
│       └── image_1.nii.gz
│       └── image_2.nii.gz
│       ...
│       └── image_n.nii.gz
|       └── mask.nii.gz
|
|   ...
└── 
```
Please note that the naming convention here is for this example only. There is no specific 
naming convention for the files within your dataset. However, the MIST pipeline assumes that the naming of the 
images in each patient directory is consistent or that each type of image is identifiable by a 
list of identifier strings.

MIST offers support for MSD and CSV formatted datasets. For more details, please see [MSD and CSV Formatted Data](#msd-and-csv-formatted-data).

Once your dataset is in the correct format, the final step is to prepare a small JSON 
file containing the details of the dataset. We specifically ask for the following key-value 
pairs.

| Key                 | Value                                                                                                                                                    |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```task```          | Name of task (i.e., brats, lits, etc.).                                                                                                                  |
| ```modality```      | Options are ``ct``, ``mr``, or ``other``.                                                                                                                |
| ```train-data```    | Full path to training data.                                                                                                                              |
| ```test-data```     | Full path to test data (optional).                                                                                                                       |
| ```mask```          | List containing identifying strings for mask or ground truth images in dataset.                                                                          |
| ```images```        | Dictionary where each key is an image type (i.e., T1, T2, CT, etc.) and each value is a list containing identifying strings for that image type.         |
| ```labels```        | List of labels in dataset (starting with 0).                                                                                                             |
| ```final_classes``` | Dictionary where each key is the name of the final segmentation class (i.e., WT, ET, TC for BraTS) and each value is a list of the labels in that class. |

Here is an example for the BraTS 2023 dataset.

```
{
    "task": "brats2023",
    "modality": "mr",
    "train-data": "/full/path/to/raw/data/train",
    "test-data": "/full/path/to/raw/data/validation",
    "mask": ["seg.nii.gz"],
    "images": {"t1": ["t1n.nii.gz"],
               "t2": ["t2w.nii.gz"],
               "tc": ["t1c.nii.gz"],
               "fl": ["t2f.nii.gz"]},
    "labels": [0, 1, 2, 3],
    "final_classes": {"WT": [1, 2, 3],
                      "TC": [1, 3],
                      "ET": [3]}
}
```

Examples for several datasets are provided in the ```examples/dataset-json``` directory.

### Getting Started
#### Conda
To start running the MIST pipeline via a Conda environment, first clone this repository and build
the environment via the YAML file provided.


```
git clone https://github.com/aecelaya/MIST.git
cd MIST
```

Change the ```prefix``` line at the bottom of the ```mist-torch.yml``` file to match your system
and then run the following line to create the environment.

```
conda env --create --file mist-torch.yml
```

Once the Conda environment is ready, you can run (with all default values) the MIST pipeline with
the following command:

```
conda activate mist-torch
cd MIST/src
python main.py --data examples/brats.json \
--numpy /path/to/save/numpy/files \
--results /path/to/save/results
```

For more options see the [Advanced Usage](#advanced-usage) section below.

If your system can support AMP, then we strongly recommend running the MIST pipeline with the ```--amp``` flag.


#### Docker
To start running the MIST pipeline via Docker, first clone this repository.

```
git clone https://github.com/aecelaya/MIST.git
cd MIST/mist-torch
```

Next we use Dockerfile in this repository to build a Docker image named ```mist-torch```.
```
docker build -t mist-torch .
```

Once the container is ready, we launch an interactive session with the following command. 
Notice that we mount a directory to the ```/workspace``` directory in the container. The idea 
is to mount the local directory containing our data to the workspace directory. Please modify 
this command to match your system.
```
docker run --rm -it -u $(id -u):$(id -g) \ 
--gpus all --ipc=host --ulimit memlock=-1 \ 
--ulimit stack=67108864 \ 
-v /your/working/directory/:/workspace \
 mist-torch
```

Once inside the container, we can run the MIST pipeline (with all default values) with the 
following command.
```
cd src
python main.py --data examples/brats.json \
--numpy /workspace/path/to/brats/numpy/files \
--results /workspace/path/to/save/results
```
For more options see the [Advanced Usage](#advanced-usage) section below.

Again, if your system can support AMP, then we strongly recommend running the MIST pipeline with the ```--amp``` flag.


### Output
The output of the MIST pipeline has the following structure:
```
results/
│   └── logs/
│   └── models/
│   └── predictions/
│   config.json
│   results.csv
│   train_paths.csv
│   test_paths.csv (if test set is specified in dataset.json)
│   fg_bboxes.csv
└── 
```

Here is a breakdown of what each file/directory contains.

|     Directory/File    |                                                                           Description                                                                          |
|:---------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|      ```logs/```      | TensorBoard logs for each fold.                                                                                                                                |
|     ```models/```     | PyTorch models for each fold and ```model_config.json```, which stores the settings (i.e., architecture, patch size) to load the model from the saved weights. |
|   ```predictions/```  | Raw and postprocessed predictions from five fold cross validation and test set (if specified).                                                                 |
|   ```config.json```   | JSON file containing all of the attributes of the dataset (i.e., target spacing, crop to foreground, etc.).                                                    |
|   ```results.csv```   | CSV file with the Dice, 95th percentile Hausdorff, and average surface distance scores for all of the folds.                                                   |
| ```train_paths.csv``` | CSV file starting with columns ```id``` and ```fold``` specifying the patient ID and which fold they belong to, and paths to the mask and images.              |
|  ```test_paths.csv``` | Same as ```train_paths.csv```, but for the test set if it's given.                                                                                             |
|  ```fg_bboxes.csv```  | CSV file containing information about the bounding box around the foreground for each image.                                                                   |


### Advanced usage
To see the complete list of available options and their descriptions, use the -h or --help command-line option, for example:

```
python main.py --help
```

The following output is printed when running the command above:
```
usage: main.py [-h] [--exec-mode {all,analyze,preprocess,train}] [--data DATA]
               [--gpus GPUS [GPUS ...]] [--num-workers NUM_WORKERS]
               [--master-port MASTER_PORT] [--seed SEED] [--tta [BOOLEAN]]
               [--results RESULTS] [--numpy NUMPY] [--amp [BOOLEAN]]
               [--batch-size BATCH_SIZE]
               [--patch-size PATCH_SIZE [PATCH_SIZE ...]]
               [--learning-rate LEARNING_RATE] [--exp_decay EXP_DECAY]
               [--lr-scheduler {constant,cosine_warm_restarts,exponential}]
               [--cosine-first-steps COSINE_FIRST_STEPS]
               [--optimizer {sgd,adam,adamw}] [--clip-norm [BOOLEAN]]
               [--clip-norm-max CLIP_NORM_MAX]
               [--model {nnunet,unet,attn_unet,unetr}]
               [--use-res-block [BOOLEAN]] [--pocket [BOOLEAN]]
               [--depth DEPTH] [--init-filters INIT_FILTERS]
               [--deep-supervision [BOOLEAN]]
               [--deep-supervision-heads DEEP_SUPERVISION_HEADS]
               [--vae-reg [BOOLEAN]] [--vae-penalty VAE_PENALTY]
               [--l2-reg [BOOLEAN]] [--l2-penalty L2_PENALTY]
               [--l1-reg [BOOLEAN]] [--l1-penalty L1_PENALTY]
               [--oversampling OVERSAMPLING] [--no-preprocess [BOOLEAN]]
               [--use-n4-bias-correction [BOOLEAN]]
               [--use-config-class-weights [BOOLEAN]]
               [--class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]]
               [--loss {dice_ce,dice,gdl,gdl_ce}] [--sw-overlap SW_OVERLAP]
               [--blend-mode {constant,gaussian}] [--postprocess [BOOLEAN]]
               [--nfolds NFOLDS] [--folds FOLDS [FOLDS ...]] [--epochs EPOCHS]
               [--steps-per-epoch STEPS_PER_EPOCH] [--output-std [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --exec-mode {all,analyze,preprocess,train}
                        Run all of the MIST pipeline or an individual
                        component (default: all)
  --data DATA           Path to dataset json file (default: None)
  --gpus GPUS [GPUS ...]
                        Which gpu(s) to use, defaults to all available GPUs
                        (default: [-1])
  --num-workers NUM_WORKERS
                        Number of workers to use for data loading (default: 8)
  --master-port MASTER_PORT
                        Master port for multi-gpu training (default: 12355)
  --seed SEED           Random seed (default: 42)
  --tta [BOOLEAN]       Enable test time augmentation (default: False)
  --results RESULTS     Path to output of MIST pipeline (default: None)
  --numpy NUMPY         Path to save preprocessed numpy data (default: None)
  --amp [BOOLEAN]       Enable automatic mixed precision (recommended)
                        (default: False)
  --batch-size BATCH_SIZE
                        Batch size (default: 2)
  --patch-size PATCH_SIZE [PATCH_SIZE ...]
                        Height, width, and depth of patch size (default: [64,
                        64, 64])
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.001)
  --exp_decay EXP_DECAY
                        Exponential decay factor (default: 0.9)
  --lr-scheduler {constant,cosine_warm_restarts,exponential}
                        Learning rate scheduler (default: constant)
  --cosine-first-steps COSINE_FIRST_STEPS
                        Length of a cosine decay cycle in steps, only with
                        cosine_annealing scheduler (default: 500)
  --optimizer {sgd,adam,adamw}
                        Optimizer (default: adam)
  --clip-norm [BOOLEAN]
                        Use gradient clipping (default: False)
  --clip-norm-max CLIP_NORM_MAX
                        Max threshold for global norm clipping (default: 1.0)
  --model {nnunet,unet,attn_unet,unetr}
  --use-res-block [BOOLEAN]
                        Use residual blocks for nnUNet or UNet (default:
                        False)
  --pocket [BOOLEAN]    Use pocket version of network (default: False)
  --depth DEPTH         Depth of U-Net or similar architecture (default: None)
  --init-filters INIT_FILTERS
                        Number of filters to start network (default: 32)
  --deep-supervision [BOOLEAN]
                        Use deep supervision (default: False)
  --deep-supervision-heads DEEP_SUPERVISION_HEADS
                        Number of deep supervision heads (default: 2)
  --vae-reg [BOOLEAN]   Use VAE regularization (default: False)
  --vae-penalty VAE_PENALTY
                        Weight for VAE regularization loss (default: 0.1)
  --l2-reg [BOOLEAN]    Use L2 regularization (default: False)
  --l2-penalty L2_PENALTY
                        L2 penalty (default: 1e-05)
  --l1-reg [BOOLEAN]    Use L1 regularization (default: False)
  --l1-penalty L1_PENALTY
                        L1 penalty (default: 1e-05)
  --oversampling OVERSAMPLING
                        Probability of crop centered on foreground voxel
                        (default: 0.4)
  --no-preprocess [BOOLEAN]
                        Turn off preprocessing (default: False)
  --use-n4-bias-correction [BOOLEAN]
                        Use N4 bias field correction (only for MR images)
                        (default: False)
  --use-config-class-weights [BOOLEAN]
                        Use class weights in config file (default: False)
  --class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        Specify class weights (default: None)
  --loss {dice_ce,dice,gdl,gdl_ce}
                        Loss function for training (default: dice_ce)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between scans during sliding window
                        inference (default: 0.25)
  --blend-mode {constant,gaussian}
                        How to blend output of overlapping windows (default:
                        constant)
  --postprocess [BOOLEAN]
                        Run post processing on MIST output (default: False)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --folds FOLDS [FOLDS ...]
                        Which folds to run (default: [0, 1, 2, 3, 4])
  --epochs EPOCHS       Number of epochs (default: 300)
  --steps-per-epoch STEPS_PER_EPOCH
                        Steps per epoch. By default ceil(training_dataset_size
                        / batch_size / gpus) (default: None)
  --output-std [BOOLEAN]
                        Output standard deviation for ensemble predictions
                        (default: False)
```

## Inference
### Overview
Once the MIST pipeline finishes running, the saved models are in the ```models/``` directory in 
the specified ```--results``` folder. Additionally, the MIST pipeline outputs a file 
called ```config.json``` at the top level of the user-defined ```--results``` folder. 
We can run the MIST pipeline on new test data using the following command.

```
python predict.py --models /path/to/entire/models/directory \  *** PLEASE PASS ENTIRE MODELS DIRECTORY HERE ***
--config /path/to/config.json \
--data /path/to/new-data.csv/or/new-data.json \
--output /path/to/save/new/predictions
```

MIST supports two formats for test data: CSV and JSON. For CSV formatted data, the CSV file must,
at a minimum, have an ```id``` column with the new patient IDs and one column for each image type.
A column for the ```mask``` is allowed if you want to run the evaluation portion of the pipeline
(see below). For example, for the BraTS dataset, our CSV header would look like the following.

| id         | mask (optional) | t1               | t2               | tc               | fl               |
|------------|-----------------|------------------|------------------|------------------|------------------|
| Patient ID | Path to mask    | Path to t1 image | Path to t2 image | Path to tc image | Path to fl image |

Similarly, for JSON formatted data, we would have the following.
```
{
    "Patient ID": {
        "mask": "Path to mask", *** (optional) ***
        "t1": "Path to t1 image",
        "t2": "Path to t2 image",
        "tc": "Path to tc image", 
        "fl": "Path to fl image"
    }
}
```
Note that the order of the image types should be the same as the order given in the JSON file
input to the MIST training pipeline.

### Advanced Usage
A complete list of available options and their descriptions, can be accessed by 
using ```-h``` or ```--help``` command-line option, for example:
```
usage: predict.py [-h] [--models MODELS] [--config CONFIG] [--data DATA]
                  [--output OUTPUT] [--fast [BOOLEAN]] [--gpu GPU]
                  [--sw-overlap SW_OVERLAP] [--blend-mode {constant,gaussian}]
                  [--tta [BOOLEAN]] [--no-preprocess [BOOLEAN]]
                  [--output_std [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS       Directory containing saved models (default: None)
  --config CONFIG       Path and name of config.json file from results of MIST
                        pipeline (default: None)
  --data DATA           CSV or JSON file containing paths to data (default:
                        None)
  --output OUTPUT       Directory to save predictions (default: None)
  --fast [BOOLEAN]      Use only one model for prediction to speed up
                        inference time (default: False)
  --gpu GPU             GPU id to run inference on (default: 0)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between scans during sliding window
                        inference (default: 0.25)
  --blend-mode {constant,gaussian}
                        How to blend output of overlapping windows (default:
                        constant)
  --tta [BOOLEAN]       Use test time augmentation (default: False)
  --no-preprocess [BOOLEAN]
                        Turn off preprocessing (default: False)
  --output_std [BOOLEAN]
                        Outputs standard deviation image (default: False)
```

## Evaluating
If you have a set of predictions that you want to evaluate, then run the ```eval_preds.py``` script
as follows:

```
python eval_preds.py --data-json /path/to/dataset.json \
--paths /path/to/original-paths.csv/or/original-paths.json \
--preds-dir /path/to/directory/with/predictions \ 
--output-csv /path/to/evaluation/output.csv
```

A full list of options for the evaluation scirpt is below.

```
usage: eval_preds.py [-h] [--data-json DATA_JSON] [--paths PATHS]
                     [--preds-dir PREDS_DIR] [--output-csv OUTPUT_CSV]

optional arguments:
  -h, --help            show this help message and exit
  --data-json DATA_JSON
                        Path to dataset JSON file (default: None)
  --paths PATHS         Path to CSV or JSON file with original mask/data
                        (default: None)
  --preds-dir PREDS_DIR
                        Path to directory containing predictions (default:
                        None)
  --output-csv OUTPUT_CSV
                        Path to CSV containing evaluation results (default:
                        None)
```

## MSD and CSV Formatted Data
If your dataset in the MSD or CSV format, then you can use the ```convert_to_mist.py``` script to convert your dataset to the standard MIST format described above. For example, if your dataset is in the MSD format, then use the following command to convert it to the standard MSD format.
```
python convert_to_mist.py --msd-source /path/to/msd/dataset \
--dest /path/to/mist/format/dataset
```
This will reformat the MSD dataset to the MIST format and produce the corresponding JSON file to run the MIST pipeline.

If your dataset is given as a CSV file with ```id```, ```mask```, and columns for each image type, then use the following command to convert it to a MIST-compatible dataset.
```
python convert_to_mist.py --format csv \
--train-csv /path/to/csv/dataset.csv \
--dest /path/to/mist/format/dataset
```
Like the MSD formatted dataset, this command will reformat the CSV dataset to a MIST-compatible 
one but will require the user to fill in details in its corresponding JSON file.

A complete list of available options and their descriptions, can be accessed by 
using ```-h``` or ```--help``` command-line option, for example:
```
usage: convert_to_mist.py [-h] [--format {msd,csv}] [--msd-source MSD_SOURCE]
                          [--train-csv TRAIN_CSV] [--test-csv TEST_CSV]
                          [--dest DEST]

optional arguments:
  -h, --help            show this help message and exit
  --format {msd,csv}    Format of dataset to be converted (default: msd)
  --msd-source MSD_SOURCE
                        Directory containing MSD formatted dataset (default:
                        None)
  --train-csv TRAIN_CSV
                        Path to and name of csv containing training ids, mask,
                        and images (default: None)
  --test-csv TEST_CSV   Path to and name of csv containing test ids and images
                        (default: None)
  --dest DEST           Directory to save converted, MIST formatted dataset
                        (default: None)
```
