# Medical Imaging Segmentation Toolkit
Please cite the following if you use this code for your work:

> A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on Medical Imaging, doi: 10.1109/TMI.2022.3224873.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
    * [Requirements](#requirements)
    * [Data Format](#data-format)
    * [Getting Started](#getting-started)
    * [Advanced Usage](#advanced-usage)
- [Inference](#inference)
    * [Overview](#overview)
    * [Advanced Usage](#advanced-usage)
- [MSD and CSV Formatted Data](#msd-and-csv-formatted-data)

## Overview
The Medical Imaging Segmentation Toolkit (MIST) is a simple 3D medical imaging segmentation framework for TensorFlow 2. MIST allows researchers to quickly test a variety of deep learning architectures to achieve state-of-the-art performance. The following architectures are implemented on MIST:

* nnUNet
* U-Net
* ResNet
* DenseNet
* HRNet

The following features are supported by MIST: 

| Feature                                                                                           | MIST |
|---------------------------------------------------------------------------------------------------|------|
| [NVIDIA Data Loading Library (DALI)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)  | Yes  |
| [Automatic mixed precision (AMP)](https://www.tensorflow.org/guide/mixed_precision)               | Yes  |
| [Accelerated Linear Algebra (XLA)](https://www.tensorflow.org/xla)                                | Yes  |
| [Multi-GPU training with tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute) | Yes  |

Support for PyTorch is coming soon.
    
## Setup
### Requirements
We include a Dockerfile in this repository that pulls NVIDIA's TensorFlow 2 NGC container and installs the necessary dependencies. Please make sure to have the following components ready.

* [Docker](https://www.docker.com/)
* [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* GPU with sufficient memory and compute capacity to handle 3D image segmentation

### Data Format
The MIST pipeline, by default, assumes that your train and test data directories are set up in the following manner.
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
Please note that the naming convention here is for this example only. There is no specific naming convention for the files within your dataset. However, we impose that the naming of the images in each patient directory is consistent or that each type of image is identifiable by a list of identifier strings.

MIST offers support for MSD and CSV formatted datasets. For more details, please see [MSD and CSV Formatted Data](#msd-and-csv-formatted-data).

Once your dataset is in the correct format, the final step is to prepare a small JSON file containing the details of the dataset. We specifically ask for the following key-value pairs.

| Key                 | Value                                                                                                                                                                  |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```task```          | Name of task (i.e., brats, lits, etc.).                                                                                                                                |
| ```modality```      | Options are ``ct``, ``mr``, or ``other``.                                                                                                                              |
| ```train-data```    | Path to training data.                                                                                                                                                 |
| ```test-data```     | Path to test data (optional).                                                                                                                                          |
| ```mask```          | List containing identifying strings for mask or ground truth images in dataset.                                                                                        |
| ```images```        | Dictionary where each key is an image type (i.e., T1, T2, CT, etc.) and each value is a list containing identifying strings for that image type.                       |
| ```labels```        | List of labels in dataset (must include 0).                                                                                                                            |
| ```final_classes``` | Dictionary where each key is the name of the final segmentation class (i.e., WT, ET, TC for BraTS) and each value is a list of the labels corresponding to that class. |

An example of the BraTS dataset is given below.

```
{
    "task": "brats2020",
    "modality": "mr",
    "train-data": "/workspace/data/brats_2020/raw/train",
    "test-data": "/workspace/data/brats_2020/raw/validation",
    "mask": [
        "seg.nii.gz"
      ],
    "images": {
        "t1": [
          "t1.nii.gz"
        ],
        "t2": [
          "t2.nii.gz"
        ],
        "tc": [
          "t1ce.nii.gz"
        ],
        "fl": [
          "flair.nii.gz"
        ]
      },
    "labels": [
        0,
        1,
        2,
        4
      ],
    "final_classes": {
        "WT": [
          1,
          2,
          4
        ],
        "TC": [
          1,
          4
        ],
        "ET": [
          4
        ]
      }
}
```
Several other examples are provided in the ```examples``` directory.


### Getting Started
To start running the MIST pipeline, first clone this repository.

```
git clone https://github.com/aecelaya/MIST.git
cd MIST
```

Next we use Dockerfile in this repository to build a Docker image named ```mist```.
```
docker build -t mist .
```

Once the container is ready, we launch an interactive session with the following command. Notice that we mount a directory to the ```/workspace``` directory in the container. The idea is to mount the local directory containing our data to the workspace directory. Please modify this command to match your system.
```
docker run --rm -it -u $(id -u):$(id -g) --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /rsrch1/ip/aecelaya/:/workspace mist
```

Once inside the container, we can run the MIST pipeline (with all default values) with the following command.
```
python main.py --data examples/brats.json --processed-data /workspace/data/mist-examples/brats/numpy/ --results /workspace/data/mist-examples/brats/results
```

If your system can support AMP, then we strongly recommend running the MIST pipeline with the ```--amp``` and ```--xla``` flags.


### Advanced usage
To see the complete list of available options and their descriptions, use the -h or --help command-line option, for example:

```
python main.py --help
```

The following output is printed when running the command above:
```
usage: main.py [-h] [--exec-mode {all,analyze,preprocess,train}] [--data DATA]
               [--gpus GPUS [GPUS ...]] [--seed SEED] [--tta [BOOLEAN]] [--results RESULTS]
               [--processed-data PROCESSED_DATA] [--config CONFIG] [--paths PATHS] [--amp [BOOLEAN]]
               [--xla [BOOLEAN]] [--batch-size BATCH_SIZE] [--patch-size PATCH_SIZE [PATCH_SIZE ...]]
               [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
               [--lr-scheduler {none,poly,cosine_annealing}] [--end-learning-rate END_LEARNING_RATE]
               [--cosine-annealing-first-cycle-steps COSINE_ANNEALING_FIRST_CYCLE_STEPS]
               [--cosine-annealing-peak-decay COSINE_ANNEALING_PEAK_DECAY] [--optimizer {sgd,adam}]
               [--lookahead [BOOLEAN]] [--clip-norm [BOOLEAN]] [--clip-norm-max CLIP_NORM_MAX]
               [--model {nnunet,unet,resnet,densenet,hrnet}] [--depth DEPTH]
               [--init-filters INIT_FILTERS] [--deep-supervision [BOOLEAN]] [--pocket [BOOLEAN]]
               [--oversampling OVERSAMPLING] [--use-precomputed-weights [BOOLEAN]]
               [--class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]] [--loss {dice_ce,dice,gdl,gdl_ce}]
               [--sw-overlap SW_OVERLAP] [--blend-mode {gaussian,constant}]
               [--post-no-morph [BOOLEAN]] [--post-no-largest [BOOLEAN]] [--nfolds NFOLDS]
               [--folds FOLDS [FOLDS ...]] [--epochs EPOCHS] [--steps-per-epoch STEPS_PER_EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --exec-mode {all,analyze,preprocess,train}
                        Run all of the MIST pipeline or an individual component (default: all)
  --data DATA           Path to dataset json file (default: None)
  --gpus GPUS [GPUS ...]
                        Which gpu(s) to use (default: [0])
  --seed SEED           Random seed (default: None)
  --tta [BOOLEAN]       Enable test time augmentation (default: False)
  --results RESULTS     Path to output of MIST pipeline (default: /mist/results)
  --processed-data PROCESSED_DATA
                        Path to save input parameters for MIST pipeline (default: /mist/numpy)
  --config CONFIG       Path to config.json file (default: None)
  --paths PATHS         Path to csv containing raw data paths (default: None)
  --amp [BOOLEAN]       Enable automatic mixed precision (recommended) (default: False)
  --xla [BOOLEAN]       Enable XLA compiling (default: False)
  --batch-size BATCH_SIZE
                        Batch size (default: 2)
  --patch-size PATCH_SIZE [PATCH_SIZE ...]
                        Height, width, and depth of patch size to use for cropping (default: None)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.0003)
  --momentum MOMENTUM   Momentum factor (SGD only) (default: 0.99)
  --lr-scheduler {none,poly,cosine_annealing}
                        Learning rate scheduler (default: none)
  --end-learning-rate END_LEARNING_RATE
                        End learning rate for poly scheduler and decrease on plateau (default: 8e-05)
  --cosine-annealing-first-cycle-steps COSINE_ANNEALING_FIRST_CYCLE_STEPS
                        Length of a cosine decay cycle in steps, only with cosine_annealing scheduler
                        (default: 512)
  --cosine-annealing-peak-decay COSINE_ANNEALING_PEAK_DECAY
                        Multiplier reducing initial learning rate for cosine annealing (default:
                        0.95)
  --optimizer {sgd,adam}
                        Optimizer (default: adam)
  --lookahead [BOOLEAN]
                        Use Lookahead with the optimizer (default: False)
  --clip-norm [BOOLEAN]
                        Use gradient clipping (default: False)
  --clip-norm-max CLIP_NORM_MAX
                        Max threshold for global norm clipping (default: 1.0)
  --model {nnunet,unet,resnet,densenet,hrnet}
  --depth DEPTH         Depth of U-Net (default: None)
  --init-filters INIT_FILTERS
                        Number of filters to start network (default: 32)
  --deep-supervision [BOOLEAN]
                        Use deep supervision (default: False)
  --pocket [BOOLEAN]    Use pocket version of network (default: False)
  --oversampling OVERSAMPLING
                        Probability of crop centered on foreground voxel (default: 0.4)
  --use-precomputed-weights [BOOLEAN]
                        Use precomputed class weights (default: False)
  --class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        Specify class weights (default: None)
  --loss {dice_ce,dice,gdl,gdl_ce}
                        Loss function for training (default: dice_ce)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between scans during sliding window inference (default:
                        0.5)
  --blend-mode {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --post-no-morph [BOOLEAN]
                        Do not try morphological smoothing for postprocessing (default: False)
  --post-no-largest [BOOLEAN]
                        Do not run connected components analysis for postprocessing (default: False)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --folds FOLDS [FOLDS ...]
                        Which folds to run (default: [0, 1, 2, 3, 4])
  --epochs EPOCHS       Number of epochs (default: 300)
  --steps-per-epoch STEPS_PER_EPOCH
                        Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)
                        (default: None)

```

## Inference
### Overview
Once the MIST pipeline finishes running, the saved models are in the ```models``` directory in the specified ```--results``` folder. Additionally, the MIST pipeline outputs a file called ```config.json``` at the top level of the user-defined ```--results``` folder. We can run the MIST pipeline on new test data using the following command.

```
python predict.py --models /workspace/data/mist-examples/brats/results/models --config /workspace/data/mist-examples/brats/results/config.json --data /workspace/data/mist-examples/brats/new-data.csv --output /workspace/data/mist-examples/brats/new-data-predictions
```

MIST supports two formats for test data: CSV and JSON. For CSV formatted data, the CSV file must have an ```id`` column with the new patient IDs and one column for each image type. For example, for the BraTS dataset, our CSV header would look like the following.

| id         | t1                    | t2                    | tc                    | fl                    |
|------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Patient ID | Full path to t1 image | Full path to t2 image | Full path to tc image | Full path to fl image | 

Similaryly, for JSON fomratted data, we would have the following.
```
{
    "Patient ID": {
        "t1": "Full path to t1 image",
        "t2": "Full path to t2 image",
        "tc": "Full path to tc image", 
        "fl": "full path to fl image"
    }
}
```
Note that the order of the image types should be the same as the order given in the JSON file input to the MIST training pipeline.

### Advanced Usage
To see the complete list of available options and their descriptions, use the -h or --help command-line option, for example:

```
python predict.py --help
```

The following output is printed when running the command above:

```
usage: predict.py [-h] [--models MODELS] [--config CONFIG] [--data DATA] [--output OUTPUT]
                  [--fast [BOOLEAN]] [--gpu GPU] [--amp [BOOLEAN]] [--xla [BOOLEAN]]
                  [--sw-overlap SW_OVERLAP] [--blend-mode {gaussian,constant}] [--tta [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --models MODELS       Directory containing saved models (default: None)
  --config CONFIG       Path and name of config.json file from results of MIST pipeline (default: None)
  --data DATA           CSV or JSON file containing paths to data (default: None)
  --output OUTPUT       Directory to save predictions (default: None)
  --fast [BOOLEAN]      Use only one model for prediction to speed up inference time (default: False)
  --gpu GPU             GPU id to run inference on (default: 0)
  --amp [BOOLEAN]       Use automatic mixed precision (default: False)
  --xla [BOOLEAN]       Use XLA (default: False)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between scans during sliding window inference (default: 0.5)
  --blend-mode {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --tta [BOOLEAN]       Use test time augmentation (default: False)
```

## MSD and CSV Formatted Data
If your dataset in the MSD or CSV format, then you can use the ```convert_to_mist.py``` script to convert your dataset to the standard MIST format described above. For example, if your dataset is in the MSD format, then use the following command to convert it to the standard MSD format.
```
python convert_to_mist.py --msd-source /workspace/path/to/msd/dataset --dest /workspace/path/to/mist/format/dataset
```
This will reformat the MSD dataset to the MIST format and produce the corresponding JSON file to run the MIST pipeline.

If your dataset is given as a CSV file with ```id```, ```mask```, and columns for each image type, then use the following command to convert it to a MIST-compatible dataset.
```
python convert_to_mist.py --format csv --train-csv /workspace/path/to/csv/dataset.csv --dest /workspace/path/to/mist/format/dataset
```
Like the MSD formatted dataset, this command will reformat the CSV dataset to a MIST-compatible one but will require the user to fill in details in its corresponding JSON file.

To see the complete list of available options and their descriptions, use the -h or --help command-line option, for example:

```
python convert_to_mist.py --help
```

The following output is printed when running the command above:

```
usage: convert_to_mist.py [-h] [--format {msd,csv}] [--msd-source MSD_SOURCE] [--train-csv TRAIN_CSV]
                          [--test-csv TEST_CSV] [--dest DEST]

optional arguments:
  -h, --help            show this help message and exit
  --format {msd,csv}    Format of dataset to be converted (default: msd)
  --msd-source MSD_SOURCE
                        Directory containing MSD formatted dataset (default: None)
  --train-csv TRAIN_CSV
                        Path to and name of csv containing training ids, mask, and images (default:
                        None)
  --test-csv TEST_CSV   Path to and name of csv containing test ids and images (default: None)
  --dest DEST           Directory to save converted, MIST formatted dataset (default: None)
```
