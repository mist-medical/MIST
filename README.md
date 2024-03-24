# Medical Imaging Segmentation Toolkit

## Overview
The Medical Imaging Segmentation Toolkit (MIST) is a simple, fully automated 3D medical imaging segmentation 
framework. MIST allows researchers to quickly set up, train, and test a variety of deep learning models for 3D 
medical imaging segmentation. The following architectures are implemented on MIST:

* nnUNet
* U-Net
* FMG-Net
* W-Net
* Attention U-Net
* UNETR

The following features are supported by MIST: 

* [NVIDIA Data Loading Library (DALI)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)
* [Automatic mixed precision (AMP)](https://www.tensorflow.org/guide/mixed_precision)
* [Multi-GPU training with DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

Please cite the following papers if you use this code for your work:

> [A. Celaya et al., "PocketNet: A Smaller Neural Network For Medical Image Analysis," in IEEE Transactions on 
> Medical Imaging, doi: 10.1109/TMI.2022.3224873.](https://ieeexplore.ieee.org/document/9964128)

> [A. Celaya et al., "FMG-Net and W-Net: Multigrid Inspired Deep Learning Architectures For Medical Imaging Segmentation", in
> Proceedings of LatinX in AI (LXAI) Research Workshop @ NeurIPS 2023, doi: 10.52591/lxai202312104](https://research.latinxinai.org/papers/neurips/2023/pdf/Adrian_Celaya.pdf)


## What's New
* March 2024 - Simplify and decouple postprocessing from main MIST pipeline.
* March 2024 - Support for using transfer learning with pretrained MIST models is now available.
* March 2024 - Boundary-based loss functions are now available.
* Feb. 2024 - MIST is now available as PyPI package and as a Docker image on DockerHub.
* Feb. 2024 - Major improvements to the analysis, preprocessing, and postprocessing pipelines, 
and new network architectures like UNETR added.
* Feb. 2024 - We have moved the TensorFlow version of MIST to [mist-tf](https://github.com/aecelaya/mist-tf).

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
    * [Install](#install)
    * [Data Format](#data-format)
- [Getting Started](#getting-started)
- [Output](#output)
- [Advanced Usage](#advanced-usage)

 
## Setup
MIST assumes that your system as at least one GPU and sufficient memory to handle 3D medical images.

### Install
To install the latest version of MIST as an out-of-the-box segmentation pipeline, use 
```
pip install mist-medical
```

If you want to install MIST and customize the underlying code (i.e., add a loss function or new architecture), 
then clone the MIST repo and install as follows:
```
git clone https://github.com/aecelaya/MIST.git
cd MIST
pip install -e .
```

### Data Format
The MIST pipeline assumes that your train and test data directories are set up in the following structure.
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

## Getting Started
The MIST pipeline consists of three stages:
1. Analyze - Gathering parameters about the dataset like target spacing, normalization parameters, etc. This produces a ```config.json``` file, which will be used for the rest of the pipeline.

2. Preprocess - Use the learned parameters from the analysis phase of the MIST pipeline to preprocess the data (i.e., reorient, resample, etc.) and convert it to numpy files.

3. Train - Train on preprocessed data using a five-fold cross validation to produce a final set of models for inference.

Additionally, MIST provides auxiliary commands that handle postprocessing, test-time prediction, evaluating a given set of predictions, and converting other datasets to the MIST format.

When you install the MIST package, the following commands are included:

* ```mist_run_all```: This command runs the entire MIST pipeline and requires the following arguments:
	- ```--data```: The full path to your dataset JSON file
	- ```--numpy```: The full path to the directory to save preprocessed Numpy files
	- ```--results```: The full path to the directory to save the output of the MIST pipeline
    - ```--amp```: Optional, but highly recommended, if your system supports AMP
    - ```--pocket```: Optional, but highly recommended, for using smaller, but just as accurate networks
	
* ```mist_analyze```: This command runs only the analysis portion of the pipeline and requires the following arguments:
	- ```--data```: The full path to your dataset JSON file
	- ```--results```: The full path to the directory to save the output of the MIST pipeline

* ```mist_preprocess```: This command runs only the preprocessing pipeline and assumes that the output of the analysis pipeline (or a modified version of it) are in the ```--results``` folder. This command requires the following arguments:
	- ```--data```: The full path to your dataset JSON file
	- ```--numpy```: The full path to the directory to save preprocessed Numpy files
	- ```--results```: The full path to the directory to save the output of the MIST pipeline

* ```mist_train```: This command runs only the training pipeline and assumes that the results of the previous two commands are in the ```--results``` and ```--numpy``` folders. This command requires the following arguments:
	- ```--data```: The full path to your dataset JSON file
	- ```--numpy```: The full path to the directory to save preprocessed Numpy files
	- ```--results```: The full path to the directory to save the output of the MIST pipeline
    - ```--amp```: Optional, but highly recommended, if your system supports AMP
    - ```--pocket```: Optional, but highly recommended, for using smaller, but just as accurate networks

* ```mist_predict```: This command runs test time inference on a given set of test data given as either a JSON or CSV file. This command is a stand alone command that can be used anytime outside of the MIST pipeline. To run this command, we need the following arguments:
	- ```--models```: The full path to the ```models``` folder in the output of the MIST pipeline
	- ```--config```: The full path to the directory to the ```config.json``` file in the output of the MIST pipeline
	- ```--data```: The full path to the JSON or CSV file, which contains the path to your test data
	- ```--output```: The full path to the directory to save the predictions

  For CSV formatted data, the CSV file must, at a minimum, have an ```id``` column with the new patient IDs and one column for each image type. A column for the ```mask``` is allowed if you want to run the evaluation portion of the pipeline. For example, for the BraTS dataset, our CSV header would look like the following.

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
  
* ```mist_evaluate```: This command is used to evaluate a set of predictions and requires the following arguments:
	- ```--paths```: The full paths to the ground truth masks formatted in either CSV or JSON format (see ```mist_predict```)
	- ```--config```: The full path to the directory to the ```config.json``` file in the output of the MIST pipeline
	- ```--preds-dir```: The full path to the directory where the predictions are saved
	- ```--output-csv```: The full path to the CSV file where the results will be saved

* ```mist_convert_dataset```: This command converts either MSD or CSV formatted datasets into MIST formatted data. The following arguments are required for this command:
	- ```--format```: The format of the given dataset, which can be either ```msd``` or ```csv```
	- ```--msd-source```: The full path to the MSD dataset, if that is what you are converting
	- ```--csv-source```: The full path to the CSV dataset, if that is what you are using
	- ```--dest```: The full path to the new MIST formatted dataset
	
	Note that the CSV file should mirror the CSV format shown in the ```mist_predict``` command. Additionally, this command will reformat the CSV dataset to a MIST-compatible one but will require the user to fill in details in its corresponding dataset JSON file.

* ```mist_postprocess```: This command runs the postprocessing pipeline on the original predictions produced by the main MIST pipeline. The following arguments are required for this command:
	- ```--base-results```: The full path to the results of the MIST pipeline
	- ```--output```: The full path to the output of the postprocessing pipeline
	- ```--apply-to-labels```: Labels to apply postprocessing transformations, default is ```[-1]``` or all labels combined
	- At least one postprocessing transformation:
		- ```--remove-small-objects```: Removes small objects of size less than or equal to ```--small-object-threshold```, which is 64 by default
		- ```--top-k-cc```: Only keep the ```--top-k``` (by default 2) connected components
		- ```--fill-holes```: Fill holes with ```--fill-label``` (you need to specify this)
	- ```--update-config```: Optional but important to note; set this if you want the config file to be updated if postprocessing, on average, improves results by at least 5%
	- ```--morph-cleanup```: Optional but recommended for ```--top-k-cc```; applies a morphological erosion prior to selecting the top k connected components and then uses a dilation after the connected components are selected
	
  WARNING: ```mist_postprocess``` can be very slow if your images are extremely large and/or you have a lot of labels (which will slow down evaluation).

### Docker
The MIST package is also available as a Docker image. Start by pulling the ```mistmedical/mist``` image from DockerHub:
```
docker pull mistmedical/mist:latest
```

Use the following command to run an interactive Docker container with the MIST package:
```
docker run --rm -it -u $(id -u):$(id -g) --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /your/working/directory/:/workspace mist
```

From there, you can run any of the commands described above inside the Docker container. Additionally, you can use the Docker 
entrypoint command to run any of the MIST scripts.

## Output
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

| Directory/File        | Description                                                                                                                                                    |
|:----------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```logs/```           | TensorBoard logs for each fold.                                                                                                                                |
| ```models/```         | PyTorch models for each fold and ```model_config.json```, which stores the settings (i.e., architecture, patch size) to load the model from the saved weights. |
| ```predictions/```    | Raw and postprocessed predictions from five fold cross validation and test set (if specified).                                                                 |
| ```config.json```     | JSON file containing all of the attributes of the dataset (i.e., target spacing, crop to foreground, etc.).                                                    |
| ```results.csv```     | CSV file with the Dice, 95th percentile Hausdorff, and average surface distance scores for all of the folds.                                                   |
| ```train_paths.csv``` | CSV file starting with columns ```id``` and ```fold``` specifying the patient ID and which fold they belong to, and paths to the mask and images.              |
| ```test_paths.csv```  | Same as ```train_paths.csv```, but for the test set if it's given.                                                                                             |
| ```fg_bboxes.csv```   | CSV file containing information about the bounding box around the foreground for each image.                                                                   |

## Advanced Usage
All MIST commands come with ```--help``` or ```-h``` option, which allows you to see all the available settings/arguments for that command.

For the ```mist_run_all```, ```mist_analyze```, ```mist_preprocess```, and ```mist_train``` commands, here is a complete list of the available arguments:
```
usage: mist_run_all [-h] [--exec-mode {all,analyze,preprocess,train}] [--data DATA]
                    [--gpus GPUS [GPUS ...]] [--num-workers NUM_WORKERS]
                    [--master-port MASTER_PORT] [--seed_val SEED_VAL]
                    [--tta [BOOLEAN]] [--results RESULTS] [--numpy NUMPY]
                    [--amp [BOOLEAN]] [--batch-size BATCH_SIZE]
                    [--patch-size PATCH_SIZE [PATCH_SIZE ...]]
                    [--max-patch-size MAX_PATCH_SIZE [MAX_PATCH_SIZE ...]]
                    [--val-percent VAL_PERCENT] [--learning-rate LEARNING_RATE]
                    [--exp_decay EXP_DECAY]
                    [--lr-scheduler {constant,cosine_warm_restarts,exponential}]
                    [--cosine-first-steps COSINE_FIRST_STEPS]
                    [--optimizer {sgd,adam,adamw}] [--clip-norm [BOOLEAN]]
                    [--clip-norm-max CLIP_NORM_MAX]
                    [--model {nnunet,unet,fmgnet,wnet,attn_unet,unetr,pretrained}]
                    [--pretrained-model-path PRETRAINED_MODEL_PATH]
                    [--use-res-block [BOOLEAN]] [--pocket [BOOLEAN]] [--depth DEPTH]
                    [--deep-supervision [BOOLEAN]]
                    [--deep-supervision-heads DEEP_SUPERVISION_HEADS]
                    [--vae-reg [BOOLEAN]] [--vae-penalty VAE_PENALTY]
                    [--l2-reg [BOOLEAN]] [--l2-penalty L2_PENALTY]
                    [--l1-reg [BOOLEAN]] [--l1-penalty L1_PENALTY]
                    [--oversampling OVERSAMPLING] [--no-preprocess [BOOLEAN]]
                    [--use-n4-bias-correction [BOOLEAN]]
                    [--use-config-class-weights [BOOLEAN]] [--use-dtms [BOOLEAN]]
                    [--class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]]
                    [--loss {dice_ce,dice,gdl,gdl_ce,bl,hdl,gsl}]
                    [--boundary-loss-schedule {constant,linear,step,cosine}]
                    [--loss-schedule-constant LOSS_SCHEDULE_CONSTANT]
                    [--linear-schedule-pause LINEAR_SCHEDULE_PAUSE]
                    [--step-schedule-step-length STEP_SCHEDULE_STEP_LENGTH]
                    [--sw-overlap SW_OVERLAP] [--val-sw-overlap VAL_SW_OVERLAP]
                    [--blend-mode {gaussian,constant}] [--nfolds NFOLDS]
                    [--folds FOLDS [FOLDS ...]] [--epochs EPOCHS]
                    [--steps-per-epoch STEPS_PER_EPOCH]
                    [--use-native-spacing [BOOLEAN]] [--output-std [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --exec-mode {all,analyze,preprocess,train}
                        Run all of the MIST pipeline or an individual component
                        (default: all)
  --data DATA           Path to dataset json file (default: None)
  --gpus GPUS [GPUS ...]
                        Which gpu(s) to use, defaults to all available GPUs
                        (default: [-1])
  --num-workers NUM_WORKERS
                        Number of workers to use for data loading (default: 8)
  --master-port MASTER_PORT
                        Master port for multi-gpu training (default: 12355)
  --seed_val SEED_VAL   Random seed (default: 42)
  --tta [BOOLEAN]       Enable test time augmentation (default: False)
  --results RESULTS     Path to output of MIST pipeline (default: None)
  --numpy NUMPY         Path to save preprocessed numpy data (default: None)
  --amp [BOOLEAN]       Enable automatic mixed precision (recommended) (default:
                        False)
  --batch-size BATCH_SIZE
                        Batch size (default: None)
  --patch-size PATCH_SIZE [PATCH_SIZE ...]
                        Height, width, and depth of patch size (default: None)
  --max-patch-size MAX_PATCH_SIZE [MAX_PATCH_SIZE ...]
                        Max patch size (default: [256, 256, 256])
  --val-percent VAL_PERCENT
                        Percentage of training data used for validation (default:
                        0.1)
  --learning-rate LEARNING_RATE
                        Learning rate (default: 0.0003)
  --exp_decay EXP_DECAY
                        Exponential decay factor (default: 0.9999)
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
  --model {nnunet,unet,fmgnet,wnet,attn_unet,unetr,pretrained}
  --pretrained-model-path PRETRAINED_MODEL_PATH
                        Full path to pretrained mist models directory (default:
                        None)
  --use-res-block [BOOLEAN]
                        Use residual blocks for nnUNet or UNet (default: False)
  --pocket [BOOLEAN]    Use pocket version of network (default: False)
  --depth DEPTH         Depth of U-Net or similar architecture (default: None)
  --deep-supervision [BOOLEAN]
                        Use deep supervision (default: False)
  --deep-supervision-heads DEEP_SUPERVISION_HEADS
                        Number of deep supervision heads (default: 2)
  --vae-reg [BOOLEAN]   Use VAE regularization (default: False)
  --vae-penalty VAE_PENALTY
                        Weight for VAE regularization loss (default: 0.01)
  --l2-reg [BOOLEAN]    Use L2 regularization (default: False)
  --l2-penalty L2_PENALTY
                        L2 penalty (default: 1e-05)
  --l1-reg [BOOLEAN]    Use L1 regularization (default: False)
  --l1-penalty L1_PENALTY
                        L1 penalty (default: 1e-05)
  --oversampling OVERSAMPLING
                        Probability of crop centered on foreground voxel (default:
                        0.4)
  --no-preprocess [BOOLEAN]
                        Turn off preprocessing (default: False)
  --use-n4-bias-correction [BOOLEAN]
                        Use N4 bias field correction (only for MR images) (default:
                        False)
  --use-config-class-weights [BOOLEAN]
                        Use class weights in config file (default: False)
  --use-dtms [BOOLEAN]  Compute and use DTMs during training (default: False)
  --class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        Specify class weights (default: None)
  --loss {dice_ce,dice,gdl,gdl_ce,bl,hdl,gsl}
                        Loss function for training (default: dice_ce)
  --boundary-loss-schedule {constant,linear,step,cosine}
                        Weighting schedule for boundary losses (default: constant)
  --loss-schedule-constant LOSS_SCHEDULE_CONSTANT
                        Constant for fixed alpha schedule (default: 0.5)
  --linear-schedule-pause LINEAR_SCHEDULE_PAUSE
                        Number of epochs before linear alpha scheduler starts
                        (default: 5)
  --step-schedule-step-length STEP_SCHEDULE_STEP_LENGTH
                        Number of epochs before in each section of the step-wise
                        alpha scheduler (default: 5)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between patches during sliding window
                        inference at test time (default: 0.5)
  --val-sw-overlap VAL_SW_OVERLAP
                        Amount of overlap between patches during sliding window
                        inference during validation (default: 0.5)
  --blend-mode {gaussian,constant}
                        How to blend output of overlapping windows (default:
                        gaussian)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --folds FOLDS [FOLDS ...]
                        Which folds to run (default: [0, 1, 2, 3, 4])
  --epochs EPOCHS       Number of epochs (default: 1000)
  --steps-per-epoch STEPS_PER_EPOCH
                        Steps per epoch. By default ceil(training_dataset_size /
                        (batch_size * gpus) (default: None)
  --use-native-spacing [BOOLEAN]
                        Use native image spacing to compute Hausdorff distances
                        (default: False)
  --output-std [BOOLEAN]
                        Output standard deviation for ensemble predictions (default:
                        False)
```

Here are the available arguments for ```mist_postprocess```:
```
usage: mist_postprocess [-h] [--base-results BASE_RESULTS] [--output OUTPUT]
                        [--apply-to-labels APPLY_TO_LABELS [APPLY_TO_LABELS ...]]
                        [--remove-small-objects [BOOLEAN]]
                        [--top-k-cc [BOOLEAN]] [--morph-cleanup [BOOLEAN]]
                        [--fill-holes [BOOLEAN]] [--update-config [BOOLEAN]]
                        [--small-object-threshold SMALL_OBJECT_THRESHOLD]
                        [--top-k TOP_K]
                        [--morph-cleanup-iterations MORPH_CLEANUP_ITERATIONS]
                        [--fill-label FILL_LABEL]
                        [--use-native-spacing [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --base-results BASE_RESULTS
                        Path to original MIST results directory (default: None)
  --output OUTPUT       Path to save postprocessed results (default: None)
  --apply-to-labels APPLY_TO_LABELS [APPLY_TO_LABELS ...]
                        List of labels to apply postprocessing (default: [-1])
  --remove-small-objects [BOOLEAN]
                        Remove small objects (default: False)
  --top-k-cc [BOOLEAN]  Keep k largest connected components (CCs) (default:
                        False)
  --morph-cleanup [BOOLEAN]
                        Turn on morphological cleaning for k largest CCs
                        (default: False)
  --fill-holes [BOOLEAN]
                        Fill holes (default: False)
  --update-config [BOOLEAN]
                        Update config file if results improve with
                        postprocessing strategy (default: False)
  --small-object-threshold SMALL_OBJECT_THRESHOLD
                        Threshold size for small objects (default: 64)
  --top-k TOP_K         How many of top connected components to keep (default:
                        2)
  --morph-cleanup-iterations MORPH_CLEANUP_ITERATIONS
                        How many iterations for morphological cleaning (default:
                        2)
  --fill-label FILL_LABEL
                        Fill label for fill holes transformation (default: None)
  --use-native-spacing [BOOLEAN]
                        Use native image spacing to compute Hausdorff distances
                        (default: False)
```

Here are the available arguments for ```mist_predict```:
```
usage: mist_predict [-h] [--models MODELS] [--config CONFIG] [--data DATA]
                    [--output OUTPUT] [--fast [BOOLEAN]] [--gpu GPU]
                    [--sw-overlap SW_OVERLAP]
                    [--blend-mode {constant,gaussian}] [--tta [BOOLEAN]]
                    [--no-preprocess [BOOLEAN]] [--output_std [BOOLEAN]]

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

Here are the available arguments for ```mist_evaluate```:
```
usage: mist_evaluate [-h] [--config CONFIG] [--paths PATHS]
                     [--preds-dir PREDS_DIR] [--output-csv OUTPUT_CSV]
                     [--use-native-spacing [BOOLEAN]]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config.json file from MIST output (default:
                        None)
  --paths PATHS         Path to CSV or JSON file with original mask/data
                        (default: None)
  --preds-dir PREDS_DIR
                        Path to directory containing predictions (default:
                        None)
  --output-csv OUTPUT_CSV
                        Path to CSV containing evaluation results (default:
                        None)
  --use-native-spacing [BOOLEAN]
                        Use native image spacing to compute Hausdorff
                        distances (default: False)
```

Here are the available arguments for ```mist_convert_dataset```:
```
usage: mist_convert_dataset [-h] [--format {msd,csv}]
                            [--msd-source MSD_SOURCE] [--train-csv TRAIN_CSV]
                            [--test-csv TEST_CSV] [--dest DEST]

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
