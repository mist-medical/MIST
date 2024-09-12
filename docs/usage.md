Usage
===

## Overview

MIST is a command line tool. The MIST pipeline consists of three stages:

1. Analyze - Gathering parameters about the dataset like target spacing, normalization parameters, etc. This stage produces a ```config.json``` file, which will be used for the rest of the pipeline.

2. Preprocess - Use the learned parameters from the analysis phase of the MIST pipeline to preprocess the data (i.e., reorient, resample, etc.) and convert it to numpy files.

3. Train - Train on preprocessed data using a five-fold cross validation to produce a final set of models for inference.

Additionally, MIST provides auxiliary commands that handle postprocessing, test-time prediction, evaluating a given set of predictions, and converting other datasets to the MIST format.

To run the entire MIST pipeline (with all default arguments), use the ```mist_run_all``` command. This command and its arguments are described below:

* ```mist_run_all```: Runs the entire MIST pipeline
	- ```--data```: (required) Full path to your dataset JSON file
	- ```--numpy```: (required) Full path to the preprocessed numpy files
	- ```--results```: (required) Full path to the directory to save the output of the MIST pipeline
    - ```--amp```: (optional, recommended) Turns on automatic mixed precision (AMP)
    - ```--pocket```: (optional, recommended) Turns on use of pocket networks (except for Attention U-Net or UNETR)
    - ```--use-res-block```: (optional, recommended) Turns on residual blocks in architectures (except for Attention U-Net or UNETR)

!!! note
	The ```numpy``` and ```results``` directories will be created if they do not exist already.

Here is an example of how to use the ```mist_run_all``` command:
```
mist_run_all --data /path/to/dataset.json --numpy /path/to/preprocessed/data --results /path/to/results
```

See below for more details about each command and how to run them individually.

### Output
The output of the MIST pipeline has the following structure:
```text
results/
    logs/
    models/
    predictions/
    config.json
    results.csv
    train_paths.csv
    test_paths.csv (if test set is specified in dataset.json)
    fg_bboxes.csv
```

Here is a breakdown of what each file/directory contains.

| Directory/File        | Description                                                                                                                                            |
|:----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```logs/```           | TensorBoard logs for each fold.                                                                                                                        |
| ```models/```         | PyTorch models for each fold and ```model_config.json```.                                                                                              |
| ```predictions/```    | Predictions from five fold cross validation and test set (if specified).                                                                               |
| ```config.json```     | JSON file containing all of the attributes of the dataset.                                                                                             |
| ```results.csv```     | CSV file with the evaluation results from the five fold cross validation.                                                                              |
| ```train_paths.csv``` | CSV file starting with columns ```id``` and ```fold``` specifying the patient ID and which <br/>fold they belong to, and paths to the mask and images. |
| ```test_paths.csv```  | Same as ```train_paths.csv```, but for the test set if it's given.                                                                                     |
| ```fg_bboxes.csv```   | CSV file containing information about the bounding box around the foreground <br/>for each image.                                                           |


## Dataset Analysis
The first step in the MIST pipeline is the analysis step, which gathers parameters about the dataset like
target spacing, normalization parameters, patch size, median resampled image size, etc. These parameters are written in 
a ```config.json``` file. 

!!! note 
	The ```config.json``` file is extremely important for the rest of the training pipeline and is 
	required for inference.

To run the analysis portion of the MIST pipeline only, use the ```mist_analyze``` command. This command and its arguments are described below:

* ```mist_analyze```: Only run the analysis portion of the MIST pipeline 
	- ```--data```: (required) Full path to your dataset JSON file
	- ```--results```: (required) Full path to the directory to save the output of the MIST pipeline

## Preprocessing
The second step in the MIST pipeline is to take the parameters gathered from the analysis step and use them to preprocess the dataset.
In this step, we take the raw nifti files and convert them to the numpy files which will be used for training. 

To run the preprocessing portion of the MIST pipeline only, use the ```mist_preprocess``` command. This command and its arguments are described below:

* ```mist_preprocess```: Only run the preprocessing pipeline and assumes that the output of the analysis pipeline are in the ```--results``` folder
	- ```--data```: (required) Full path to your dataset JSON file
	- ```--numpy```: (required) Full path to the preprocessed numpy files
	- ```--results```: (required) Full path to the directory to save the output of the MIST pipeline

## Training
The final step in the MIST pipeline is to take the preprocessed data and train models using a five fold cross validation scheme.

To run the training portion of the MIST pipeline only, use the ```mist_train``` command. This command and its arguments are described below:

* ```mist_train```: Only run training pipeline. This assumes that the results of the previous two commands are in the ```--results``` and ```--numpy``` folders
	- ```--data```: (required) Full path to your dataset JSON file
	- ```--numpy```: (required) Full path to the preprocessed numpy files
	- ```--results```: (required) Full path to the directory to save the output of the MIST pipeline
    - ```--amp```: (optional, recommended) Turns on automatic mixed precision (AMP)
    - ```--pocket```: (optional, recommended) Turns on use of pocket networks (except for Attention U-Net or UNETR)
    - ```--use-res-block```: (optional, recommended) Turns on residual blocks in architectures (except for Attention U-Net or UNETR)

At the end of training, the evaluation script will automatically be triggered to gather metrics for the predictions
from the five fold cross validation. These individual metrics and their statistics are saved in the ```results.csv``` file.

## Inference
The MIST pipeline analyzes and preprocesses a dataset and trains models using the preprocessed data. It also includes 
auxiliary commands such as ```mist_predict```, which performs inference using trained MIST models on new data.

!!! note
	To use ```mist_predict```, you need the models directory and config.json file from the output of the main 
	MIST pipeline.

The ```mist_predict``` command and its arguments are described below:

* ```mist_predict```: Test time inference on new data given as either a JSON or CSV file 
	- ```--models```: (required) The full path to the ```models``` folder in the output of the MIST pipeline
	- ```--config```: (required) The full path to the directory to the ```config.json``` file in the output of the MIST pipeline
	- ```--data```: (required) The full path to the JSON or CSV file, which contains the path to your test data
	- ```--output```: (required) The full path to the directory to save the predictions
    - ```--fast```: (optional, default: False) Only use the first model (not all five) to get prediction
    - ```--gpu```: (optional, default: 0) Which GPU to run inference on
    - ```--sw-overlap```: (optional, default: 0.5) Value between 0 and 1 that sets how much overlap is between patches during inference
    - ```--blend-mode```: (optional, default: gaussian) How to blend output of overlapping windows
    - ```--tta```: (optional, default: False) Use this to turn on test time augmentation
    - ```--no-preprocess``` (optional, default: False) Use this to turn off the preprocessing pipeline before inference
    - ```--output-std``` (optional, default: False) Use this to output the standard deviation for predictions from multiple models

For CSV formatted data, the CSV file must, at a minimum, have an ```id``` column with the new patient IDs and one column for each image type. A column for the ```mask``` is allowed if you want to run the evaluation portion of the pipeline. For example, for the BraTS dataset, our CSV header would look like the following.

| id         | mask (optional) | t1               | t2               | tc               | fl               |
|------------|-----------------|------------------|------------------|------------------|------------------|
| Patient ID | Path to mask    | Path to t1 image | Path to t2 image | Path to tc image | Path to fl image |

Similarly, for JSON formatted data, we would have the following.

```text
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

## Postprocessing
MIST provides an optional postprocessing tool to test different postprocessing strategies after training. This command 
will apply a combination of user-provided postprocessing transformations and evaluate the new set of predictions. After
evaluation, an improvement score that ranges from 0 to 100 based on the average change for each metric vs. the original 
predictions is computed. If this improvement score exceeds 5, the MIST pipeline can update the ```config.json```
file with this new postprocessing strategy. 

The ```mist_postprocess``` and its arguments are described below:

* ```mist_postprocess```: Runs the postprocessing pipeline on the original predictions produced by the main MIST pipeline
	- ```--base-results```: (required) The full path to the results of the MIST pipeline
	- ```--output```: (required) The full path to the output of the postprocessing pipeline
	- ```--apply-to-labels```: (required, default: [-1]) List of labels to apply postprocessing transformations, default is ```[-1]``` or all labels combined
	- At least one postprocessing transformation:
      	- ```--remove-small-objects```: Removes small objects of size less than or equal to ```--small-object-threshold```, which is 64 by default
      	- ```--top-k-cc```: Only keep the ```--top-k``` (by default 2) connected components
      	- ```--fill-holes```: Fill holes with ```--fill-label``` (you need to specify this)
    - ```--metrics```: (required, default: ["dice", "haus95"]) List of metrics to use during evaluation (see ```mist_evaluate```)
	- ```--update-config```: (optional, default: False) Set this if you want the config file to be updated if postprocessing, on average, improves results by at least 5%
	- ```--morph-cleanup```: (optional, default: False) Applies a morphological erosion prior to selecting the top k connected components and then uses a dilation after the connected components are selected
    - ```--normalize-hd```: (optional, default: False) Turn this on to normalize Hausdorff distances during evaluation
    - ```--use-native-spacing```: (optional, default: False) Use native image spacing for computing Hausdorff distances during evaluation
	
!!! warning
	```mist_postprocess``` can be very slow if your images are extremely large and/or you have a lot of labels (which will slow down evaluation).

## Evaluation
The MIST evaluation command computes a set of metrics for a given set of predictions and ground truth masks. 

The ```mist_evaluate``` command and its arguments are described below:

* ```mist_evaluate```: Evaluate a set of predictions 
	- ```--paths```: (required) Path to CSV or JSON file with paths to ground truth masks (see ```mist_predict```)
	- ```--config```: (required) Path to ```config.json``` file from MIST output
	- ```--preds-dir```: (required) Path to directory where predictions are saved
	- ```--output-csv```: (required) Path to CSV file where evaluation results will be saved
    - ```--metrics```: (required, default: ["dice", "haus95"]) List of metrics to compute. The default metrics are the Dice coefficient (```dice```) and 95th percentile Hausdorff distance (```haus95```)
  						other metrics which can be included in this list are the average surface distance (```avg_surf```) and surface Dice (```surf_dice```) 
    - ```--normalize-hd```: (optional, default: False) Turn this on to normalize Hausdorff distances
    - ```--use-native-spacing```: (optional, default: False) Use native image spacing for computing Hausdorff distances
  
## Converting CSV and MSD Data
Several popular formats exist for different datasets, like the Medical Segmentation Decathlon (MSD) or simple CSV files 
with file paths to images and masks. To bridge the usability gap between these kinds of datasets and MIST, we provide 
a conversion tool called ```mist_convert_dataset``` to take MSD or CSV formatted data and convert it to 
MIST-compatible data.

The ```mist_convert_dataset``` command and its arguments are described below:

* ```mist_convert_dataset```: Converts either MSD or CSV formatted datasets into MIST-compatible data
	- ```--format```: (required: default: ```msd```) The format of the given dataset, which can be either ```msd``` or ```csv```
    - At least one of the following are required:
      	- ```--msd-source```: Full path to the MSD dataset, if that is what you are converting
      	- ```--train-csv```: Full path to the CSV file containing training data, if that is what you are using
	- ```--dest```: (required) Full path to the new MIST formatted dataset
    - ```--test-csv```: (optional) Full path to CSV file containing test data

!!! note	
	If converting a CSV file, it should mirror the CSV format shown in the ```mist_predict``` command. Additionally, this command will reformat the CSV dataset to a MIST-compatible one but will require the user to fill in details in its corresponding dataset JSON file.
