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
```console
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
	- ```--data```: (required) The full path to the CSV file, which contains the path to your test data
	- ```--output```: (required) The full path to the directory to save the predictions
    - ```--fast```: (optional, default: False) Only use the first model (not all five) to get prediction
    - ```--gpu```: (optional, default: 0) Which GPU to run inference on
    - ```--sw-overlap```: (optional, default: 0.5) Value between 0 and 1 that sets how much overlap is between patches during inference
    - ```--blend-mode```: (optional, default: gaussian) How to blend output of overlapping windows
    - ```--tta```: (optional, default: False) Use this to turn on test time augmentation
    - ```--no-preprocess``` (optional, default: False) Use this to turn off the preprocessing pipeline before inference
    - ```--output-std``` (optional, default: False) Use this to output the standard deviation for predictions from multiple models

For CSV formatted data, the CSV file must, at a minimum, have an ```id``` column with the new patient IDs and one column for each image type. For example, for the BraTS dataset, our CSV header would look like the following.

| id         | t1               | t2               | tc               | fl               |
|------------|------------------|------------------|------------------|------------------|
| Patient ID | Path to t1 image | Path to t2 image | Path to tc image | Path to fl image |

## Postprocessing

MIST includes a flexible postprocessing utility that allows users to apply custom postprocessing strategies to prediction masks. These strategies are defined via a JSON file and support operations like removing small objects, extracting connected components, and filling holes. This enables experimentation with a range of postprocessing techniques to improve segmentation accuracy.

Postprocessing is run using the `mist_postprocess` command and requires:
- a directory containing prediction masks (`--base-predictions`)
- an output directory to save the transformed predictions (`--output`)
- a strategy file specifying the sequence of transformations (`--postprocess-strategy`)

### Strategy-Based Postprocessing

Postprocessing is configured using a JSON strategy file. Each strategy is a list of steps, where each step includes the transformation name, the target labels, a flag for whether the operation should be applied sequentially per label, and any additional parameters.

### Usage

```bash
mist_postprocess \
  --base-predictions /path/to/results/predictions/train/raw \
  --output /path/to/results/predictions/train/post \
  --postprocess-strategy /path/to/strategy.json
```

### Strategy File Format
The strategy file is a JSON file containing a list of postprocessing steps. Each step is a dictionary with the following required fields:

- **`transform`** (`str`):  
  Name of the postprocessing transformation. Currently supported transformations are
  	- `remove_small_objects`: Remove connected components below a given size threshold.
	- `fill_holes_with_label`: Fill holes in a mask with a specified label.
	- `get_top_k_connected_components`: Keep the largest `k` connected components.
	- `replace_small_objects_with_label`: Replace the label of small objects with a different label.

  Each transformation can be applied either **sequentially per label** or **across a grouped set of labels**, controlled via the `apply_sequentially` flag.

  Each transform is registered in the `transform_registry.py` file. You can add custom postprocessing transforms by implementing them in this file and registering them with the `@register_transform('name')` decorator.

- **`apply_to_labels`** (`List[int]`):  
  A list of label integers to which the transform should be applied.  
  For example, `[1, 2]` will apply the transform to labels 1 and 2.

- **`apply_sequentially`** (`bool`):  
  Indicates whether to apply the transform to each label individually (`true`),  
  or to all the labels in the group at once (`false`).

- **`kwargs`** *(optional, `Dict[str, Any]`)*:  
  Additional keyword arguments passed directly to the transform function. These are transform-specific. For the currently available transforms, the following keyword arguments are valid:

  - `small_object_threshold` – A threshold for removing or replacing small objects.  
    Used in `remove_small_objects` and `replace_small_objects_with_label`. A common default value is `64`.

  - `top_k` – The number of largest connected components to retain.  
    Used in `get_top_k_connected_components`. For example, `top_k: 1` retains only the largest component.

  - `fill_label` – The label value to use when filling holes.  
    Used in `fill_holes_with_label`. For example, `fill_label: 1` will fill holes using label `1`.

  - `replacement_label` – The label used to replace small components. Used in `replace_small_objects_with_label`. For example, `replacement_label: 1` will update the label of objects smaller than `small_object_threshold` with `1`.

  - `morph_cleanup` – Whether to apply morphological erosion/dilation during `get_top_k_connected_components`. Set to `true` to activate cleanup.

  - `morph_cleanup_iterations` – The number of iterations to use for morphological erosion/dilation if `morph_cleanup` is enabled. Default is usually `2`.


Below is an example strategy file that demonstrates several transformations:

```json
[
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [1],
    "apply_sequentially": true,
    "kwargs": {
      "small_object_threshold": 64
    }
  },
  {
    "transform": "remove_small_objects",
    "apply_to_labels": [2, 4],
    "apply_sequentially": false,
    "kwargs": {
      "small_object_threshold": 100
    }
  },
  {
    "transform": "fill_holes_with_label",
    "apply_to_labels": [1, 2],
    "apply_sequentially": false,
    "kwargs": {
      "fill_label": 1
    }
  },
  {
    "transform": "get_top_k_connected_components",
    "apply_to_labels": [4],
    "apply_sequentially": true,
    "kwargs": {
      "top_k": 1,
      "morph_cleanup": true,
      "morph_cleanup_iterations": 1
    }
  },
  {
    "transform": "replace_small_objects_with_label",
    "apply_to_labels": [1, 2, 4],
    "apply_sequentially": true,
    "kwargs": {
      "small_object_threshold": 50,
      "replacement_label": 0
    }
  }
]
```

### Notes

- This version of `mist_postprocess` does **not** perform evaluation or update the `config.json` file. It is designed to decouple strategy testing from evaluation logic.
- Evaluation of postprocessing performance can be done separately using `mist_evaluate`.

## Evaluation

MIST provides a flexible command-line tool to evaluate prediction masks against ground truth using various metrics. The evaluation script supports several metrics and outputs a detailed summary of the evaluation in CSV format.

### Command

```bash
mist_evaluate --config CONFIG_PATH \
              --paths-csv PATHS_CSV \
              --output-csv OUTPUT_CSV \
              --metrics dice haus95
```

### Arguments

- **`--config`** *(str, required)*:  
  Path to the `config.json` file generated by the MIST pipeline. This file must contain the `final_classes` list, which defines the evaluation classes.

- **`--paths-csv`** *(str, required)*:  
  Path to a CSV file that contains absolute paths to prediction and ground truth files. This file should contain the following columns `id`, `mask`, `prediction`, which are the unique patient id, path to the ground truth mask, and path to the prediction, respectively.

- **`--output-csv`** *(str, required)*:  
  Path to the CSV file where evaluation results will be saved.

- **`--metrics`** *(List[str], optional)*:  
  A list of evaluation metrics to compute. Available options include:
  - `dice`: Dice coefficient  
  - `haus95`: 95th percentile Hausdorff distance  
  - `surf_dice`: Surface Dice  
  - `avg_surf`: Average symmetric surface distance  
  **Default**: `["dice", "haus95"]`

  Custom metrics can be implemented and registered in the `metrics_registry.py` file.

- **`--surf-dice-tol`** *(float, optional)*:  
  Tolerance value (in mm) used to compute the surface Dice.  
  Only applicable if `surf_dice` is included in `--metrics`.  
  **Default**: `1.0`
  
## Converting CSV and MSD Data
Several popular formats exist for different datasets, like the Medical Segmentation Decathlon (MSD) or simple CSV files 
with file paths to images and masks. To bridge the usability gap between these kinds of datasets and MIST, we provide 
a conversion tool called ```mist_convert_dataset``` to take MSD or CSV formatted data and convert it to 
MIST-compatible data.

The ```mist_convert_dataset``` command and its arguments are described below:

* ```mist_convert_dataset```: Converts either MSD or CSV formatted datasets into MIST-compatible data
	- ```--format```: (required) The format of the given dataset, which can be either ```msd``` or ```csv```
  - ```--output```: (required) Full path to the new MIST formatted dataset
    - At least one of the following are required depending on the chosen format:
      	- ```--msd-source```: Full path to the MSD dataset, if that is what you are converting
      	- ```--train-csv```: Full path to the CSV file containing training data, if that is what you are using
    - ```--test-csv```: (optional) Full path to CSV file containing test data

!!! note	
	If converting a CSV file, it should mirror the CSV format shown in the ```mist_predict``` command. Additionally, this command will reformat the CSV dataset to a MIST-compatible one but will require the user to fill in details in its corresponding dataset JSON file.
