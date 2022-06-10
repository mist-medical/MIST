# Medical Imaging Segmentation Toolkit

Work in progress. MIST is a simple, fully automated 3D medical image segmentation framework built for TensorFlow. 

### Getting Started
Before starting with MIST, please ensure that your data is structured as follows:
    your_data/
    │   └── patient_1
    │       └── mask.nii.gz
    |       └── image_1.nii.gz
    |       └── image_2.nii.gz
    |       ...
    |       └── image_n.nii.gz
    |
    │   └── patient_2
    │       └── mask.nii.gz
    |       └── image_1.nii.gz
    |       └── image_2.nii.gz
    |       ...
    |       └── image_n.nii.gz
    |   ...
    └── 

To initiate the MIST pipeline, you need to provide it a JSON file with some parameters (i.e., path to data, modality, class labels, etc.). The `example.ipynb` notebook provides a use case for MIST on the LiTS dataset. 

### Inputs
Below is a table of all possible inputs to the MIST pipeline. Note that not all of these inputs are required.

| Input                | Type                    | Required? | Default                        | Description                                                                                                                                                                                                                                                                                            |
|----------------------|-------------------------|:---------:|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `raw_data_dir`       | `str`                   |     *     |                                | Path to directory containing raw nifti files                                                                                                                                                                                                                                                           |
| `processed_data_dir` | `str`                   |           | `cwd/tfrecord/`                | Path to directory where processed data will be written                                                                                                                                                                                                                                                 |
| `base_model_name`    | `str`                   |           | `'model'`                      | Name of models that will be saved                                                                                                                                                                                                                                                                      |
| `model_dir`          | `str`                   |           | `cwd/models/`                  | Path to directory where trained models will be saved                                                                                                                                                                                                                                                   |
| `raw_paths_csv`      | `str`                   |           | `cwd/paths.csv`                | Path and name of CSV file where paths to raw nifti files are stored                                                                                                                                                                                                                                    |
| `inferred_params`    | `str`                   |           | `cwd/inferred_params.json`     | Path and name of JSON file where properties of your dataset are saved                                                                                                                                                                                                                                  |
| `results_csv`        | `str`                   |           | `cwd/results.csv`              | Path and name of CSV file where prediction metrics are saved                                                                                                                                                                                                                                           |
| `modality`           | `str`                   |     *     |                                | Type of images in your dataset: <li> `'mr'` - MR images <li> `'ct'` - CT images <li> `'other'` - Will use same protocols as MR                                                                                                                                                                         |
| `mask`               | `list`                  |     *     |                                | List of naming conventions for ground truth masks: <li> i.e., `['mask.nii.gz, 'truth.nii.gz']`                                                                                                                                                                                                         |
| `images`             | `dict`                  |     *     |                                | Dictionary of naming conventions for each image: <li> i.e., `{'t1': ['t1.nii.gz'], 't2: ['t2.nii.gz']}` <li> i.e., `{'ct': ['volume.nii.gz']}`                                                                                                                                                         |
| `labels`             | `list`                  |     *     |                                | List containing the class labels in your dataset <li> i.e., `[0, 1, 2, 4]` <li> Note that the label 0 must be included in this list                                                                                                                                                                    |
| `final_classes`      | `dict`                  |     *     |                                | Dictionary with keys as final segmentation classes and values as lists with their associated labels: <li> i.e., `{'WT': [1, 2, 4], 'TC': [2, 4]}` <li> i.e., `{'liver': [1, 2], 'tumor': [2]}`                                                                                                         |
| `loss`               | `str`                   |           | `'dice'`                       | Loss functions used for training. Options are: <li> `'dice'` - Vanilla dice loss <li> `'gdl'` Weighted dice loss <li> `'bl'` - Boundary loss <li> `'hdos'` - One-sided Hausdorff loss <li> `'wnbl'` - Weighted normalized boundary loss                                                                |
| `model`              | `str`                   |           | `'unet'`                       | Architecture used for training. Options are: <li> `'unet'` - Standard 3D U-Net <li> `'resnet'` - U-Net with ResNet blocks <li> `'densenet'` - U-Net with DenseNet blocks <li> `'hrnet'` - Standard HRNet  </li> Additionally, you can provide a path to a pretrained MIST model for transfer learning. |
| `pocket`             | `bool`                  |           | `False`                        | Use Pocket version of selected network architecture                                                                                                                                                                                                                                                    |
| `fold`               | `int` or `list`         |           | `[0, 1, 2, 3, 4]`              | Specify which fold from a five-fold cross validation that you want to train on: <li> i.e., `0` <li> i.e., `[2, 4]`                                                                                                                                                                                     |
| `gpu`                | `int`, `list`, or `str` |           | `'auto'`                       | Specify which GPU(s) that you want to use for training. MIST easily supports multi-gpu training. <li> `0` - Train on GPU 0 <li> `[0, 1, 2]` - Train on GPUs 0, 1, and 2 <li> `'auto'` - Train on GPU with most available memory                                                                        |
| `patch_size`         | `list`                  |           | Automatically selected by MIST | Specify patch size used for training. By default, MIST will automatically select the largest patch size that can support a batch size of two.                                                                                                                                                          |
| `epochs`             | `int`                   |           | `250`                          | Specify the number of epochs used during training.                                                                                                                                                                                                                                                     |

TODO:

- Testing testing testing! So far, MIST has been tested on the LiTS, BraTS, and MSD datasets
- Create a MIST pypi package
- Add command line support
- Create documentation
- Benchmark against nnUNet for accuracy, speed, and resource consumption
