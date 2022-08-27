# MIST
MIST: A simple 3D medical imaging segmentation framework for TensorFlow.

### Main usage
```
usage: main.py [-h] [--exec-mode {all,analyze,preprocess,train}] [--data DATA] [--gpus GPUS [GPUS ...]]
               [--seed SEED] [--tta [BOOLEAN]] [--results RESULTS] [--processed-data PROCESSED_DATA]
               [--config CONFIG] [--paths PATHS] [--amp [BOOLEAN]] [--xla [BOOLEAN]]
               [--batch-size BATCH_SIZE] [--patch-size PATCH_SIZE [PATCH_SIZE ...]]
               [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
               [--lr-scheduler {none,poly,cosine_annealing}] [--end-learning-rate END_LEARNING_RATE]
               [--cosine-annealing-first-cycle-steps COSINE_ANNEALING_FIRST_CYCLE_STEPS]
               [--cosine-annealing-peak-decay COSINE_ANNEALING_PEAK_DECAY] [--optimizer {sgd,adam}]
               [--lookahead [BOOLEAN]] [--clip-norm [BOOLEAN]] [--clip-norm-max CLIP_NORM_MAX]
               [--model {nnunet,unet,resnet,densenet,hrnet}] [--depth DEPTH] [--init-filters INIT_FILTERS]
               [--pocket [BOOLEAN]] [--oversampling OVERSAMPLING]
               [--class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]] [--loss {dice_ce,dice,gdl,gdl_ce}]
               [--sw-overlap SW_OVERLAP] [--blend-mode {gaussian,constant}] [--nfolds NFOLDS]
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
  --results RESULTS     Path to output of MIST pipeline (default:
                        /workspace/github/mist_memory_test/mist_dali_loader/mist/results)
  --processed-data PROCESSED_DATA
                        Path to save input parameters for MIST pipeline (default:
                        /workspace/github/mist_memory_test/mist_dali_loader/mist/numpy)
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
                        Multiplier reducing initial learning rate for cosine annealing (default: 0.95)
  --optimizer {sgd,adam}
                        Optimizer (default: adam)
  --lookahead [BOOLEAN]
                        Use Lookahead with the optimizer (default: False)
  --clip-norm [BOOLEAN]
                        Use gradient clipping (default: False)
  --clip-norm-max CLIP_NORM_MAX
                        Max threshold for global norm clipping (default: 1.0)
  --model {nnunet,unet,resnet,densenet,hrnet}
  --depth DEPTH         Depth of U-Net (default: 3)
  --init-filters INIT_FILTERS
                        Number of filters to start U-Net (default: 32)
  --pocket [BOOLEAN]    Use pocket U-Net (default: False)
  --oversampling OVERSAMPLING
                        Probability of crop centered on foreground voxel (default: 0.4)
  --class-weights CLASS_WEIGHTS [CLASS_WEIGHTS ...]
                        Specify class weights (default: None)
  --loss {dice_ce,dice,gdl,gdl_ce}
                        Loss function for training (default: dice_ce)
  --sw-overlap SW_OVERLAP
                        Amount of overlap between scans during sliding window inference (default: 0.5)
  --blend-mode {gaussian,constant}
                        How to blend output of overlapping windows (default: gaussian)
  --nfolds NFOLDS       Number of cross-validation folds (default: 5)
  --folds FOLDS [FOLDS ...]
                        Which folds to run (default: [0, 1, 2, 3, 4])
  --epochs EPOCHS       Number of epochs (default: 1000)
  --steps-per-epoch STEPS_PER_EPOCH
                        Steps per epoch. By default ceil(training_dataset_size / batch_size / gpus)
                        (default: None)
```

### Predict usage
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

### Conversion usage
```
usage: convert_to_mist.py [-h] [--format {msd,csv}] [--msd-source MSD_SOURCE] [--train-csv TRAIN_CSV]
                          [--test-csv TEST_CSV] [--dest DEST]

optional arguments:
  -h, --help            show this help message and exit
  --format {msd,csv}    Format of dataset to be converted (default: msd)
  --msd-source MSD_SOURCE
                        Directory containing MSD formatted dataset (default: None)
  --train-csv TRAIN_CSV
                        Path to and name of csv containing training ids, mask, and images (default: None)
  --test-csv TEST_CSV   Path to and name of csv containing test ids and images (default: None)
  --dest DEST           Directory to save converted, MIST formatted dataset (default: None)
```