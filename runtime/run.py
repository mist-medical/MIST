import os
import gc
import json

import ants
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from models.get_model import get_model
from runtime.loss import DiceLoss, get_loss
from runtime.checkpoints import Checkpoints
from runtime.progress_bar import ProgressBar
from runtime.logger import Logger
from inference.main_inference import predict_single_example, load_test_time_models, test_time_inference
from inference.sliding_window import sliding_window_inference
from postprocess_preds.postprocess import Postprocess
from runtime.utils import get_files_df, get_optimizer, get_flip_axes, \
    evaluate_prediction, compute_results_stats, init_results_df, set_seed, set_tf_flags, set_visible_devices, \
    set_memory_growth, set_amp, set_xla, get_test_df
from data_loading.dali_loader import get_validation_dataset, get_distributed_train_dataset


class RunTime:

    def __init__(self, args):
        # Read user defined parameters
        self.args = args
        with open(self.args.data, 'r') as file:
            self.data = json.load(file)

        if self.args.config is None:
            self.config_file = os.path.join(self.args.results, 'config.json')
        else:
            self.config_file = self.args.config

        with open(self.config_file, 'r') as file:
            self.config = json.load(file)

        self.n_channels = len(self.data['images'])
        self.n_classes = len(self.data['labels'])

        self.flip_axes = get_flip_axes()

        self.reduction = tf.keras.losses.Reduction.NONE
        self.dice_loss = DiceLoss(reduction=self.reduction)

        # Get paths to dataset
        if self.args.paths is None:
            self.df = pd.read_csv(os.path.join(self.args.results, 'train_paths.csv'))
        else:
            self.df = pd.read_csv(self.args.paths)

        self.results_df = init_results_df(self.data)

    def predict_and_evaluate_val(self, model_path, df, ds):
        model = load_model(model_path, compile=False)
        num_patients = len(df)
        pbar = tqdm(total=num_patients)
        pred_temp_filename = os.path.join(self.args.results, 'predictions', 'train', 'raw', 'pred_temp.nii.gz')
        mask_temp_filename = os.path.join(self.args.results, 'predictions', 'train', 'raw', 'mask_temp.nii.gz')

        for step, (image, _) in enumerate(ds.take(len(df))):
            patient = df.iloc[step].to_dict()
            image_list = list(patient.values())[2:len(patient)]
            original_image = ants.image_read(image_list[0])

            prediction = predict_single_example(image,
                                                original_image,
                                                self.config,
                                                [model],
                                                self.args.sw_overlap,
                                                self.args.blend_mode,
                                                False)

            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction,
                             os.path.join(self.args.results, 'predictions', 'train', 'raw', prediction_filename))

            # Evaluate prediction
            original_mask = ants.image_read(patient['mask'])
            eval_results = evaluate_prediction(prediction,
                                               original_mask,
                                               patient['id'],
                                               self.data,
                                               pred_temp_filename,
                                               mask_temp_filename,
                                               self.results_df.columns)

            # Update results df
            self.results_df = self.results_df.append(eval_results, ignore_index=True)

            # gc.collect()
            pbar.update(1)

        # Delete temporary files
        pbar.close()
        os.remove(pred_temp_filename)
        os.remove(mask_temp_filename)

        K.clear_session()
        del model
        gc.collect()

    def train(self):
        # Get folds for k-fold cross validation
        kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=42)

        images_dir = os.path.join(self.args.processed_data, 'images')
        images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]

        labels_dir = os.path.join(self.args.processed_data, 'labels')
        labels = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir)]

        splits = kfold.split(list(range(len(images))))

        # Extract folds so that users can specify folds to train on
        train_splits = list()
        test_splits = list()
        for split in splits:
            train_splits.append(split[0])
            test_splits.append(split[1])

        # Get patch size if not specified by user
        if self.args.patch_size is None:
            patch_size = [64, 64, 64]
        else:
            patch_size = self.args.patch_size

        # Get network depth based on patch size
        if self.args.depth is None:
            depth = np.min([int(np.log(np.min(patch_size) // 4) // np.log(2)), 6])
        else:
            depth = self.args.depth

        self.config['patch_size'] = patch_size
        with open(self.config_file, 'w') as outfile:
            json.dump(self.config, outfile, indent=2)

        # Start training loop
        for fold in self.args.folds:
            print('Starting fold {}...'.format(fold))
            train_images = [images[idx] for idx in train_splits[fold]]
            train_labels = [labels[idx] for idx in train_splits[fold]]

            # Get validation set from training split
            train_images, val_images, train_labels, val_labels = train_test_split(train_images,
                                                                                  train_labels,
                                                                                  test_size=0.1,
                                                                                  random_state=self.args.seed)

            # Get DALI loaders
            train_loader = get_distributed_train_dataset(imgs=train_images,
                                                         lbls=train_labels,
                                                         batch_size=self.args.batch_size,
                                                         strategy=self.strategy,
                                                         n_gpus=self.n_gpus,
                                                         seed=self.args.seed,
                                                         num_workers=8,
                                                         oversampling=self.args.oversampling,
                                                         patch_size=patch_size)

            val_loader = get_validation_dataset(imgs=val_images,
                                                lbls=val_labels,
                                                batch_size=1,
                                                mode='eval',
                                                seed=self.args.seed,
                                                num_workers=8)

            # Get steps per epoch
            if self.args.steps_per_epoch is None:
                self.args.steps_per_epoch = len(train_images) // self.args.batch_size
            else:
                self.args.steps_per_epoch = self.args.steps_per_epoch

            # Get class weights if we are using them
            if self.args.use_precomputed_weights:
                class_weights = self.config['class_weights']
            else:
                class_weights = None

            # Set up optimizer, model, and loss under distribution strategy scope
            with self.strategy.scope():
                # Set up optimizer
                optimizer = get_optimizer(self.args)

                # Get model
                model = get_model(self.args.model,
                                  input_shape=tuple(patch_size),
                                  n_channels=len(self.data['images']),
                                  n_classes=self.n_classes,
                                  init_filters=self.args.init_filters,
                                  depth=depth,
                                  deep_supervision=self.args.deep_supervision,
                                  pocket=self.args.pocket,
                                  config=self.config)

                # Get loss function
                loss_fn = get_loss(self.args,
                                   reduction=self.reduction,
                                   class_weights=class_weights)

            train_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

            def train_step(image, mask):
                with tf.GradientTape() as tape:
                    pred = model(image)

                    if self.args.deep_supervision:
                        unscaled_loss = loss_fn(mask, pred[0])

                        for i, p in enumerate(pred[1:]):
                            unscaled_loss += 0.5 ** (i + 1) * loss_fn(mask, p)

                        c_norm = 1 / (2 - 2 ** (-len(pred)))
                        unscaled_loss *= c_norm
                    else:
                        unscaled_loss = loss_fn(mask, pred)

                    loss = unscaled_loss
                    if self.args.amp:
                        loss = optimizer.get_scaled_loss(unscaled_loss)

                gradients = tape.gradient(loss, model.trainable_variables)
                if self.args.amp:
                    gradients = optimizer.get_unscaled_gradients(gradients)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(unscaled_loss)
                return loss

            @tf.function
            def distributed_train_step(image, mask):
                per_replica_losses = self.strategy.run(train_step, args=(image, mask))
                return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

            val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

            def val_step(image, mask):
                pred = sliding_window_inference(image,
                                                n_class=self.n_classes,
                                                roi_size=tuple(patch_size),
                                                sw_batch_size=1,
                                                overlap=self.args.sw_overlap,
                                                blend_mode=self.args.blend_mode,
                                                model=model)

                val_loss(self.dice_loss(mask, pred))

            # Setup checkpoints for training
            best_val_loss = np.Inf
            best_model_path = os.path.join(self.args.results, 'models', 'best',
                                           '{}_best_model_split_{}'.format(self.data['task'], fold))
            last_model_path = os.path.join(self.args.results, 'models', 'last',
                                           '{}_last_model_split_{}'.format(self.data['task'], fold))

            checkpoint = Checkpoints(best_model_path, last_model_path, best_val_loss)

            # Setup progress bar
            progress_bar = ProgressBar(self.args.steps_per_epoch, len(val_images), train_loss, val_loss)

            # Setup logging
            logs = Logger(self.args, fold, train_loss, val_loss)

            total_steps = self.args.epochs * self.args.steps_per_epoch
            current_epoch = 1
            local_step = 1
            for global_step, (image, mask) in enumerate(train_loader):
                if global_step >= total_steps:
                    break

                if local_step == 1:
                    print('Fold {}: Epoch {}/{}'.format(fold, current_epoch, self.args.epochs))

                distributed_train_step(image, mask)
                progress_bar.update_train_bar()
                local_step += 1

                if (global_step + 1) % self.args.steps_per_epoch == 0 and global_step > 0:
                    # Perform validation
                    for _, (val_image, val_mask) in enumerate(val_loader.take(len(val_images))):
                        val_step(val_image, val_mask)
                        progress_bar.update_val_bar()

                    current_val_loss = val_loss.result().numpy()

                    checkpoint.update(model, current_val_loss)
                    logs.update(current_epoch)

                    progress_bar.reset()
                    current_epoch += 1
                    local_step = 1
                    gc.collect()

            # End of training for fold

            # Save last model
            print('Training for fold {} complete...'.format(fold))
            checkpoint.save_last_model(model)

            K.clear_session()
            del model, train_loader, val_loader
            gc.collect()

            # Run prediction on test set and write results to .nii.gz format
            # Prepare test set
            test_images = [images[idx] for idx in test_splits[fold]]
            test_images.sort()

            test_labels = [labels[idx] for idx in test_splits[fold]]
            test_labels.sort()

            test_loader = get_validation_dataset(imgs=test_images,
                                                 lbls=test_labels,
                                                 batch_size=1,
                                                 mode='eval',
                                                 seed=42,
                                                 num_workers=8)

            # Bug fix: Strange behavior with numerical ids
            test_df_ids = [pat.split('/')[-1].split('.')[0] for pat in test_images]
            test_df = get_test_df(self.df, test_df_ids)

            # print('Running inference on validation set...')
            self.predict_and_evaluate_val(best_model_path, test_df, test_loader)

            K.clear_session()
            del test_loader
            gc.collect()

            # End of fold

        K.clear_session()
        gc.collect()
        # End train function

    def run(self):
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "1"

        # Set seed if specified
        if not (self.args.seed is None):
            set_seed(self.args.seed)

        # Set visible devices
        set_visible_devices(self.args)

        # Allow memory growth
        set_memory_growth()

        # Set AMP and XLA if called for
        if self.args.amp:
            set_amp()

        if self.args.xla:
            set_xla(self.args)

        # Set tf flags
        set_tf_flags(self.args)

        # Define tf distribution strategy for multi or single gpu training
        self.n_gpus = len(self.args.gpus)
        self.strategy = tf.distribute.MirroredStrategy()

        # Run training pipeline
        self.train()

        # Get final statistics
        self.results_df = compute_results_stats(self.results_df)

        # Write results to csv file
        self.results_df.to_csv(os.path.join(self.args.results, 'results.csv'), index=False)

        # Run post-processing
        postprocess = Postprocess(self.args)
        postprocess.run()

        # Run inference on test set if it is provided
        if 'test-data' in self.data.keys():
            print('Running inference on test set...')
            test_df = get_files_df(self.data, 'test')
            models_dir = os.path.join(self.args.results, 'models', 'best')

            models = load_test_time_models(models_dir, False)

            test_time_inference(test_df,
                                os.path.join(self.args.results, 'predictions', 'test'),
                                self.config_file,
                                models,
                                self.args.sw_overlap,
                                self.args.blend_mode,
                                True)

        K.clear_session()
        gc.collect()
