import os
import json
import ants
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Rich progres bar
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn
)
from rich.console import Console
from rich.text import Text
from runtime.progress_bar import TrainProgressBar, ValidationProgressBar

# PyTorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# MONAI
from monai.inferers import sliding_window_inference

# Custom code
from data_loading.dali_loader import get_training_dataset, get_validation_dataset, get_test_dataset
from models.get_model import get_model
from runtime.loss import get_loss, DiceLoss, VAELoss
from inference.main_inference import predict_single_example
from runtime.utils import get_optimizer, get_lr_schedule, Mean, get_test_df, create_model_config_file, \
    load_model_from_config, get_master_port

console = Console()


class Trainer:
    def __init__(self, args):
        # Read user defined parameters
        self.args = args
        with open(self.args.data, "r") as file:
            self.data = json.load(file)

        # Get dataset configuration file
        self.config_file = os.path.join(self.args.results, "config.json")
        with open(self.config_file, "r") as file:
            self.config = json.load(file)

        self.n_channels = len(self.data["images"])
        self.n_classes = len(self.data["labels"])

        # Get paths to dataset
        self.df = pd.read_csv(os.path.join(self.args.results, "train_paths.csv"))

        # Get patch size if not specified by user
        self.patch_size = self.args.patch_size

        self.config['patch_size'] = self.patch_size
        with open(self.config_file, 'w') as outfile:
            json.dump(self.config, outfile, indent=2)

        # Get network depth based on patch size
        if self.args.depth is None:
            self.depth = np.min([int(np.log(np.min(self.patch_size) // 4) // np.log(2)), 5])
        else:
            self.depth = self.args.depth

        # Get latent dimension for VAE regularization
        self.latent_dim = int(np.prod(np.array(self.patch_size) // 2**self.depth))

        # Create model configuration file for inference later
        self.model_config_path = os.path.join(self.args.results, "models", "model_config.json")
        self.model_config = create_model_config_file(self.args,
                                                     self.config,
                                                     self.data,
                                                     self.depth,
                                                     self.latent_dim,
                                                     self.model_config_path)

        # Get class weights if we are using them
        if self.args.use_precomputed_weights:
            self.class_weights = self.config['class_weights']
        else:
            self.class_weights = None

        # Get standard dice loss for validation
        self.dice_loss = DiceLoss()

        # Get VAE regularization loss
        self.vae_loss = VAELoss()

    def predict_on_val(self, model_path, loader, df):
        # Load model
        model = load_model_from_config(model_path, self.model_config_path)
        model.eval()
        model.to("cuda")

        # Set up rich progress bar
        testing_progress = Progress(TextColumn("Testing on fold"),
                                    BarColumn(),
                                    MofNCompleteColumn(),
                                    TextColumn("•"),
                                    TimeElapsedColumn())

        # Run prediction on all samples and compute metrics
        with torch.no_grad(), testing_progress as pb:
            for i in pb.track(range(len(df))):
                # Get original patient data
                patient = df.iloc[i].to_dict()
                image_list = list(patient.values())[2:len(patient)]
                original_image = ants.image_read(image_list[0])

                # Get preprocessed image from DALI loader
                data = loader.next()[0]
                image = data["image"]

                # Predict with model and put back into original image space
                pred = predict_single_example(image,
                                              original_image,
                                              self.config,
                                              [model],
                                              self.args.sw_overlap,
                                              self.args.blend_mode,
                                              self.args.tta)

                # Write prediction as .nii.gz file
                prediction_filename = '{}.nii.gz'.format(patient['id'])
                ants.image_write(pred,
                                 os.path.join(self.args.results, 'predictions', 'train', 'raw', prediction_filename))

        text = Text("\n")
        console.print(text)

    # Set up for distributed training
    def setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = get_master_port()
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Clean up processes after distributed training
    def cleanup(self):
        dist.destroy_process_group()

    def train(self, rank, world_size):
        # Set up for distributed training
        self.setup(rank, world_size)

        # Set device rank for each process
        torch.cuda.set_device(rank)

        # Get folds for k-fold cross validation
        kfold = KFold(n_splits=self.args.nfolds, shuffle=True, random_state=42)

        images_dir = os.path.join(self.args.numpy, 'images')
        images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]

        labels_dir = os.path.join(self.args.numpy, 'labels')
        labels = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir)]

        splits = kfold.split(list(range(len(images))))

        # Extract folds so that users can specify folds to train on
        train_splits = list()
        test_splits = list()
        for split in splits:
            train_splits.append(split[0])
            test_splits.append(split[1])

        # Start training loop
        if rank == 0:
            text = Text("\nStarting training\n")
            text.stylize("bold")
            console.print(text)

        for fold in self.args.folds:
            train_images = [images[idx] for idx in train_splits[fold]]
            train_labels = [labels[idx] for idx in train_splits[fold]]

            # Get validation set from training split
            train_images, val_images, train_labels, val_labels = train_test_split(train_images,
                                                                                  train_labels,
                                                                                  test_size=0.1,
                                                                                  random_state=self.args.seed)

            # Get number of validation steps per epoch
            # Divide by world size since this dataset is sharded across all GPUs
            val_steps = len(val_images) // world_size

            # Get DALI loaders
            train_loader = get_training_dataset(train_images,
                                                train_labels,
                                                batch_size=self.args.batch_size // world_size,
                                                oversampling=self.args.oversampling,
                                                patch_size=self.patch_size,
                                                seed=self.args.seed,
                                                num_workers=self.args.num_workers,
                                                rank=rank,
                                                world_size=world_size)

            val_loader = get_validation_dataset(val_images,
                                                val_labels,
                                                seed=self.args.seed,
                                                num_workers=self.args.num_workers,
                                                rank=rank,
                                                world_size=world_size)

            # Get steps per epoch
            if self.args.steps_per_epoch is None:
                self.args.steps_per_epoch = len(train_images) // self.args.batch_size // world_size
            else:
                self.args.steps_per_epoch = self.args.steps_per_epoch

            # Get loss function
            loss_fn = get_loss(self.args, class_weights=self.class_weights)

            # Get model
            model = get_model(**self.model_config)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.to(rank)
            model = DDP(model, device_ids=[rank])

            # Get optimizer and lr scheduler
            optimizer = get_optimizer(self.args, model)
            scheduler = get_lr_schedule(self.args, optimizer)

            # If using AMP, use gradient scaling
            if self.args.amp:
                scaler = torch.cuda.amp.GradScaler()

            # Only log metrics on first process (i.e., rank 0)
            if rank == 0:
                # Compute running averages for losses
                running_loss_train = Mean()
                running_loss_validation = Mean()

                # Initialize best validation loss
                best_loss = np.Inf

                # Set up tensorboard summary writer
                writer = SummaryWriter(os.path.join(self.args.results, 'logs', 'fold_{}'.format(fold)))

                # Best model path
                best_model_name = os.path.join(self.args.results, 'models', 'fold_{}.pt'.format(fold))

            # Function to perform a single training step
            def train_step(image, label):
                # Loss computation
                def compute_loss():
                    output = model(image)

                    loss = loss_fn(label, output["prediction"])

                    if self.args.deep_supervision:
                        for k, p in enumerate(output["deep_supervision"]):
                            loss += 0.5 ** (k + 1) * loss_fn(label, p)

                        c_norm = 1 / (2 - 2 ** (-(len(output["deep_supervision"]) + 1)))
                        loss *= c_norm

                    if self.args.vae_reg:
                        loss += self.args.vae_penalty * self.vae_loss(image, output["vae_reg"])

                    # L2 regularization term
                    if self.args.l2_reg:
                        l2_reg = 0.0
                        for param in model.parameters():
                            l2_reg += torch.norm(param, p=2)

                        loss += self.args.l2_penalty * l2_reg

                    # L1 regularization term
                    if self.args.l1_reg:
                        l1_reg = 0.0
                        for param in model.parameters():
                            l1_reg += torch.norm(param, p=1)

                        loss += self.args.l1_penalty * l1_reg

                    return loss

                # Zero your gradients for every batch!
                optimizer.zero_grad()

                # Make predictions for this batch
                if self.args.amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = compute_loss()

                    # Compute the loss and its gradients
                    scaler.scale(loss).backward()

                    # Apply clip norm
                    if self.args.clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm_max)

                    # Adjust learning weights
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = compute_loss()

                    # Compute the loss and its gradients
                    loss.backward()

                    # Apply clip norm
                    if self.args.clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm_max)

                    optimizer.step()

                scheduler.step()
                return loss

            def val_step(image, label):
                pred = sliding_window_inference(image,
                                                roi_size=self.patch_size,
                                                overlap=self.args.sw_overlap,
                                                sw_batch_size=1,
                                                predictor=model,
                                                device=torch.device("cuda"))

                return self.dice_loss(label, pred)

            for epoch in range(self.args.epochs):
                # Make sure gradient tracking is on, and do a pass over the data
                model.train(True)
                if rank == 0:
                    with TrainProgressBar(epoch + 1, fold, self.args.epochs, self.args.steps_per_epoch) as pb:
                        for i in range(self.args.steps_per_epoch):
                            data = train_loader.next()[0]
                            image, label = data["image"], data["label"]

                            # Compute loss for single step
                            loss = train_step(image, label)

                            # Send all training losses to device 0 for average
                            dist.reduce(loss, dst=0)

                            # Print running loss to progress bar for rank 0 only
                            current_loss = loss.item() / world_size
                            running_loss = running_loss_train(current_loss)
                            pb.update(loss=running_loss)
                else:
                    for i in range(self.args.steps_per_epoch):
                        data = train_loader.next()[0]
                        image, label = data["image"], data["label"]

                        # Compute loss for single step
                        loss = train_step(image, label)

                        # Send loss to device 0
                        dist.reduce(loss, dst=0)

                dist.barrier()

                # Start validation
                # We don't need gradients on to do reporting
                model.eval()
                with torch.no_grad():
                    if rank == 0:
                        with ValidationProgressBar(val_steps) as pb:
                            for i in range(val_steps):
                                # Get data from validation loader
                                data = val_loader.next()[0]
                                image, label = data["image"], data["label"]

                                # Compute loss for single validation step
                                val_loss = val_step(image, label)

                                # Send all validation losses to device 0 for average
                                dist.reduce(val_loss, dst=0)

                                # Print running loss to progress bar for rank 0 only
                                current_val_loss = val_loss.item() / world_size
                                running_val_loss = running_loss_validation(current_val_loss)
                                pb.update(loss=running_val_loss)

                        # Check if validation loss is lower than the current best loss
                        # Save best model
                        if running_val_loss < best_loss:
                            text = Text(f"Validation loss IMPROVED from {best_loss:.4} to {running_val_loss:.4}\n")
                            text.stylize("bold")
                            console.print(text)
                            best_loss = running_val_loss
                            torch.save(model.state_dict(), best_model_name)
                        else:
                            text = Text(f"Validation loss did NOT improve from {best_loss:.4}\n")
                            console.print(text)
                    else:
                        for i in range(val_steps):
                            # Get data from validation loader
                            data = val_loader.next()[0]
                            image, label = data["image"], data["label"]

                            val_loss = val_step(image, label)
                            dist.reduce(val_loss, dst=0)

                # Reset training and validation loaders after each epoch
                train_loader.reset()
                val_loader.reset()

                if rank == 0:
                    # Log the running loss for validation
                    writer.add_scalars('Training vs. Validation Loss',
                                       {'Training': running_loss, 'Validation': running_val_loss},
                                       epoch + 1)
                    writer.flush()

                    # Reset running losses for new epoch
                    running_loss_train.reset_states()
                    running_loss_validation.reset_states()

            dist.barrier()
            if rank == 0:
                writer.close()

                # Prepare test set on rank 0 device only
                test_images = [images[idx] for idx in test_splits[fold]]
                test_images.sort()

                test_labels = [labels[idx] for idx in test_splits[fold]]
                test_labels.sort()

                test_loader = get_test_dataset(test_images,
                                               seed=self.args.seed,
                                               num_workers=self.args.num_workers,
                                               rank=0,
                                               world_size=1)

                # Bug fix: Strange behavior with numerical ids
                test_df_ids = [pat.split('/')[-1].split('.')[0] for pat in test_images]
                test_df = get_test_df(self.df, test_df_ids)

                # Run inference on test set
                self.predict_on_val(best_model_name, test_loader, test_df)

        self.cleanup()

    def fit(self):
        # Train model
        world_size = torch.cuda.device_count()
        mp.spawn(self.train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
