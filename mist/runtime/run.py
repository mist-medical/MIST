import os
import json
import ants
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Rich progress bar
from rich.console import Console
from rich.text import Text
from mist.runtime.progress_bar import TrainProgressBar, ValidationProgressBar

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

# MIST modules
from mist.data_loading.dali_loader import (
    get_training_dataset,
    get_validation_dataset,
    get_test_dataset
)

from mist.models.get_model import get_model, load_model_from_config, configure_pretrained_model
from mist.runtime.loss import get_loss, DiceLoss, VAELoss, MAELoss, MSELoss, WeightedMSELoss, DVHLoss
from mist.inference.main_inference import predict_single_example

from mist.runtime.utils import (
    read_json_file,
    get_optimizer,
    get_lr_schedule,
    Mean,
    create_model_config_file,
    create_pretrained_config_file,
    get_progress_bar,
    AlphaSchedule,
)

console = Console()


class Trainer:
    def __init__(self, args):
        # Read user defined parameters
        self.args = args
        
        # Read dataset.json file
        self.data = read_json_file(self.args.data)
        
        # Read config.json file
        self.config_file = os.path.join(
            self.args.results, 'config.json'
        )
        self.config = read_json_file(self.config_file)

        # Get number of channels and classes from dataset.json
        if self.data["modality"] == "ct":
            self.n_channels = len(self.data["images"])
            self.n_classes = len(self.data["labels"])
        elif self.data["modality"] == "dose":
            self.n_channels = len(self.data["images"]["ct"]) + len(self.data["mask"]) + len(self.data["images"]["ptvs"]) 
            self.n_classes = len(self.data["images"]["dose"])  # Should be 1 for dose prediction as it is regression

        # Get paths to dataset
        self.df = pd.read_csv(os.path.join(self.args.results, "train_paths.csv"))

        # If custom patch size is given, then use it
        if self.args.patch_size is not None:
            self.patch_size = self.args.patch_size
            self.config["patch_size"] = self.args.patch_size
        else:
            self.patch_size = self.config["patch_size"]

        # Get bounding box data. For now no preprocessing for dose data
        if self.config["crop_to_fg"] and not self.args.no_preprocess:  # We could add and self.data["modality"] != "dose" in preprocess_dataset() , run.py, test_on_fold???
        # if self.config['modality'] != 'dose' and self.args.no_preprocess == False:
            self.fg_bboxes = pd.read_csv(os.path.join(self.args.results, "fg_bboxes.csv"))

        # Create model configuration file for inference later
        self.model_config_path = os.path.join(self.args.results, "models", "model_config.json")
        if self.args.model != "pretrained":  
            self.model_config = create_model_config_file(self.args,
                                                         self.config,
                                                         self.data,
                                                         self.model_config_path)
        else:
            assert self.args.pretrained_model_path is not None, "No pretrained model path given!"
            self.model_config = create_pretrained_config_file(self.args.pretrained_model_path,
                                                              self.data,
                                                              self.model_config_path)
            self.patch_size = self.model_config["patch_size"]

        with open(self.config_file, "w") as outfile:
            json.dump(self.config, outfile, indent=2)

        # Get class weights if we are using them
        if self.args.use_config_class_weights:
            self.class_weights = self.config["class_weights"]
        else:
            self.class_weights = None

        # Get alpha schedule for weighting boundary losses 
        self.alpha = AlphaSchedule(
            self.args.epochs,
            self.args.boundary_loss_schedule,
            constant=self.args.loss_schedule_constant,
            init_pause=self.args.linear_schedule_pause,
            step_length=self.args.step_schedule_step_length
        )

        # Get standard dice loss for validation. Need to match training one, right!? But does not in default MIST.
        if self.data["modality"] != "dose":
            self.val_loss = DiceLoss()  # This is different from default training/input args loss that is dice_ce
        else:
            self.val_loss = MSELoss() # default val loss

        # Get VAE regularization loss. Variational Autoencoders
        self.vae_loss = VAELoss()

        self.__logger = logging.getLogger(__name__)  # Maybe because of how the distr training is, we cannot log info for training step.

    # Set up for distributed training
    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.args.master_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Clean up processes after distributed training
    def cleanup(self):
        dist.destroy_process_group()

    def train(self, rank, world_size):
        # Set up for distributed training
        self.setup(rank, world_size)

        # Set device rank for each process
        torch.cuda.set_device(rank)

        # Start training loop
        if rank == 0:
            text = Text("\nStarting training\n")
            text.stylize("bold")
            console.print(text)

        for fold in self.args.folds:
            # Get training ids from dataframe
            train_ids = list(self.df.loc[self.df["fold"] != fold]["id"])
            train_images = [os.path.join(self.args.numpy, "images", "{}.npy".format(pat)) for pat in train_ids]     # 3 channels: ct, oars, ptvs. e.g. zyxC (214, 188, 277, 3)
            train_labels = [os.path.join(self.args.numpy, "labels", "{}.npy".format(pat)) for pat in train_ids]     # 2 channel: 0 gt_dose, 1 weights. zyxC (221, 170, 248, 1)

            if self.args.use_dtms:
                train_dtms = [os.path.join(self.args.numpy, "dtms", "{}.npy".format(pat)) for pat in train_ids]

                zip_labels_dtms = [vol for vol in zip(train_labels, train_dtms)]

                # Get validation set from training split with DTMs
                train_images, val_images, train_labels_dtms, val_labels_dtms = train_test_split(train_images,
                                                                                                zip_labels_dtms,
                                                                                                test_size=self.args.val_percent,
                                                                                                random_state=self.args.seed_val)

                train_labels = [vol[0] for vol in train_labels_dtms]
                train_dtms = [vol[1] for vol in train_labels_dtms]
                val_labels = [vol[0] for vol in val_labels_dtms]
            else:
                # Get validation set from training split
                train_images, val_images, train_labels, val_labels = train_test_split(train_images,
                                                                                      train_labels,
                                                                                      test_size=self.args.val_percent,
                                                                                      random_state=self.args.seed_val)

                train_dtms = None

            # Get number of validation steps per epoch
            # Divide by world size since this dataset is sharded across all GPUs
            val_steps = len(val_images) // world_size

            # Get DALI loaders
            train_loader = get_training_dataset(train_images,
                                                train_labels,
                                                train_dtms,
                                                modality=self.data['modality'],
                                                batch_size=self.args.batch_size // world_size,
                                                oversampling=self.args.oversampling,
                                                patch_size=self.patch_size,
                                                seed=self.args.seed_val,
                                                num_workers=self.args.num_workers,
                                                rank=rank,
                                                world_size=world_size)
            # Take care of validation set in batches as training set??? Maybe add below batch_size=self.args.batch_size // world_size,
            val_loader = get_validation_dataset(val_images,
                                                val_labels,
                                                seed=self.args.seed_val,
                                                num_workers=self.args.num_workers,
                                                rank=rank,
                                                world_size=world_size)

            # Get steps per epoch if not given by user
            if self.args.steps_per_epoch is None:
                self.args.steps_per_epoch = len(train_images) // self.args.batch_size
            else:
                self.args.steps_per_epoch = self.args.steps_per_epoch

            # Get loss function
            if self.data['modality'] != 'dose':  # Default training and val loss for dose. 
                loss_fn = get_loss(self.args, class_weights=self.class_weights)
            else:
                if self.args.loss == None:
                    self.args.loss = 'mse'  # Default for dose.  Same loss for train and val!
                    loss_fn = get_loss(self.args)  # Now we pass parameters and use default val loss mse
                else:
                    loss_fn = get_loss(self.args)  # Now we pass parameters
                    self.val_loss = get_loss(self.args)

            # Make sure we are using/have DTMs for boundary-based loss functions
            if self.args.loss in ["bl", "hdl", "gsl"]:
                assert self.args.use_dtms, f"For {self.args.loss}, use --use_dtms flag."
                assert len(train_images) == len(train_labels) == len(train_dtms), \
                    ("Number of distance transforms does not match number of training images and labels. Please "
                        "check that distance transforms were computed.")

            # Get model
            if self.args.model != "pretrained":
                model = get_model(**self.model_config)
            else:
                model = configure_pretrained_model(self.args.pretrained_model_path, self.n_channels, self.n_classes)  # Change for dose prediction???

            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.to(rank)
            model = DDP(model, device_ids=[rank])

            # Get optimizer and lr scheduler
            optimizer = get_optimizer(self.args, model)
            scheduler = get_lr_schedule(self.args, optimizer)

            # If using AMP, use gradient scaling
            if self.args.amp:
                scaler = torch.amp.GradScaler("cuda")

            # Only log metrics on first process (i.e., rank 0)
            if rank == 0:
                # Compute running averages for losses
                running_loss_train = Mean()
                running_loss_validation = Mean()

                # Initialize best validation loss
                best_loss = np.Inf

                # Set up tensorboard summary writer
                writer = SummaryWriter(os.path.join(self.args.results, "logs", "fold_{}".format(fold)))

                # Best model path
                best_model_name = os.path.join(self.args.results, "models", "fold_{}.pt".format(fold))

            # Function to perform a single training step
            def train_step(image, label, dtm, alpha):
                # Loss computation
                def compute_loss():
                    output = model(image)
                    
                    if self.args.use_dtms:
                        loss = loss_fn(label, output["prediction"], dtm, alpha)
                    elif self.args.loss in ["cldice"]:
                        loss = loss_fn(label, output["prediction"], alpha)
                    else:
                        loss = loss_fn(label, output["prediction"])

                    if self.args.deep_supervision:
                        for k, p in enumerate(output["deep_supervision"]):
                            if self.args.use_dtms:
                                loss += 0.5 ** (k + 1) * loss_fn(label, p, dtm, alpha)
                            elif self.args.loss in ["cldice"]:
                                loss += 0.5 ** (k + 1) * loss_fn(label, p, alpha)
                            else:
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

                return loss

            def val_step(image, label):
                pred = sliding_window_inference(image,
                                                roi_size=self.patch_size,
                                                overlap=self.args.val_sw_overlap,
                                                sw_batch_size=1,
                                                predictor=model,
                                                device=torch.device("cuda"))

                return self.val_loss(label, pred)

            for epoch in range(self.args.epochs):
                # Make sure gradient tracking is on, and do a pass over the data
                model.train(True)
                if rank == 0:
                    with TrainProgressBar(epoch + 1, fold, self.args.epochs, self.args.steps_per_epoch) as pb:
                        for i in range(self.args.steps_per_epoch):
                            data = train_loader.next()[0]
                            alpha = self.alpha(epoch)
                            if self.args.use_dtms:
                                image, label, dtm = data["image"], data["label"], data["dtm"]
                                loss = train_step(image, label, dtm, alpha)
                            else:
                                image, label = data["image"], data["label"]
                                if self.args.loss in ["cldice"]:
                                    loss = train_step(image, label, None, alpha)
                                else:
                                    loss = train_step(image, label, None, None)

                            # Update lr schedule
                            scheduler.step()  # Update scheduler.step(val_loss) due to use of reducelronplateau lrscheduler that requires a metrics

                            # Send all training losses to device 0 for average
                            dist.reduce(loss, dst=0)

                            # Print running loss to progress bar for rank 0 only
                            current_loss = loss.item() / world_size
                            running_loss = running_loss_train(current_loss)
                            pb.update(loss=running_loss)
                else:
                    for i in range(self.args.steps_per_epoch):
                        data = train_loader.next()[0]
                        alpha = self.alpha(epoch)
                        if self.args.use_dtms:
                            image, label, dtm = data["image"], data["label"], data["dtm"]
                            loss = train_step(image, label, dtm, alpha)
                        else:
                            image, label = data["image"], data["label"]
                            if self.args.loss in ["cldice"]:
                                loss = train_step(image, label, None, alpha)
                            else:
                                loss = train_step(image, label, None, None)

                        # Update lr schedule
                        scheduler.step()

                        # Send loss to device 0
                        dist.reduce(loss, dst=0)

                dist.barrier()

                # Start validation
                # We don"t need gradients on to do reporting
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
                        # Save best model. Why not doing this if rank1+0???
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
                    writer.add_scalars("Training vs. Validation Loss",
                                       {"Training": running_loss, "Validation": running_val_loss},
                                       epoch + 1)
                    writer.flush()

                    # Reset running losses for new epoch
                    running_loss_train.reset_states()
                    running_loss_validation.reset_states()

            dist.barrier()
            if rank == 0:
                writer.close()

        self.cleanup()

    def fit(self):
        # Train model
        world_size = torch.cuda.device_count()
        mp.spawn(
            self.train,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
