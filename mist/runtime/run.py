"""Training class for MIST."""
import os
from typing import Optional

import numpy as np
import pandas as pd
import rich
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter # type: ignore
from monai.inferers import sliding_window_inference # type: ignore

# Import MIST modules.
from mist.data_loading import dali_loader
from mist.models.model_loader import get_model
from mist.runtime import exceptions
from mist.runtime import loss_functions
from mist.runtime import progress_bar
from mist.runtime import utils
from mist.runtime.runtime_constants import RuntimeConstants as constants


# Define console for rich.
console = rich.console.Console()


class Trainer:
    """Training class for MIST.

    Attributes:
        mist_arguments: User defined arguments for MIST.
        file_paths: Paths to files like dataset description, config file, and
            model configuration file.
        data_structures: Data structures like dataset description and
            configuration data.
        boundary_loss_weighting_schedule: Weighting schedule for boundary loss
            functions.
        fixed_loss_functions: Loss function for validation and possibly others
            in the future.
    """

    def __init__(self, mist_arguments):
        """Initialize the trainer class."""
        # Store user arguments.
        self.mist_arguments = mist_arguments

        # Initialize data paths dictionary. This dictionary contains paths to
        # files like the dataset description, MIST configuration, model
        # configuration, and training paths dataframe.
        self._initialize_file_paths()

        # Initialize data structures. This function reads the dataset
        # description, MIST configuration, and training paths dataframe from the
        # corresponding files. These data structures are used during to set up
        # the training process.
        self._initialize_data_structures()

        # Set up model configuration. The model configuration saves parameters
        # like the model name, number of channels, number of classes, deep
        # supervision, deep supervision heads, pocket, patch size, target
        # spacing, and use of residual blocks. We use these parameters to build
        # the model during training and for inference.
        self._create_model_configuration()

        # Initialize fixed loss functions.
        self.fixed_loss_functions = {
            "validation": loss_functions.DiceLoss(exclude_background=True),
        }

    def _initialize_file_paths(self):
        """Initialize and store necessary file paths."""
        self.file_paths = {
            "dataset_description": self.mist_arguments.data,
            "mist_configuration": os.path.join(
                self.mist_arguments.results, "config.json"
            ),
            "model_configuration": os.path.join(
                self.mist_arguments.results, "models", "model_config.json"
            ),
            "training_paths_dataframe": os.path.join(
                self.mist_arguments.results, "train_paths.csv"
            ),
        }

    def _initialize_data_structures(self):
        """Read and store data structures such as configuration and paths."""
        # Initialize data structures dictionary.
        self.data_structures = {}

        # Check if the corresponding files exist. We omit the model
        # configuration file since it does not exist yet. The model
        # configuration will be created later.
        for file_path in (
            path for key, path in self.file_paths.items()
            if key != "model_configuration"
        ):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

        # Read the dataset description file.
        self.data_structures["dataset_description"] = utils.read_json_file(
            self.file_paths["dataset_description"]
        )

        # Read the MIST configuration file.
        self.data_structures["mist_configuration"] = utils.read_json_file(
            self.file_paths["mist_configuration"]
        )

        # Read the training paths dataframe.
        self.data_structures["training_paths_dataframe"] = pd.read_csv(
            self.file_paths["training_paths_dataframe"]
        )

    def _create_model_configuration(self):
        """Create model configuration.

        This function creates the model configuration based on the user
        arguments. This will either create a new model configuration or
        read an existing model configuration from a file (i.e., pretrained).
        """
        # Get the number of channels and classes from the dataset description.
        number_of_channels = len(
            self.data_structures["dataset_description"]["images"]
        )
        number_of_classes = len(
            self.data_structures["dataset_description"]["labels"]
        )

        # Save model blend mode and patch overlap in MIST configuration.
        self.data_structures["mist_configuration"]["patch_overlap"] = (
            self.mist_arguments.sw_overlap
        )
        self.data_structures["mist_configuration"]["patch_blend_mode"] = (
            self.mist_arguments.blend_mode
        )

        # If the model is not pretrained, create a new model configuration.
        # Update the patch size if the user overrides it.
        if self.mist_arguments.patch_size is not None:
            self.data_structures["mist_configuration"]["patch_size"] = (
                self.mist_arguments.patch_size
            )

        # Create a new model configuration based on user arguments.
        self.data_structures["model_configuration"] = {
            "model_name": self.mist_arguments.model,
            "n_channels": number_of_channels,
            "n_classes": number_of_classes,
            "deep_supervision": self.mist_arguments.deep_supervision,
            "pocket": self.mist_arguments.pocket,
            "patch_size": (
                self.data_structures["mist_configuration"]["patch_size"]
            ),
            "target_spacing": (
                self.data_structures["mist_configuration"]["target_spacing"]
            ),
            "use_res_block": self.mist_arguments.use_res_block,
        }

        # Save the model configuration to file.
        utils.write_json_file(
            self.file_paths["model_configuration"],
            self.data_structures["model_configuration"],
        )

        # Update the MIST configuration file with the inference parameters.
        utils.write_json_file(
            self.file_paths["mist_configuration"],
            self.data_structures["mist_configuration"],
        )

    # Set up for distributed training
    def setup(self, rank: int, world_size: int) -> None:
        """Set up for distributed training.

        Args:
            rank: Rank of the process.
            world_size: Number of processes.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = self.mist_arguments.master_port
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Clean up processes after distributed training
    def cleanup(self):
        """Clean up processes after distributed training."""
        dist.destroy_process_group()

    def train(self, rank: int, world_size: int) -> None:
        """Train the model.

        Args:
            rank: Rank of the process.
            world_size: Number of processes
        """
        # Set up for distributed training.
        self.setup(rank, world_size)

        # Set device rank for each process.
        torch.cuda.set_device(rank)

        # Display the start of training message.
        if rank == 0:
            text = rich.text.Text("\nStarting training\n") # type: ignore
            text.stylize("bold")
            console.print(text)

        # Start training for each fold.
        for fold in self.mist_arguments.folds:
            # Get training ids from dataframe.
            train_ids = list(
                self.data_structures["training_paths_dataframe"].loc[
                    self.data_structures["training_paths_dataframe"]["fold"]
                    != fold
                ]["id"]
            )

            # Get test ids from dataframe.
            test_ids = list(
                self.data_structures["training_paths_dataframe"].loc[
                    self.data_structures["training_paths_dataframe"]["fold"]
                    == fold
                ]["id"]
            )

            # Get list of training images and labels.
            train_images = utils.get_numpy_file_paths_list(
                base_dir=self.mist_arguments.numpy,
                folder="images",
                patient_ids=train_ids,
            )
            train_labels = utils.get_numpy_file_paths_list(
                base_dir=self.mist_arguments.numpy,
                folder="labels",
                patient_ids=train_ids,
            )

            # Get list of validation images and labels.
            val_images = utils.get_numpy_file_paths_list(
                base_dir=self.mist_arguments.numpy,
                folder="images",
                patient_ids=test_ids,
            )
            val_labels = utils.get_numpy_file_paths_list(
                base_dir=self.mist_arguments.numpy,
                folder="labels",
                patient_ids=test_ids,
            )

            # If we are using distance transform maps, get the list of training
            # distance transform maps.
            if self.mist_arguments.use_dtms:
                train_dtms = utils.get_numpy_file_paths_list(
                    base_dir=self.mist_arguments.numpy,
                    folder="dtms",
                    patient_ids=train_ids,
                )
            else:
                train_dtms = None

            # Split training data into training and validation sets if the
            # validation percentage is greater than zero. The idea here is to
            # leave the original validation set as an unseen test set and use
            # the smaller partition of the training dataset as the validation
            # set to pick the best model.
            if self.mist_arguments.val_percent > 0:
                (
                    train_images,
                    val_images,
                    train_labels,
                    val_labels,
                ) = train_test_split(
                    train_images,
                    train_labels,
                    test_size=self.mist_arguments.val_percent,
                    random_state=self.mist_arguments.seed_val,
                )

                # If we are using distance transform maps, split them as well.
                if self.mist_arguments.use_dtms:
                    train_dtms, _ = train_test_split(
                        train_dtms,
                        test_size=self.mist_arguments.val_percent,
                        random_state=self.mist_arguments.seed_val,
                    )

            # The number of validation images must be greater than or equal to
            # the number of GPUs used for training.
            if len(val_images) < world_size:
                raise exceptions.InsufficientValidationSetError(
                    val_size=len(val_images), world_size=world_size
                )

            # Get number of validation steps. This is the number of validation
            # images divided by the number of GPUs (i.e., the world size).
            val_steps = len(val_images) // world_size

            # Get training data loader.
            # The training labels are different from what's specified in the
            # dataset description. The preprocessed masks have labels 0,1,...,N.
            # We exclude the background label (0) from the training labels and
            # pass the rest in as the labels for the training data loader.
            training_labels = list(range(len(
                self.data_structures["mist_configuration"]["labels"]
            )))[1:]
            train_loader = dali_loader.get_training_dataset(
                image_paths=train_images,
                label_paths=train_labels,
                dtm_paths=train_dtms,
                batch_size=self.mist_arguments.batch_size // world_size,
                oversampling=self.mist_arguments.oversampling,
                labels=training_labels,
                roi_size=(
                    self.data_structures["mist_configuration"]["patch_size"]
                ),
                seed=self.mist_arguments.seed_val,
                num_workers=self.mist_arguments.num_workers,
                rank=rank,
                world_size=world_size,
                extract_patches=True,
                use_augmentation=not self.mist_arguments.no_augmentation,
                use_flips=not self.mist_arguments.augmentation_no_flips,
                use_blur=not self.mist_arguments.augmentation_no_blur,
                use_noise=not self.mist_arguments.augmentation_no_noise,
                use_brightness=(
                    not self.mist_arguments.augmentation_no_brightness
                ),
                use_contrast=not self.mist_arguments.augmentation_no_contrast,
                use_zoom=not self.mist_arguments.augmentation_no_zoom,
            )

            # Get validation data loader.
            validation_loader = dali_loader.get_validation_dataset(
                image_paths=val_images,
                label_paths=val_labels,
                seed=self.mist_arguments.seed_val,
                num_workers=self.mist_arguments.num_workers,
                rank=rank,
                world_size=world_size
            )

            # Get steps per epoch, number of epochs, and validation parameters
            # like validate after n epochs and validate every n epochs.
            epochs_and_validation_params = (
                utils.get_epochs_and_validation_params(
                    mist_arguments=self.mist_arguments,
                    num_train_examples=len(train_images),
                    num_optimization_steps=constants.TOTAL_OPTIMIZATION_STEPS,
                    validate_every_n_steps=constants.VALIDATE_EVERY_N_STEPS,
                )
            )
            steps_per_epoch = epochs_and_validation_params["steps_per_epoch"]
            epochs = epochs_and_validation_params["epochs"]
            validate_every_n_epochs = (
                epochs_and_validation_params["validate_every_n_epochs"]
            )
            validate_after_n_epochs = (
                epochs_and_validation_params["validate_after_n_epochs"]
            )

            # Initialize boundary loss weighting schedule.
            boundary_loss_weighting_schedule = utils.AlphaSchedule(
                n_epochs=epochs,
                schedule=self.mist_arguments.boundary_loss_schedule,
                constant=self.mist_arguments.loss_schedule_constant,
                init_pause=self.mist_arguments.linear_schedule_pause,
                step_length=self.mist_arguments.step_schedule_step_length,
            )

            # Get loss function based on user arguments.
            loss_fn = loss_functions.get_loss(self.mist_arguments)
            loss_fn_with_deep_supervision = (
                loss_functions.DeepSupervisionLoss(loss_fn)
            )

            # Make sure we are using/have DTMs for boundary-based loss
            # functions.
            if self.mist_arguments.loss in ["bl", "hdl", "gsl"]:
                if not self.mist_arguments.use_dtms:
                    raise ValueError(
                        f"For loss function '{self.mist_arguments.loss}', the "
                        "--use-dtms flag must be enabled."
                    )

                if train_dtms:
                    # Check if the number of training images, labels, and
                    # distance transforms match. If not, raise an error.
                    if not(
                        len(train_images) == len(train_labels) == len(
                            train_dtms
                        )
                    ):
                        raise ValueError(
                            "Mismatch in the number of training images, "
                            "labels, and distance transforms. Ensure that the "
                            "number of distance transforms matches the number "
                            "of training images and labels. Found "
                            f"{len(train_images)} training images, "
                            f"{len(train_labels)} training labels, and "
                            f"{len(train_dtms)} training distance transforms."
                        )

            # Define the model from the model configuration file.
            model = get_model(**self.data_structures["model_configuration"])

            # Make batch normalization compatible with DDP.
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            # Set up model for distributed data parallel training.
            model.to(rank)
            if self.mist_arguments.model != "pretrained":
                model = DDP(model, device_ids=[rank])
            else:
                # This seems to work with pretrained models. We will need to
                # test this further.
                model = DDP(
                    model, device_ids=[rank], find_unused_parameters=True
                )

            # Get optimizer and lr scheduler
            optimizer = utils.get_optimizer(self.mist_arguments, model)
            learning_rate_scheduler = utils.get_lr_schedule(
                self.mist_arguments, optimizer, epochs
            )

            # Float16 inputs during the forward pass produce float16 gradients
            # in the backward pass. Small gradient values may underflow to zero,
            # causing updates to corresponding parameters to be lost. To prevent
            # this, "gradient scaling" multiplies the loss by a scale factor
            # before the backward pass, increasing gradient magnitudes to avoid
            # underflow. Gradients must be unscaled before the optimizer updates
            # the parameters to ensure the learning rate is unaffected.
            if self.mist_arguments.amp:
                amp_gradient_scaler = torch.amp.GradScaler("cuda") # type: ignore

            # Only log metrics on first process (i.e., rank 0).
            if rank == 0:
                # Compute running averages for training and validation losses.
                running_loss_train = utils.RunningMean()
                running_loss_validation = utils.RunningMean()

                # Initialize best validation loss to infinity.
                best_validation_loss = np.inf # type: ignore

                # Set up tensorboard summary writer.
                writer = SummaryWriter(
                    os.path.join(
                        self.mist_arguments.results, "logs", f"fold_{fold}"
                    )
                )

                # Path and name for best model for this fold.
                best_model_name = os.path.join(
                    self.mist_arguments.results, "models", f"fold_{fold}.pt"
                )

            def train_step(
                    image: torch.Tensor,
                    label: torch.Tensor,
                    dtm: Optional[torch.Tensor],
                    alpha: Optional[float],
            ) -> torch.Tensor:
                """Perform a single training step.

                Args:
                    image: Input image.
                    label: Ground truth label.
                    dtm: Distance transform map.
                    alpha: Weighting factor for boundary-based loss functions.

                Returns:
                    loss: Loss value for the batch.
                """
                # Compute loss for the batch.
                def compute_loss() -> torch.Tensor:
                    """Compute loss for the batch.

                    Args:
                        None

                    Returns:
                        loss: Loss value for the batch.
                    """
                    # Make predictions for the batch.
                    output = model(image) # pylint: disable=cell-var-from-loop

                    # Compute loss based on the output and ground truth label.
                    # Apply deep supervision if enabled.
                    y_supervision = (
                        output["deep_supervision"]
                        if self.mist_arguments.deep_supervision
                        else None
                    )
                    loss = loss_fn_with_deep_supervision( # pylint: disable=cell-var-from-loop
                        y_true=label,
                        y_pred=output["prediction"],
                        y_supervision=y_supervision,
                        alpha=alpha,
                        dtm=dtm,
                    )

                    # L2 regularization term. This term adds a penalty to the
                    # loss based on the L2 norm of the model's parameters.
                    if self.mist_arguments.l2_reg:
                        l2_norm_of_model_parameters = 0.0
                        for param in model.parameters(): # pylint: disable=cell-var-from-loop
                            l2_norm_of_model_parameters += (
                                torch.norm(param, p=2)
                            )

                        # Update the loss with the L2 regularization term scaled
                        # by the l2_penalty parameter.
                        loss += (
                            self.mist_arguments.l2_penalty *
                            l2_norm_of_model_parameters
                        )

                    # L1 regularization term. This term adds a penalty to the
                    # loss based on the L1 norm of the model's parameters.
                    if self.mist_arguments.l1_reg:
                        l1_norm_of_model_parameters = 0.0
                        for param in model.parameters(): # pylint: disable=cell-var-from-loop
                            l1_norm_of_model_parameters += (
                                torch.norm(param, p=1)
                            )

                        # Update the loss with the L1 regularization term scaled
                        # by the l1_penalty parameter.
                        loss += (
                            self.mist_arguments.l1_penalty *
                            l1_norm_of_model_parameters
                        )
                    return loss

                # Zero out the gradients from the previous batch.
                # Gradients accumulate by default in PyTorch, so it's important
                # to reset them at the start of each training iteration to avoid
                # interference from prior batches.
                optimizer.zero_grad() # pylint: disable=cell-var-from-loop

                # Check if automatic mixed precision (AMP) is enabled for this
                # training step.
                if self.mist_arguments.amp:
                    # AMP is used to speed up training and reduce memory usage
                    # by performing certain operations in lower precision
                    # (e.g., float16). This can improve the efficiency of
                    # training on GPUs without significant loss in accuracy.

                    # Use `torch.autocast` to automatically handle mixed
                    # precision operations on the GPU. This context manager
                    # ensures that certain operations are performed in float16
                    # precision while others remain in float32, depending
                    # on what is most efficient and appropriate.
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        # Perform the forward pass and compute the loss using
                        # mixed precision.
                        loss = compute_loss()

                    # Backward pass: Compute gradients by scaling the loss to
                    # prevent underflow. Scaling is necessary when using AMP
                    # because very small gradients in float16 could underflow
                    # (become zero) during training. The scaler multiplies the
                    # loss by a large factor before computing the gradients to
                    # mitigate underflow.
                    amp_gradient_scaler.scale(loss).backward() # pylint: disable=cell-var-from-loop

                    # If gradient clipping is enabled, apply it after unscaling
                    # the gradients. Gradient clipping prevents exploding
                    # gradients by limiting the magnitude of the gradients to a
                    # specified maximum value (clip_norm_max).
                    if self.mist_arguments.clip_norm:
                        # Unscale the gradients before clipping, as they were
                        # previously scaled.
                        amp_gradient_scaler.unscale_(optimizer) # pylint: disable=cell-var-from-loop

                        # Clip gradients to the maximum norm (clip_norm_max) to
                        # stabilize training.
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), # pylint: disable=cell-var-from-loop
                            self.mist_arguments.clip_norm_max
                        )

                    # Perform the optimizer step to update the model parameters.
                    # This step adjusts the model's weights based on the
                    # computed gradients.
                    amp_gradient_scaler.step(optimizer) # pylint: disable=cell-var-from-loop

                    # Update the scaler after each iteration. This adjusts the
                    # scale factor used to prevent underflows or overflows in
                    # the future. The scaler increases or decreases the scaling
                    # factor dynamically based on whether gradients overflow.
                    amp_gradient_scaler.update() # pylint: disable=cell-var-from-loop
                else:
                    # If AMP is not enabled, perform the forward pass and
                    # compute the loss using float32 precision.
                    loss = compute_loss()

                    # Compute the loss and its gradients.
                    loss.backward()

                    # Apply gradient clipping if enabled.
                    if self.mist_arguments.clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), # pylint: disable=cell-var-from-loop
                            self.mist_arguments.clip_norm_max
                        )

                    # Perform the optimizer step to update the model parameters.
                    optimizer.step() # pylint: disable=cell-var-from-loop
                return loss

            def val_step(
                    image: torch.Tensor,
                    label: torch.Tensor
            ) -> torch.Tensor:
                """Perform a single validation step.

                Args:
                    image: Input image.
                    label: Ground truth label.

                Returns:
                    loss: Loss value for the batch.
                """
                pred = sliding_window_inference(
                    image,
                    roi_size=(
                        self.data_structures["mist_configuration"][
                            "patch_size"
                        ]
                    ),
                    overlap=self.mist_arguments.val_sw_overlap,
                    sw_batch_size=1,
                    predictor=model, # pylint: disable=cell-var-from-loop
                    device=torch.device("cuda")
                )

                return self.fixed_loss_functions["validation"](label, pred)

            # Train the model for the specified number of epochs.
            for epoch in range(epochs):
                # Make sure gradient tracking is on, and do a pass over the
                # training data.
                model.train(True)

                # Compute alpha for boundary loss functions. The alpha parameter
                # is used to weight the boundary loss function with a region-
                # based loss function like dice or cross entropy.
                alpha = boundary_loss_weighting_schedule(epoch)
                # Only log metrics on first process (i.e., rank 0).
                if rank == 0:
                    with progress_bar.TrainProgressBar(
                        epoch + 1,
                        fold,
                        epochs,
                        steps_per_epoch,
                    ) as pb:
                        for _ in range(steps_per_epoch):
                            # Get data from training loader.
                            data = train_loader.next()[0]

                            if self.mist_arguments.use_dtms:
                                # Use distance transform maps for boundary-based
                                # loss functions. In this case, we pass them
                                # and the alpha parameter to the train_step.
                                image, label, dtm = (
                                    data["image"], data["label"], data["dtm"]
                                )

                                # Perform a single training step. Return
                                # the loss for the batch.
                                loss = train_step(image, label, dtm, alpha)
                            else:
                                # If distance transform maps are not used, pass
                                # None for the dtm parameter. If we are using
                                # cldice loss, pass the alpha parameter to the
                                # train_step. Otherwise, pass None.
                                image, label = data["image"], data["label"]
                                if self.mist_arguments.loss in ["cldice"]:
                                    loss = train_step(image, label, None, alpha)
                                else:
                                    loss = train_step(image, label, None, None)

                            # Send all training losses to device 0 to add them.
                            dist.reduce(loss, dst=0)

                            # Average the loss across all GPUs.
                            current_loss = loss.item() / world_size

                            # Update the running loss for the progress bar.
                            running_loss = running_loss_train(current_loss)

                            # Update the progress bar with the running loss and
                            # learning rate.
                            pb.update(
                                loss=running_loss,
                                lr=optimizer.param_groups[0]["lr"]
                            )
                else:
                    # For all other processes, do not display the progress bar.
                    # Repeat the training steps shown above for the other GPUs.
                    for _ in range(steps_per_epoch):
                        # Get data from training loader.
                        data = train_loader.next()[0]

                        if self.mist_arguments.use_dtms:
                            image, label, dtm = (
                                data["image"], data["label"], data["dtm"]
                            )
                            loss = train_step(image, label, dtm, alpha)
                        else:
                            image, label = data["image"], data["label"]
                            if self.mist_arguments.loss in ["cldice"]:
                                loss = train_step(image, label, None, alpha)
                            else:
                                loss = train_step(image, label, None, None)

                        # Update the learning rate scheduler.
                        learning_rate_scheduler.step()

                        # Send the loss on the current GPU to device 0.
                        dist.reduce(loss, dst=0)

                # Update the learning rate scheduler.
                learning_rate_scheduler.step()

                # Wait for all processes to finish the epoch.
                dist.barrier()

                # Start validation. We don't need gradients on to do reporting.
                # Only validate on the first and last epochs or periodically
                # after validate_after_n_epochs.
                validate = (
                    epoch == 0 or epoch == epochs - 1 or
                    (
                        epoch >= validate_after_n_epochs and
                        epoch % validate_every_n_epochs == 0
                    )
                )
                if validate:
                    model.eval()
                    with torch.no_grad():
                        # Only log metrics on first process (i.e., rank 0).
                        if rank == 0:
                            with progress_bar.ValidationProgressBar(
                                val_steps
                            ) as val_pb:
                                for _ in range(val_steps):
                                    # Get data from validation loader.
                                    data = validation_loader.next()[0]
                                    image, label = data["image"], data["label"]

                                    # Compute loss for single validation step.
                                    val_loss = val_step(image, label)

                                    # Send all validation losses to device 0 to
                                    # add them.
                                    dist.reduce(val_loss, dst=0)

                                    # Average the loss across all GPUs.
                                    current_val_loss = (
                                        val_loss.item() / world_size
                                    )

                                    # Update the running loss for the progress
                                    # bar.
                                    running_val_loss = running_loss_validation(
                                        current_val_loss
                                    )

                                    # Update the progress bar with the running
                                    # loss.
                                    val_pb.update(loss=running_val_loss)

                            # Check if validation loss is lower than the current
                            # best validation loss. If so, save the model.
                            if running_val_loss < best_validation_loss: # type: ignore
                                text = rich.text.Text( # type: ignore
                                    "Validation loss IMPROVED from "
                                    f"{best_validation_loss:.4} "
                                    f"to {running_val_loss:.4}\n"
                                )
                                text.stylize("bold")
                                console.print(text)

                                # Update the current best validation loss.
                                best_validation_loss = running_val_loss

                                # Save the model with the best validation loss.
                                torch.save(model.state_dict(), best_model_name)
                            else:
                                # Otherwise, log that the validation loss did
                                # not improve and display the best validation
                                # loss. Continue training with current model.
                                text = rich.text.Text( # type: ignore
                                    "Validation loss did NOT improve from "
                                    f"{best_validation_loss:.4}\n"
                                )
                                console.print(text)
                        else:
                            # Repeat the validation steps for the other GPUs. Do
                            # not display the progress bar for these GPUs.
                            for _ in range(val_steps):
                                # Get data from validation loader.
                                data = validation_loader.next()[0]
                                image, label = data["image"], data["label"]

                                # Compute loss for single validation step.
                                val_loss = val_step(image, label)

                                # Send the loss on the current GPU to device 0.
                                dist.reduce(val_loss, dst=0)

                # Reset training and validation loader after each epoch.
                validation_loader.reset()
                train_loader.reset()

                # Log the running loss for training and validation after each
                # epoch. Only log metrics on first process (i.e., rank 0).
                if rank == 0:
                    # Log the running loss for validation.
                    summary_data = {
                        "Training": running_loss,
                        "Validation": running_val_loss,
                    }
                    writer.add_scalars(
                        "Training vs. Validation Loss",
                        summary_data,
                        epoch + 1,
                    )
                    writer.flush()

                    # Reset running losses for new epoch.
                    running_loss_train.reset_states()
                    running_loss_validation.reset_states()

            # Wait for all processes to finish the fold.
            dist.barrier()

            # Close the tensorboard summary writer after each fold. Only
            # close the writer on the first process (i.e., rank 0).
            if rank == 0:
                writer.close()

        # Clean up processes after distributed training.
        self.cleanup()

    def fit(self):
        """Fit the model using multiprocessing.

        This function uses multiprocessing to train the model on multiple GPUs.
        It uses the `torch.multiprocessing.spawn` function to create multiple
        instances of the training function, each on a separate GPU.
        """
        # Train model.
        world_size = torch.cuda.device_count()
        if world_size > 1:
            mp.spawn( # type: ignore
                self.train,
                args=(world_size,),
                nprocs=world_size,
                join=True,
            )
        # To enable pdb do not spawn multiprocessing for world_size = 1.
        else:
            self.train(0, world_size)
