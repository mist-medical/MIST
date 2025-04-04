"""Loss function implementations for training segmentation models."""
from typing import Tuple, Optional, Callable
import argparse
import torch
from torch import nn
from torch.nn import functional as F

from mist.runtime import loss_utils


class DeepSupervisionLoss(nn.Module):
    """Loss function for deep supervision in segmentation tasks.

    This class calculates the loss for the main output and additional deep
    supervision heads using a geometric weighting scheme. Deep supervision
    provides intermediate outputs during training to guide the model's learning
    at multiple stages.

    Attributes:
        loss_fn: The base loss function to apply (e.g., Dice loss).
        scaling_fn: A function to scale the loss for each supervision head.
            Defaults to geometric scaling by 0.5 ** k, where k is the index.
    """
    def __init__(
            self,
            loss_fn: nn.Module,
            scaling_fn: Optional[Callable[[int], float]]=None
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.scaling_fn = scaling_fn or (lambda k: 0.5 ** k)

    def apply_loss(self, y_true, y_pred, alpha=None, dtm=None):
        """Applies the configured loss function with appropriate arguments."""
        if dtm is not None:
            return self.loss_fn(y_true, y_pred, dtm, alpha)
        if alpha is not None:
            return self.loss_fn(y_true, y_pred, alpha)
        return self.loss_fn(y_true, y_pred)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_supervision: Optional[Tuple[torch.Tensor, ...]] = None,
        alpha: Optional[float] = None,
        dtm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the total loss, including contributions from deep supervision.

        Args:
            y_true: Ground truth mask of shape (batch_size, 1, height, width,
                depth). This mask is not one-hot encoded because of the we way
                we construct the data loader. The one-hot encoding is applied
                in the forward pass of the loss function.
            y_pred: Predicted main output of shape  (batch_size, num_classes,
                height, width, depth). This is the main output of the network.
                We assume that the predicted mask is the raw output of a network
                that has not been passed through a softmax function. We apply
                the softmax function in the forward pass of the loss function.
            y_supervision (optional): Deep supervision outputs, each of shape
                (batch_size, num_classes, height, width, depth). Like y_pred,
                these are raw outputs of the network. We apply the softmax
                function in the forward pass of the loss function.
            alpha (optional): Balances region and boundary losses. This is a
                hyperparameter that should be in the interval [0, 1].
            dtm (optional): Distance transform maps for boundary-based loss.

        Returns:
            The total weighted loss.
        """
        # Collect main prediction and deep supervision outputs.
        _y_pred = [y_pred] + (list(y_supervision) if y_supervision else [])

        # Compute weighted loss.
        losses = torch.stack(
            [
                self.scaling_fn(k) * self.apply_loss(y_true, pred, alpha, dtm)
                for k, pred in enumerate(_y_pred)
            ]
        )

        # Normalize using the sum of the scaling factors.
        normalization = sum(self.scaling_fn(k) for k in range(len(_y_pred)))
        return losses.sum() / normalization


class DiceLoss(nn.Module):
    """Soft Dice loss function for segmentation tasks.

    For each class, the Dice loss is defined as:
        L(x, y) = ||x - y||^2 / (||x||^2 + ||y||^2 + smooth)

    We then take the mean of the Dice loss across all classes. By default, the
    Dice loss function includes the background class.

    Attributes:
        smooth: A small constant to prevent division by zero.
        axes: The axes along which to compute the Dice loss.
        include_background: Whether to include the background class in the Dice
            loss computation.
        exclude_background: Whether to exclude the background class in the Dice
            loss computation. This defaults to False.
    """
    def __init__(self, exclude_background: bool=False):
        super().__init__()
        # Smooth constant to prevent division by zero.
        self.smooth = 1e-6

        # The axes along which to compute the Dice loss. The tensors are assumed
        # to have shape (batch_size, num_classes, height, width, depth) for 3D
        # data. The axes are (2, 3, 4) for 3D data.
        self.axes = (2, 3, 4)

        # Indicates whether to exclude the background class in the Dice loss
        # computation. By default, we include the background class.
        self.exclude_background = exclude_background

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the Dice loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, 1, height, width, depth). This is not one-hot
                encoded. We do not one-hot encode the ground truth mask because
                of the way the data is loaded. We apply one-hot encoding in the
                forward pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.

        Returns:
            The Dice loss, which is one minus the Dice coefficient.
        """
        # Check inputs.
        loss_utils.check_loss_fn_inputs(y_true, y_pred)

        # Prepare inputs.
        # Apply one-hot encoding to the ground truth mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1])

        # Apply the softmax function to the predicted mask to turn this into a
        # probability mask.
        y_pred = F.softmax(y_pred, dim=1)

        if self.exclude_background:
            # Exclude the background class from the computation.
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        # Compute the numerator and denominator of the Dice loss.
        numerator = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        denominator = (
            torch.sum(torch.square(y_true), dim=self.axes) +
            torch.sum(torch.square(y_pred), dim=self.axes) +
            self.smooth
        )

        # Average the Dice loss across all classes.
        loss = torch.mean(numerator / denominator, dim=1)

        # Average the Dice loss across the batch.
        loss = torch.mean(loss)
        return loss

class DiceCELoss(DiceLoss):
    """Dice loss combined with cross entropy loss for segmentation tasks.

    This loss is defined as the mean of the Dice loss and the cross entropy loss
    between the predicted and ground truth segmentation masks. The Dice loss is
    inherited from the DiceLoss class. The cross entropy loss is computed using
    PyTorch's CrossEntropyLoss.

    Attributes:
        cross_entropy: The cross entropy loss function.
    """
    def __init__(self, exclude_background: bool = False):
        super().__init__(exclude_background=exclude_background)

        # Cross entropy loss function.
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the Dice cross entropy loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, 1, height, width, depth). This is not one-hot
                encoded. We do not one-hot encode the ground truth mask because
                of the way the data is loaded. We apply one-hot encoding in the
                forward pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.

        Returns:
            The Dice cross entropy loss.
        """
        # Compute the Dice loss using the inherited logic.
        loss_dice = super().forward(y_true, y_pred)

        # Prepare inputs for the cross entropy loss. PyTorch's cross entropy
        # loss function already applies the softmax function to the prediction.
        # We only need to apply one-hot encoding to the ground truth mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1]).to(torch.float)

        # Exclude the background class from the computation if necessary.
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]

        # Compute cross entropy loss.
        loss_ce = self.cross_entropy(y_pred, y_true)

        # Combine Dice and Cross Entropy losses.
        return 0.5 * (loss_ce + loss_dice)


class SoftCLDice(nn.Module):
    """Soft clDice loss function for segmentation tasks.

    This loss function is proposed in the following paper:

    https://arxiv.org/abs/2003.07311

    It is shown to be good for segmenting tubular structures. The cl in clDice
    stands for "center line". This loss function uses a soft skeletonization
    operation to compute the center line of the predicted and ground truth. The
    loss then compares these center lines.

    Attributes:
        iterations: The number of iterations to use in the soft skeletonization
            operation.
        smooth: A small constant to prevent division by zero.
        soft_skeletonize: The soft skeletonization operation.
        exclude_background: Whether to exclude the background class in the loss
            computation. This defaults
    """
    def __init__(
            self,
            iterations: int=10,
            smooth: float=1.,
            exclude_background: bool=False,
    ):
        super().__init__()
        # Number of iterations to use in the soft skeletonization operation.
        self.iterations = iterations

        # Smooth constant to prevent division by zero.
        self.smooth = smooth

        # Soft skeletonization operation.
        self.soft_skeletonize = loss_utils.SoftSkeletonize(
            num_iter=self.iterations
        )

        # Indicates whether to exclude the background class in the loss.
        self.exclude_background = exclude_background

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the soft clDice loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, height, width, depth). This is not one-hot encoded.
                We do not one-hot encode the ground truth mask because of the
                way the data is loaded. We apply one-hot encoding in the forward
                pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.

        Returns:
            The soft clDice loss.
        """
        # Prepare inputs. Apply one-hot encoding to the ground truth mask.
        # Apply the softmax function to the predicted mask to turn this into a
        # probability mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1]).to(torch.float)
        y_pred = F.softmax(y_pred, dim=1)

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]

        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (
            (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) /
            (torch.sum(skel_pred) + self.smooth)
        )
        tsens = (
            (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) /
            (torch.sum(skel_true) + self.smooth)
        )
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class SoftDiceCLDice(nn.Module):
    """Soft Dice CE loss combined with soft clDice loss for segmentation tasks.

    This is a weighted combination of the Dice CE and clDice loss functions. The
    weighted combination takes the following form:

            alpha * DiceLoss + (1 - alpha) * clDiceLoss

    where alpha is in the interval [0, 1]. The hyperparameter alpha can be
    fixed or used in a dynamic manner.

    Attributes:
        iterations: The number of iterations to use in the soft skeletonization
            operation.
        smooth: A small constant to prevent division by zero.
        dicece_loss: The Dice cross entropy loss function.
        exclude_background: Whether to exclude the background class in the loss
            computation. This defaults to False.
        cldice_loss: The soft clDice loss function.
    """
    def __init__(
            self,
            iterations: int=10,
            smooth: float=1.,
            exclude_background: bool=False
    ):
        super().__init__()
        # Number of iterations to use in the soft skeletonization operation.
        self.iterations = iterations

        # Smooth constant to prevent division by zero.
        self.smooth = smooth

        # Indicates whether to exclude the background class in the loss.
        self.exclude_background = exclude_background

        # Dice cross entropy loss function.
        self.dicece_loss = DiceCELoss(
            exclude_background=self.exclude_background
        )

        # Soft clDice loss function.
        self.cldice_loss = SoftCLDice(
            iterations=self.iterations,
            smooth=self.smooth,
            exclude_background=self.exclude_background,
        )

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            alpha: float,
    ):
        """Forward pass of the soft Dice CE and soft clDice loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, height, width, depth). This is not one-hot encoded.
                We do not one-hot encode the ground truth mask because of the
                way the data is loaded. We apply one-hot encoding in the forward
                pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.
            alpha: The hyperparameter that controls the balance between the Dice
                cross entropy and clDice loss functions. This should be in the
                interval [0, 1].
        """
        dicece_loss = self.dicece_loss(y_true, y_pred)
        cldice = self.cldice_loss(y_true, y_pred)
        return alpha * dicece_loss + (1. - alpha) * cldice


class BoundaryLoss(nn.Module):
    """Boundary loss function for segmentation tasks.

    The boundary loss is defined as the mean of the distance transform map (DTM)
    multiplied by the predicted segmentation mask. The distance transform map is
    computed from the ground truth segmentation mask.

    Attributes:
        region_loss: The region based loss function. We use the Dice cross
            entropy loss function here.
        exclude_background: Whether to exclude the background class in the loss
            computation. This defaults to False.
    """
    def __init__(self, exclude_background=False):
        super().__init__()
        self.exclude_background = exclude_background
        self.region_loss = DiceCELoss(
            exclude_background=self.exclude_background
        )

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            dtm: torch.Tensor,
            alpha: float,
    ) -> torch.Tensor:
        """Forward pass of the boundary loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, height, width, depth). This is not one-hot encoded.
                We do not one-hot encode the ground truth mask because of the
                way the data is loaded. We apply one-hot encoding in the forward
                pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.
            dtm: The distance transform map. The tensor has shape
                (batch_size, num_classes, height, width, depth). This is
                computed from the ground truth segmentation mask.
            alpha: The hyperparameter that controls the balance between the Dice
                cross entropy and boundary loss functions. This should be in the
                interval [0, 1].
        """
        # Compute region based loss.
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs for the boundary loss. We only need to apply the
        # softmax function to the predicted mask to turn this into a probability
        # mask. The distance transform map is already computed from the ground
        # truth mask.
        y_pred = F.softmax(y_pred, dim=1)

        # Exclude the background class from the computation if necessary.
        if self.exclude_background:
            y_pred = y_pred[:, 1:, :, :, :]
            dtm = dtm[:, 1:, :, :, :]

        # Compute boundary loss.
        boundary_loss = torch.mean(dtm * y_pred)

        # Return the weighted sum of the region and boundary loss.
        return alpha * region_loss + (1. - alpha) * boundary_loss


class HDOneSidedLoss(nn.Module):
    """One-sided Hausdorff distance loss function for segmentation tasks.

    The one-sided Hausdorff distance loss is defined as the weighted sum of the
    region based loss and the boundary loss. The region based loss is given by
    the Dice cross entropy loss function. The boundary loss is computed as the
    mean of the square of the difference between the predicted and ground truth
    masks multiplied by the square of the distance transform map.

    This loss function is a one-sided estimation of the Hausdorff distance.

    Attributes:
        region_loss: The region based loss function. We use the Dice cross
            entropy loss function here.
        exclude_background: Whether to exclude the background class in the loss
            computation. This defaults to False.
    """
    def __init__(self, exclude_background=False):
        super().__init__()
        self.exclude_background = exclude_background
        self.region_loss = DiceCELoss(
            exclude_background=self.exclude_background
        )

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            dtm: torch.Tensor,
            alpha: float,
    ) -> torch.Tensor:
        """Forward pass of the one-sided Hausdorff distance loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, height, width, depth). This is not one-hot encoded.
                We do not one-hot encode the ground truth mask because of the
                way the data is loaded. We apply one-hot encoding in the forward
                pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.
            dtm: The distance transform map. The tensor has shape
                (batch_size, num_classes, height, width, depth). This is
                computed from the ground truth segmentation mask.
            alpha: The hyperparameter that controls the balance between the Dice
                cross entropy and boundary loss functions. This should be in the
                interval [0, 1].
        """
        # Compute region based loss.
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs for the boundary loss. We get the one-hot encoding of
        # the ground truth mask and apply the softmax function to the predicted
        # mask to turn this into a probability mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1])
        y_pred = F.softmax(y_pred, dim=1)

        # Exclude the background class from the computation if necessary.
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]
            dtm = dtm[:, 1:, :, :, :]

        # Compute the one-sided Hausdorff distance loss.
        one_sided_hd_loss = torch.mean(
            torch.square(y_true - y_pred) * torch.square(dtm)
        )

        # Return the weighted sum of the region and one-sided Hausdorff distance
        # loss.
        return alpha * region_loss + (1. - alpha) * one_sided_hd_loss


class GenSurfLoss(nn.Module):
    """Generalized surface loss function for segmentation tasks.

    The generalized surface loss is defined as the weighted sum of the region
    based loss and the boundary loss. The region based loss is given by the Dice
    cross entropy loss function. The boundary loss is given by the following:

        L(x, y) = 1 - ||DTM * (1 - (x + y))||^2 / ||DTM||^2

    The distance transform map (DTM) is computed from the ground truth
    segmentation mask. This loss function tries to reconstruct the L2 norm of
    the distance transform map.

    For more details, see the following paper:
        https://arxiv.org/abs/2302.03868

    Attributes:
        region_loss: The region based loss function. We use the Dice cross
            entropy loss function here.
        exclude_background: Whether to exclude the background class in the loss
            computation. This defaults to False.
        smooth: A small constant to prevent division by zero.
        axes: The axes along which to compute the Dice loss.
    """
    def __init__(
        self,
        exclude_background=False
    ):
        super().__init__()
        self.exclude_background = exclude_background
        self.region_loss = DiceCELoss(self.exclude_background)
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            dtm: torch.Tensor,
            alpha: float,
    ) -> torch.Tensor:
        """Forward pass of the generalized surface loss function.

        Args:
            y_true: The ground truth segmentation mask. The tensor has shape
                (batch_size, height, width, depth). This is not one-hot encoded.
                We do not one-hot encode the ground truth mask because of the
                way the data is loaded. We apply one-hot encoding in the forward
                pass.
            y_pred: The predicted segmentation mask. The tensor has shape
                (batch_size, num_classes, height, width, depth). We assume that
                the predicted mask is the raw output of a network that has not
                been passed through a softmax function. We apply the softmax
                function in the forward pass.
            dtm: The distance transform map. The tensor has shape
                (batch_size, num_classes, height, width, depth). This is
                computed from the ground truth segmentation mask.
            alpha: The hyperparameter that controls the balance between the Dice
                cross entropy and boundary loss functions. This should be in the
                interval [0, 1].
        """
        # Compute region loss.
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs. Apply one-hot encoding to the ground truth mask.
        # Apply the softmax function to the predicted mask to turn this into a
        # probability mask.
        y_true = loss_utils.get_one_hot(y_true, y_pred.shape[1])
        y_pred = F.softmax(y_pred, dim=1)

        # Exclude the background class from the computation if necessary.
        if self.exclude_background:
            y_true = y_true[:, 1:, :, :, :]
            y_pred = y_pred[:, 1:, :, :, :]
            dtm = dtm[:, 1:, :, :, :]

        # Compute generalize surface loss.
        numerator = torch.sum(
            torch.square(dtm * (1 - (y_true + y_pred))), dim=self.axes
        )
        denominator = torch.sum(torch.square(dtm), dim=self.axes) + self.smooth

        # Average the generalized surface loss across all classes.
        boundary_loss = torch.mean(1 - (numerator / denominator), dim=1)

        # Average the generalized surface loss across the batch.
        boundary_loss = torch.mean(boundary_loss)

        # Return the weighted sum of the region and boundary loss.
        return alpha * region_loss + (1. - alpha) * boundary_loss


def get_loss(args: argparse.Namespace) -> nn.Module:
    """Get the loss function based on the command line arguments.

    Args:
        args: The command line arguments.
        kwargs: Additional keyword arguments to pass to the loss function.

    Returns:
        A loss function module.

    Raises:
        ValueError if the loss function is not recognized.
    """
    if args.loss == "dice":
        return DiceLoss(exclude_background=args.exclude_background)
    if args.loss == "dice-ce":
        return DiceCELoss(exclude_background=args.exclude_background)
    if args.loss == "bl":
        return BoundaryLoss(exclude_background=args.exclude_background)
    if args.loss == "hdl":
        return HDOneSidedLoss(exclude_background=args.exclude_background)
    if args.loss == "gsl":
        return GenSurfLoss(exclude_background=args.exclude_background)
    if args.loss == "cldice":
        return SoftDiceCLDice(exclude_background=args.exclude_background)

    # Raise an error if the loss function is not recognized.
    raise ValueError(f"Invalid loss function: {args.loss}")
