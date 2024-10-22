"""Loss function utilities."""
import torch
from torch import nn
from torch.nn import functional as F


def get_one_hot(y_true: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Converts a 3D label tensor to one-hot encoding.

    Args:
        y_true: 3D tensor of shape (batch_size, channels, height, width, depth).
        n_classes: Number of classes.

    Returns:
        One-hot encoding of the input segmentation mask.
    """
    y_true = y_true.to(torch.int64)
    y_true = F.one_hot(y_true, num_classes=n_classes) # pylint: disable=not-callable
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true


class SoftSkeletonize(nn.Module):
    """Soft skeletonization of a binary mask.

    Attributes:
        num_iter: Number of iterations for the soft skeletonization.
    """
    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        """Soft erosion operation.

        Args:
            img: 5D tensor of shape (batch_size, channels, depth, height,
                width). This is a probability map.

        Returns:
            An eroded version of the input tensor.
        """
        # Ensure 5D input for 3D image.
        if len(img.shape) != 5:
            raise ValueError(
                "In SoftSkeletonize, len(img.shape) is not equal to 5. "
                f"Got {len(img.shape)}."
            )

        # Apply max pooling.
        p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
        p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
        p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))

        # Return the min of the pools.
        return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        """Soft dilation operation.

        Args:
            img: 5D tensor of shape (batch_size, channels, depth, height,
                width). This is a probability map.

        Returns:
            A dilated version of the input tensor.
        """
        # Ensure 5D input for 3D image.
        if len(img.shape) != 5:
            raise ValueError(
                "In SoftSkeletonize, len(img.shape) is not equal to 5. "
                f"Got {len(img.shape)}."
            )

        # Apply max pooling with 3x3x3 kernel size
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        """Soft opening operation.

        Morphological opening is an erosion followed by a dilation.

        Args:
            img: 5D tensor of shape (batch_size, channels, depth, height,
                width). This is a probability map.

        Returns:
            A morphologically opened version of the input tensor.
        """
        # Soft erode followed by soft dilate
        return self.soft_dilate(self.soft_erode(img))

    def soft_skeletonize(self, img: torch.Tensor) -> torch.Tensor:
        """Perform skeletonization.

        Args:
            img: 5D tensor of shape (batch_size, channels, depth, height,
                width). This is a probability map.

        Returns:
            The soft skeleton of the input tensor.
        """
        # Initial soft opening
        soft_opened_image = self.soft_open(img)
        skeleton = F.relu(img - soft_opened_image)

        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            soft_opened_image = self.soft_open(img)
            delta = F.relu(img - soft_opened_image)

            # Don't ues += here to avoid in-place operation error.
            skeleton = skeleton + F.relu(delta - skeleton * delta)

        return skeleton

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            img: 5D tensor of shape (batch_size, channels, depth, height,
                width). This is a probability map.

        Returns:
            The soft skeleton of the input tensor.
        """
        return self.soft_skeletonize(img)

