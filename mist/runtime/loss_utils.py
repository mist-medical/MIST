"""Loss function utilities."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_one_hot(y_true, n_classes):
    y_true = y_true.to(torch.int64)
    y_true = F.one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true


class SoftSkeletonize(nn.Module):

    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        # Ensure 5D input for 3D img
        if len(img.shape) != 5:
            raise ValueError("[Loss Error] len(img.shape) is not equal to 5")
            
        # Apply maxpooling
        p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
        p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
        p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))

        # Return the min of the pools
        return torch.min(torch.min(p1, p2), p3)
            
    def soft_dilate(self, img):
        # Ensure 5D input for 3D img
        if len(img.shape) != 5:
            raise ValueError("[Loss Error] len(img.shape) is not equal to 5")
            
        # Apply max pooling with 3x3x3 kernel size
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

    def soft_open(self, img):
        # Soft erode followed by soft dilate
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        # Initial soft opening
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        # Soft skeletonizing loop
        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        # Perform skeletonization
        return self.soft_skel(img)
