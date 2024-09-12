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


def voi_weighted_loss(y_true, y_pred):
    # grab the ground truth and predicted dose
    gt_dose, p_dose = y_true[..., 0], y_pred[..., 0] 

    # grab the voxel weights. Will only be called if use_voi_weights is True
    weights = y_true[..., -1]  # note: element 0 is the GT dose, elt 1 is weights
    
    y_pred_weighted = weights * p_dose
    y_true_weighted = weights * gt_dose

    return torch.mean((y_true_weighted - y_pred_weighted) ** 2)
    


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
