import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonize(nn.Module):

    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        # Ensure 5D input for 3D img
        if len(img.shape) == 5:
            # Apply maxpooling
            p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
            p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
            p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))
            
            # Return the min of the pools
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise ValueError("[Loss Error] len(img.shape) is not equal to 5")

    def soft_dilate(self, img):
        # Ensure 5D input for 3D img
        if len(img.shape) == 5:
            # Apply max pooling with 3x3x3 kernel size
            return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))
        else:
            raise ValueError("[Loss Error] len(img.shape) is not equal to 5")

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
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = F.softmax(y_pred, dim=1)
        
        # Perform skeletonization
        return self.soft_skel(img)

def get_one_hot(targets, num_classes):
    # Convert target indices to one-hot encoding
    return torch.eye(num_classes)[targets]

def softmax(x, dim):
    # Apply softmax along the specified dimension
    return F.softmax(x, dim=dim)
