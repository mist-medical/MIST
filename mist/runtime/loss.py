import torch
import torch.nn as nn
from torch.nn.functional import softmax, one_hot, relu
from mist.runtime.loss_utils import get_one_hot, voi_weighted_loss, SoftSkeletonize


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        den = torch.sum(torch.square(y_true), dim=self.axes) + torch.sum(torch.square(y_pred),
                                                                         dim=self.axes) + self.smooth

        loss = torch.mean(num / den, axis=1)
        loss = torch.mean(loss)
        return loss
    

class DiceCELoss(nn.Module):
    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_dice = self.dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return 0.5*(loss_ce + loss_dice)


class SoftCLDice(nn.Module):
    def __init__(self, iterations=3, smooth=1., exclude_background=False):
        super(SoftCLDice, self).__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float)
        y_pred = softmax(y_pred, dim=1)

        if self.exclude_background:
            y_true = y_true[:, 1:, :, :]
            y_pred = y_pred[:, 1:, :, :]
            
        skel_pred = self.soft_skeletonize(y_pred)
        skel_true = self.soft_skeletonize(y_true)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice


class SoftDiceCLDice(nn.Module):
    def __init__(self, iterations=3, smooth=1., exclude_background=False):
        super(SoftDiceCLDice, self).__init__()
        self.iterations = iterations
        self.smooth = smooth
        self.region_loss = DiceCELoss()
        self.exclude_background = exclude_background
        self.cldice_loss = SoftCLDice(self.iterations, self.smooth, self.exclude_background)

    def forward(self, y_true, y_pred, alpha):
        region_loss = self.region_loss(y_true, y_pred)
        cldice = self.cldice_loss(y_true, y_pred)
        return alpha * region_loss + (1. - alpha) * cldice


class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedDiceLoss, self).__init__()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not(class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):
        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1. / (torch.square(class_weights) + 1.)
        else:
            class_weights = self.class_weights

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(y_true), dim=self.axes) + torch.sum(torch.square(y_pred),
                                                                         dim=self.axes)
        den *= class_weights
        den += self.smooth

        loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        loss = torch.mean(loss)

        return loss


class WeightedDiceCELoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedDiceCELoss, self).__init__()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not(class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.weighted_dice_loss = WeightedDiceLoss(class_weights=self.class_weights)

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_weighted_dice = self.weighted_dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float32)

        # Weighted cross entropy loss
        loss_weighted_ce = self.cross_entropy(y_pred, y_true)

        return 0.5*(loss_weighted_dice + loss_weighted_ce)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.region_loss = DiceCELoss()
        self.alpha = 0.5

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_pred = softmax(y_pred, dim=1)

        # Compute boundary loss
        boundary_loss = torch.mean(dtm * y_pred)

        return alpha * region_loss + (1. - alpha) * boundary_loss


class HDOneSidedLoss(nn.Module):
    def __init__(self):
        super(HDOneSidedLoss, self).__init__()
        self.region_loss = DiceCELoss()
        self.alpha = 0.5

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        # Compute boundary loss
        boundary_loss = torch.mean(torch.square(y_true - y_pred) * torch.square(dtm))

        return alpha * region_loss + (1. - alpha) * boundary_loss


class GenSurfLoss(nn.Module):
    def __init__(self, class_weights):
        super(GenSurfLoss, self).__init__()
        self.region_loss = DiceCELoss()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if not (class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.alpha = 0.5
        self.smooth = 1e-6
        self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1. / (torch.square(class_weights) + 1.)
        else:
            class_weights = self.class_weights

        # Compute loss
        num = torch.sum(torch.square(dtm * (1 - (y_true + y_pred))), axis=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(dtm), axis=self.axes)
        den *= class_weights
        den += self.smooth

        boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1. - boundary_loss

        return alpha * region_loss + (1. - alpha) * boundary_loss


class KLDivLoss(nn.Module):
    def forward(self, z_mean, z_log_var):
        loss = -0.5 * (1. + z_log_var - torch.square(z_mean) - torch.exp(z_log_var))
        return torch.mean(loss)


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.kl_loss = KLDivLoss()

    def forward(self, y_true, y_pred):
        return self.reconstruction_loss(y_true, y_pred[0]) + self.kl_loss(y_pred[1], y_pred[2])


class MAELoss(nn.Module):
    def __init__(self): 
        super(MAELoss, self).__init__()
        self.reduction = 'mean' # default
        
    def forward(self, y_true, y_pred):  # 0 gt_dose, 1 weights
        # Prepare inputs. Regression, so output last layer activation with ReLU. 
        y_pred = relu(y_pred)

        # elt-wise MSE loss
        loss = torch.nn.L1Loss(reduction=self.reduction)  # default of reduction='mean'. Use none to get individual losses
        loss = loss(y_true[...,0], y_pred[...,0])
                 
        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction


class MSELoss(nn.Module):
    def __init__(self): 
        super(MSELoss, self).__init__()
        self.reduction = 'mean' # default
        
    def forward(self, y_true, y_pred):  # 0 gt_dose, 1 weights
        # Prepare inputs. Regression, so output last layer activation with ReLU. y_pred comes from the layer before the last that still has many neurons.
        y_pred = relu(y_pred)

        # elt-wise MSE loss
        loss = torch.nn.MSELoss(reduction=self.reduction)  # default of reduction='mean'. Use none to get individual losses
        loss = loss(y_true[...,0], y_pred[...,0])
                 
        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # No reduction


# This one has an issue as it gives tensors of different size, so fix it later!!!
class WeightedMSELoss(nn.Module):
    def __init__(self): 
        super(WeightedMSELoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')  # default of reduction='mean'. Use none to get individual losses
        self.reduction = 'mean' # For w_mse, we will just assume the mean() version
            

    def forward(self, y_true, y_pred):  
        # Prepare inputs. Regression, so output last layer activation with ReLU. y_pred comes from the layer before the last that still has many neurons.
        y_pred = relu(y_pred)

        weighted_mse = voi_weighted_loss(y_true, y_pred)

        # elt-wise MSE loss
        loss = self.mse_loss(y_true[...,0], y_pred[...,0])
        mse = loss.mean()  # Apply reduction (mean)
                 
        return mse + weighted_mse
 

# Fix this too
class DVHLoss(nn.Module):   # Later test weighted/modified dvh. Also class to combine losses???
    def __init__(self, min_dose=0, max_dose=85, b=1, m=1.0, Ns=23, norm_ord=2, name='dvh_loss'):
        super(DVHLoss, self).__init__()
        
        self.d = torch.arange(min_dose, max_dose, b, dtype=torch.float32)
        self.Nd = float(self.d.numel())  # Number of dose bins
        self.b = float(b)
        self.m = float(m)
        self.mb_factor = self.m / self.b  # m/b from UTSW equation
        self.ord = norm_ord
        self.NsNd = self.Nd * (Ns - 1)
        self.Ns = Ns

    def forward(self, y_true, y_pred):
        # Prepare inputs. Regression, so output last layer activation with ReLU. 
        y_pred = relu(y_pred)

        loss = 0.0  # Initialize loss
        print(f"Ns {self.Ns}, Nd {self.Nd}, NsNd {self.NsNd}")

        for s in range(1, self.Ns):  # Iterate over the structures

            # Get the indices of the non-zero voxels in the structure s
            non_zero_indices = (y_true[..., s] != 0).nonzero(as_tuple=False)
            print(f"non_zero_indices {non_zero_indices}, non_zero_indices_numel {non_zero_indices.numel()}")
            
            if non_zero_indices.numel() > 0:
                # Get the dose values of the non-zero voxels in the structure s
                y_true_roi = y_true[..., 0].index_select(0, non_zero_indices[:, 0]).unsqueeze(-1)
                y_pred_roi = y_pred[..., 0].index_select(0, non_zero_indices[:, 0]).unsqueeze(-1)

                # Calculate total volume
                tot_vol = float(y_true_roi.numel()) + torch.finfo(torch.float32).eps
                
                # Calculating dvh_true and dvh_pred in a vectorized manner
                dvh_true = torch.sum(torch.sigmoid(self.mb_factor * (y_true_roi - self.d)), dim=0)
                dvh_pred = torch.sum(torch.sigmoid(self.mb_factor * (y_pred_roi - self.d)), dim=0)
                
                # Calculating the loss for the structure s
                loss += torch.norm((dvh_true - dvh_pred) / tot_vol, p=self.ord)
            else:
                # Handle case with no non-zero indices
                loss += 0.0

        return loss / self.NsNd


def get_loss(args, **kwargs):  
    if args.loss == "dice":
        return DiceLoss()
    elif args.loss == "dice_ce":
        return DiceCELoss()
    elif args.loss == "gdl":
        return WeightedDiceLoss(class_weights=kwargs["class_weights"])
    elif args.loss == "gdl_ce":
        return WeightedDiceCELoss(class_weights=kwargs["class_weights"])
    elif args.loss == "bl":
        return BoundaryLoss()
    elif args.loss == "hdl":
        return HDOneSidedLoss()
    elif args.loss == "gsl":
        return GenSurfLoss(class_weights=kwargs["class_weights"])
    elif args.loss == "cldice":
        return SoftDiceCLDice()
    elif args.loss == "mae":
        return MAELoss()  
    elif args.loss == "mse":
        return MSELoss()  # default for dose prediction
    elif args.loss == "w_mse":
        return WeightedMSELoss()  
    elif args.loss == "dvh":
        return DVHLoss()  
    else:
        raise ValueError("Invalid loss function")
