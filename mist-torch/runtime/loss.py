import torch
import torch.nn as nn
from torch.nn.functional import softmax, one_hot


def get_one_hot(y_true, n_classes):
    y_true = y_true.to(torch.int64)
    y_true = one_hot(y_true, num_classes=n_classes)
    y_true = torch.transpose(y_true, dim0=5, dim1=1)
    y_true = torch.squeeze(y_true, dim=5)
    y_true = y_true.to(torch.int8)
    return y_true


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
        y_true = get_one_hot(y_true, y_pred.shape[1]).to(torch.float32)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return loss_ce + loss_dice


class WeightedDiceLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedDiceLoss, self).__init__()
        if not(class_weights is None):
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        self.axes = (2, 3, 4)
        self.smooth = 1.e-6

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
                                                                         dim=self.axes) + self.smooth
        den *= class_weights

        loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        loss = torch.mean(loss)

        return loss


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


def get_loss(args, **kwargs):
    if args.loss == "dice":
        return DiceLoss()
    elif args.loss == "dice_ce":
        return DiceCELoss()
    elif args.loss == "gdl":
        return WeightedDiceLoss(class_weights=kwargs["class_weights"])
    else:
        raise ValueError("Invalid loss function")
