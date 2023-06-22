from models.unet import UNet
from models.nnunet import NNUnet
from models.resnet import ResNet
from models.densenet import DenseNet


def get_model(**kwargs):
    if kwargs["model_name"] == "unet":
        model = UNet(kwargs["n_classes"],
                     kwargs["n_channels"],
                     kwargs["init_filters"],
                     kwargs["depth"],
                     kwargs["pocket"],
                     kwargs["deep_supervision"],
                     kwargs["deep_supervision_heads"])
    elif kwargs["model_name"] == "nnunet":
        model = NNUnet(kwargs["n_classes"],
                       kwargs["n_channels"],
                       kwargs["pocket"],
                       kwargs["deep_supervision"],
                       kwargs["deep_supervision_heads"],
                       kwargs["patch_size"],
                       kwargs["target_spacing"])
    elif kwargs["model_name"] == "resnet":
        model = ResNet(kwargs["n_classes"],
                       kwargs["n_channels"],
                       kwargs["init_filters"],
                       kwargs["depth"],
                       kwargs["pocket"],
                       kwargs["deep_supervision"],
                       kwargs["deep_supervision_heads"])
    elif kwargs["model_name"] == "densenet":
        model = DenseNet(kwargs["n_classes"],
                         kwargs["n_channels"],
                         kwargs["init_filters"],
                         kwargs["depth"],
                         kwargs["pocket"],
                         kwargs["deep_supervision"],
                         kwargs["deep_supervision_heads"])
    else:
        raise ValueError("Invalid model name!")

    return model
