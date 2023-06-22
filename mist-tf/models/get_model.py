from models.unet import UNet
from models.hrnet import HRNet
from models.nnunet import NNUnet
from models.resnet import ResNet
from models.densenet import DenseNet


def get_model(model_name, **kwargs):
    if model_name == "unet":
        model = UNet(n_classes=kwargs["n_classes"],
                     init_filters=kwargs["init_filters"],
                     depth=kwargs["depth"],
                     pocket=kwargs["pocket"],
                     deep_supervision=kwargs["deep_supervision"])
    elif model_name == "nnunet":
        model = NNUnet(kwargs["config"],
                       kwargs["n_channels"],
                       kwargs["n_classes"],
                       kwargs["pocket"],
                       kwargs["deep_supervision"])
    elif model_name == "resnet":
        model = ResNet(n_classes=kwargs["n_classes"],
                       init_filters=kwargs["init_filters"],
                       depth=kwargs["depth"],
                       pocket=kwargs["pocket"],
                       deep_supervision=kwargs["deep_supervision"])
    elif model_name == "densenet":
        model = DenseNet(n_classes=kwargs["n_classes"],
                         init_filters=kwargs["init_filters"],
                         depth=kwargs["depth"],
                         pocket=kwargs["pocket"],
                         deep_supervision=kwargs["deep_supervision"])
    elif model_name == "hrnet":
        if kwargs["deep_supervision"]:
            raise Warning("Deep supervision not supported for HRNet!")

        model = HRNet(input_shape=kwargs["input_shape"],
                      num_channels=kwargs["n_channels"],
                      num_class=kwargs["n_classes"],
                      init_filters=kwargs["init_filters"],
                      pocket=kwargs["pocket"]).build_model()
    else:
        raise ValueError("Invalid model name!")

    return model
