import os
import json
import torch
import torch.nn as nn
from collections import OrderedDict

from monai.networks.blocks.dynunet_block import UnetOutBlock
from mist.models.layers import ConvLayer

from mist.models.unet import UNet
from mist.models.mgnets import MGNet
from mist.models.nnunet import NNUnet
from mist.models.attn_unet import MONAIAttnUNet
from mist.models.swin_unetr import MONAISwinUNETR

"""
Available models:
    - nnUNet
    - U-Net
    - FMG-Net
    - W-Net
    - Attention UNet
    - Swin UNETR
"""


def get_model(**kwargs):
    if kwargs["model_name"] == "nnunet":
        model = NNUnet(
            kwargs["n_channels"],
            kwargs["n_classes"],
            kwargs["pocket"],
            kwargs["deep_supervision"],
            kwargs["deep_supervision_heads"],
            kwargs["vae_reg"],
            kwargs["patch_size"],
            kwargs["target_spacing"],
            kwargs["use_res_block"]
        )
    elif kwargs["model_name"] == "unet":
        model = UNet(
            kwargs["n_channels"],
            kwargs["n_classes"],
            kwargs["patch_size"],
            kwargs["use_res_block"],
            kwargs["pocket"],
            kwargs["deep_supervision"],
            kwargs["deep_supervision_heads"],
            kwargs["vae_reg"]
        )
    elif kwargs["model_name"] == "fmgnet":
        model = MGNet(
            "fmgnet",
            kwargs["n_channels"],
            kwargs["n_classes"],
            kwargs["patch_size"],
            kwargs["use_res_block"],
            kwargs["deep_supervision"],
            kwargs["deep_supervision_heads"],
            kwargs["vae_reg"]
        )
    elif kwargs["model_name"] == "wnet":
        model = MGNet(
            "wnet",
            kwargs["n_channels"],
            kwargs["n_classes"],
            kwargs["patch_size"],
            kwargs["use_res_block"],
            kwargs["deep_supervision"],
            kwargs["deep_supervision_heads"],
            kwargs["vae_reg"]
        )
    elif kwargs["model_name"] == "attn_unet":
        model = MONAIAttnUNet(
            kwargs["n_classes"],
            kwargs["n_channels"],
            kwargs["pocket"],
            kwargs["patch_size"]
        )
    elif kwargs["model_name"] == "unetr":
        model = MONAISwinUNETR(
            kwargs["n_classes"],
            kwargs["n_channels"],
            kwargs["patch_size"]
        )
    else:
        raise ValueError("Invalid model name")

    return model


def load_model_from_config(weights_path, model_config_path):
    # Get model configuration
    with open(model_config_path, "r") as file:
        model_config = json.load(file)

    # Load model
    model = get_model(**model_config)

    # Trick for loading DDP model
    state_dict = torch.load(weights_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove 'module.' of DataParallel/DistributedDataParallel
        name = k[7:]

        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def configure_pretrained_model(pretrained_model_path, n_channels, n_classes):
    n_files = len(os.listdir(pretrained_model_path)) - 1
    model_list = [os.path.join(pretrained_model_path, "fold_{}.pt".format(i)) for i in range(n_files)]
    model_config_path = os.path.join(pretrained_model_path, "model_config.json")
    models = [load_model_from_config(model, model_config_path) for model in model_list]
    state_dicts = [model.state_dict() for model in models]

    # Get model configuration
    with open(model_config_path, "r") as file:
        model_config = json.load(file)

    model_name = model_config["model_name"]

    # Average weights across all models
    avg_state_dict = state_dicts[0]
    for state_dict in state_dicts[1:]:
        for key in avg_state_dict.keys():
            avg_state_dict[key] += state_dict[key]

    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(state_dicts)

    # Load averaged weights into new model
    model = models[0]
    model.load_state_dict(avg_state_dict)

    # Modify model to match new input and output channels
    if model_name == "unet":
        if model.first_conv.in_channels != n_channels:
            model.first_conv = ConvLayer(n_channels, model.init_filters, **model.conv_kwargs)

            if model.vae_reg:
                model.vae_out = nn.Conv3d(in_channels=model.channels[-1][0],
                                          out_channels=n_channels,
                                          kernel_size=1)

        if model.out.out_channels != n_classes:
            model.out = nn.Conv3d(in_channels=model.init_filters,
                                  out_channels=n_classes,
                                  kernel_size=1)

            if model.deep_supervision:
                model.heads = nn.ModuleList()
                for channel_pair in model.channels[-(model.deep_supervision_heads + 1):-1]:
                    head = nn.Conv3d(in_channels=channel_pair[0],
                                     out_channels=n_classes,
                                     kernel_size=1)
                    model.heads.append(head)
    elif model_name == "fmgnet" or model_name == "wnet":
        if model.first_conv.in_channels != n_channels:
            model.first_conv = ConvLayer(n_channels, model.out_channels, **model.conv_kwargs)

            if model.vae_reg:
                model.vae_out = nn.Conv3d(in_channels=model.out_channels,
                                          out_channels=n_channels,
                                          kernel_size=1)
        if model.out.out_channels != n_channels:
            model.out = nn.Conv3d(in_channels=model.out_channels,
                                  out_channels=n_classes,
                                  kernel_size=1)

            if model.deep_supervision:
                model.heads = nn.ModuleList()
                for _ in range(model.deep_supervision_heads):
                    head = nn.Conv3d(in_channels=model.out_channels,
                                     out_channels=n_classes,
                                     kernel_size=1)
                    model.heads.append(head)
    elif model_name == "nnunet":
        if model.unet.input_block.conv1.conv.in_channels != n_channels:
            model.unet.input_block.conv1.conv = nn.Conv3d(in_channels=n_channels,
                                                          out_channels=model.unet.input_block.conv1.conv.out_channels,
                                                          kernel_size=model.unet.input_block.conv1.conv.kernel_size,
                                                          stride=model.unet.input_block.conv1.conv.stride,
                                                          padding=model.unet.input_block.conv1.conv.padding,
                                                          bias=False)

            if model.vae_reg:
                model.unet.vae_out = nn.Conv3d(in_channels=model.unet.vae_out.in_channels,
                                               out_channels=n_channels,
                                               kernel_size=model.unet.vae_out.kernel_size,
                                               stride=model.unet.vae_out.stride)

        if model.unet.output_block.conv.conv.out_channels != n_classes:
            model.unet.output_block.conv.conv = nn.Conv3d(in_channels=model.unet.output_block.conv.conv.in_channels,
                                                          out_channels=n_classes,
                                                          kernel_size=model.unet.output_block.conv.conv.kernel_size,
                                                          stride=model.unet.output_block.conv.conv.stride)

            if model.deep_supervision:
                model.unet.deep_supervision_heads = nn.ModuleList([UnetOutBlock(model.unet.spatial_dims,
                                                                                model.unet.filters[i + 1], 2,
                                                                                dropout=model.unet.dropout) for i in
                                                                   range(model.unet.deep_supr_num)])
    elif model_name == "unetr":
        if model.model.swinViT.patch_embed.proj.in_channels != n_channels:
            model.model.swinViT.patch_embed.proj = nn.Conv3d(n_channels, 24, kernel_size=(2, 2, 2), stride=(2, 2, 2))

        if model.model.out.conv.conv.out_channels != n_classes:
            model.model.out.conv.conv = nn.Conv3d(24, n_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    elif model_name == "attn_unet":
        if model.model.model[0].conv[0].conv.in_channels != n_channels:
            model.model.model[0].conv[0].conv = nn.Conv3d(n_channels, 32, kernel_size=3)

        if model.model.model[-1].conv.out_channels != n_classes:
            model.model.model[-1].conv = nn.Conv3d(32, n_classes, kernel_size=1)
    else:
        raise ValueError("Invalid model name for pretrained model. Check your model config file.")

    return model
