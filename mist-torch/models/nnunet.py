import torch.nn as nn
from monai.networks.nets import DynUNet


class NNUnet(nn.Module):
    def __init__(self,
                 n_classes,
                 n_channels,
                 pocket,
                 deep_supervision,
                 deep_supervision_heads,
                 patch_size,
                 target_spacing):
        super(NNUnet, self).__init__()
        kernels, strides = self.get_unet_params(patch_size, target_spacing)

        # Pocket nnUNet
        if pocket:
            filters = [min(2 ** (5 + 0), 320) for i in range(len(strides))]
        else:
            filters = [min(2 ** (5 + i), 320) for i in range(len(strides))]

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.deep_supervision = deep_supervision
        self.deep_supervision_heads = deep_supervision_heads
        self.use_res_block = True
        self.unet = DynUNet(
            3,
            self.n_channels,
            self.n_classes,
            kernels,
            strides,
            strides[1:],
            filters=filters,
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": False, "negative_slope": 0.01}),
            deep_supervision=self.deep_supervision,
            deep_supr_num=self.deep_supervision_heads,
            res_block=self.use_res_block,
            trans_bias=True,
        )

    def forward(self, x):
        if self.training and self.deep_supervision:
            return self.split_deep_supervision(self.unet(x))
        else:
            return self.unet(x)

    @staticmethod
    def split_deep_supervision(x):
        # Make sure output is correct dimension
        assert len(x.shape) == 6

        # Split from array to list of outputs for compatability with MIST
        output = [None] * x.shape[1]
        for i in range(x.shape[1]):
            output[i] = x[:, i, ...]

        return tuple(output)

    @staticmethod
    def get_unet_params(patch_size, spacings):
        strides, kernels, sizes = [], [], patch_size[:]
        while True:
            spacing_ratio = [spacing / min(spacings) for spacing in spacings]
            stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
            kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
            if all(s == 1 for s in stride):
                break
            sizes = [i / j for i, j in zip(sizes, stride)]
            spacings = [i * j for i, j in zip(spacings, stride)]
            kernels.append(kernel)
            strides.append(stride)
            if len(strides) == 5:
                break
        strides.insert(0, len(spacings) * [1])
        kernels.append(len(spacings) * [3])
        return kernels, strides