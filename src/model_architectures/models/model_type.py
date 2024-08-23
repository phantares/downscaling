from enum import Enum

from .swin_unet_2d import SwinUnet2D
from .swin_unet import SwinUnet
from .swin_unet_res import SwinUnetRes
from .unet import UNet
from .unet_res import UNetRes
from .unet_lcl import UNetLCL
from .discriminator import Discriminator


class ModelType(Enum):
    swin_unet_2d = SwinUnet2D
    swin_unet = SwinUnet
    swin_unet_res = SwinUnetRes
    unet = UNet
    unet_res = UNetRes
    unet_lcl = UNetLCL
    discriminator = Discriminator
