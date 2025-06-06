from enum import Enum

from .swin_unet_2d import SwinUnet2D
from .swin_unet import SwinUnet
from .unet import UNet
from .unet_lcl import UNetLCL
from .discriminator import Discriminator
from .unet_class import UNetClass
from .swin_unet_class import SwinUnetClass


class ModelType(Enum):
    swin_unet_2d = SwinUnet2D
    swin_unet = SwinUnet
    unet = UNet
    unet_lcl = UNetLCL
    discriminator = Discriminator
    unet_class = UNetClass
    swin_class = SwinUnetClass
