from enum import Enum

from .swin_unet_2d import SwinUnet2D
from .unet import UNet
from .unet_lcl import UNetLCL


class ModelType(Enum):
    swin_unet_2d = SwinUnet2D
    unet = UNet
    unet_lcl = UNetLCL
