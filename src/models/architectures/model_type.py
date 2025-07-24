from enum import Enum

from .swin_unet_2d import SwinUnet2D
from .swin_unet import SwinUnet
from .cnn_unet import CnnUnet
from .discriminator import Discriminator
from .cnn_unet_class import CnnUnetClass


class ModelType(Enum):
    swin_unet_2d = SwinUnet2D
    swin_unet = SwinUnet
    cnn_unet = CnnUnet
    discriminator = Discriminator
    cnn_unet_class = CnnUnetClass
