from enum import Enum

from .simple_framework import SimpleFramework
from .gan_framework import GanFramework


class FrameworkType(Enum):
    simple = SimpleFramework
    gan = GanFramework
