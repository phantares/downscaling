from enum import Enum

from .simple_framework import SimpleFramework
from .gan_framework import GanFramework
from .class_framework import ClassFramework


class FrameworkType(Enum):
    simple = SimpleFramework
    gan = GanFramework
    classf = ClassFramework
