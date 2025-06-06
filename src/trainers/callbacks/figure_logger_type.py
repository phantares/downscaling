from enum import Enum

from .figure_logger import FigureLogger
from .class_figure_logger import ClassFigureLogger


class FigureLoggerType(Enum):
    simple = FigureLogger
    classf = ClassFigureLogger
