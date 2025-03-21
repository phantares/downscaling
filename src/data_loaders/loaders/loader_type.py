from enum import Enum

from .precipitation_dataset import PrecipitationDataset
from .pickle_dataset import FourcastnetPickleDataset, PanguPickleDataset


class LoaderType(Enum):
    single = PrecipitationDataset
    pickle_fcn = FourcastnetPickleDataset
    pickle_pgw = PanguPickleDataset
