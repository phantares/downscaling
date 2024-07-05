import h5py
from .basic_reader import BasicReader


class Hdf5Reader(BasicReader):

    def __init__(self, file_fullname: str) -> None:
        self.check_file_exist(file_fullname)
        self.file = h5py.File(file_fullname, "r")

    def read_variable(self, variable_name: str):
        return self.file[variable_name]
