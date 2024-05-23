import abc
from pathlib import Path

class BasicReader(metaclass = abc.ABCMeta):

    @abc.abstractmethod
    def read_variable(self):
        return NotImplemented
    
    def check_file_exist(self, file_fullname: str) -> None:
        assert Path(file_fullname).exists(), f'{file_fullname} does not exist!'