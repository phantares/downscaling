import abc
from pathlib import Path

class BasicWriter(metaclass = abc.ABCMeta):
    
    @abc.abstractmethod
    def write_file(self):
        return NotImplemented

    def make_path(self, file_path: str) -> None:
        Path(file_path).mkdir(parents = True, exist_ok = True)