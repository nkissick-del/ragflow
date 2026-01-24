from abc import ABC, abstractmethod


class BaseParser(ABC):
    @abstractmethod
    def __call__(self, filename, binary=None):
        pass
