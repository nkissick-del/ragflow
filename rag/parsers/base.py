from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    """
    Base class for all parsers.
    """

    @abstractmethod
    def __call__(self, filename: str, binary: Any = None) -> Any:
        """
        Parses the input file.

        Args:
            filename (str): The path to the file.
            binary (Any): Optional binary content or flag.

        Returns:
            Any: The parsed result, typically a list of documents.
        """
        pass
