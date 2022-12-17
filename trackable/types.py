"""Internal module for typing extensions"""


from typing import Callable, Protocol, Union

from numpy import ndarray
from pandas import DataFrame

Data = Union[DataFrame, ndarray]
Metric = Callable[[Data, Data], Union[float, int]]


class GenericModel(Protocol):
    """Generic Sklearn Type model."""

    def predict(self, data: Data) -> Data:
        """Generic predict method"""
        ...
