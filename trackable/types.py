"""Internal module for typing extensions"""


from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Union

from numpy import ndarray
from pandas import DataFrame

Data = Union[DataFrame, ndarray]
Metric = Callable[[Data, Data], float]


class _ModelBaseClass(ABC):
    @abstractmethod
    def predict(self, data: Data) -> Data:
        """Predict method of generic model"""


GenericModel = TypeVar("GenericModel", bound=_ModelBaseClass)
