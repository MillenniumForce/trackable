from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Union

from numpy import array
from pandas import DataFrame

Data = Union[DataFrame, array]
Metric = Callable[[Data, Data], float]


class _ModelBaseClass(ABC):
    @abstractmethod
    def predict(self, data: Data):
        """Predict method of generic model"""


GenericModel = TypeVar("GenericModel", bound=_ModelBaseClass)
