"""Module used to generate a minimal report to track ML models"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler

from trackable import types
from trackable.exceptions import ModelAlreadyExistsError

__all__ = ["Report"]


class Report:
    def __init__(self, X_test: Any, y_test: Any, metrics: List[types.Metric]) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = metrics
        self._models: dict = {}
        self._results: List[dict] = []

    def add_model(
        self,
        model: types.GenericModel,
        name: Optional[str] = None,
        X_test: Optional[Any] = None,
        y_test: Optional[Any] = None,
    ) -> None:
        X = X_test if X_test else self.X_test
        y = y_test if y_test else self.y_test
        name = name if name else model.__class__.__name__

        y_pred = model.predict(X)

        results: Dict[str, Union[str, float]] = {metric.__name__: metric(y, y_pred) for metric in self.metrics}
        results["name"] = name
        if name in self._models:
            raise ModelAlreadyExistsError(
                """Model already exists. Are you sure you want to add this model.
                If so, try adding with a different name."""
            )
        self._results.append(results)
        self._models[name] = model

    def generate(self, highlight: Optional[bool] = True) -> Union[Styler, pd.DataFrame]:
        if not self._results:
            return pd.DataFrame([])
        results = pd.DataFrame(self._results)
        results = results.set_index("name")
        if highlight:
            return results.style.highlight_max()
        return results
