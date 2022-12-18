"""Module used to generate a minimal report to track ML models"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler

from trackable import types
from trackable.exceptions import ModelAlreadyExistsError

__all__ = ["Report"]


class Report:
    """
    A minimalistic model reporting class.

    A simple usage pipeline may include:

    1. Instantiate the class `report = Report(X_test, y_test, metrics=[...])`

    2. Add models `report.add_model(...)`

    3. Generate the report `report.generate()`

    Args:
        X_test (ndarray or DataFrame):
            Testing data which is used to evaluate with the provided metrics.
            Can be substituted for another dataframe in `.add_model`.
        y_test (ndarray or DataFrame):
            Ground truth testing data for evaluation.
            Can be substituted for another dataframe in `.add_model`.
        metrics (List of sklearn style metrics):
            A list of metrics in the style of a sklearn metric.
            In other words, the first argument should be y_true and the second y_pred.
            It is not advised to change metrics once the class is instantied. Instead,
            a new Report object should be created for a set of new metrics.
    """

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
