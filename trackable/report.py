"""Module used to generate a minimal report to track ML models."""

from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from pandas.io.formats.style import Styler

from trackable import types
from trackable.exceptions import ModelAlreadyExistsError

__all__ = ["Report"]


class Report:
    """A minimalistic model reporting class.

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
        """Instantiate a Report."""
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
        """Add a model to the report.

        The model is passed through each metric and then cached in memory.
        In future versions, caching strategies may change to account for larger models.
        Also note that each model must be unique otherwise an error is raised.

        Args:
            model (GenericModel):
                A generic sklearn type model which must have a `.predict` method.
                For more complex models, it's easy to create a meta-class which
                can abstract much of the models complexities
                (e.g. a forward pass of a neural network) into a `.predict` method.
            name (str, optional):
                Name of the model. Defaults to `model.__class__.__name__`.
            X_test (Any, optional): Overwrite the testing data.
                Defaults to X_test that was used to instantiate the class.
            y_test (Any, optional): Overwrite the testing data.
                Defaults to y_test that was used to instantiate the class.

        Raises:
            ModelAlreadyExistsError: Raised if model name already exists.
                This is essentially to avoid caching conflicts.
        """
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

    def generate(self, highlight: Literal["max", "min", False] = "max") -> Union[Styler, pd.DataFrame]:
        """Generate the report. By default, the maximum value of each column is highlighted.

        Args:
            highlight (One of: &quot;max&quot;, &quot;min&quot;, False):
                Highlight either max or min model for each metric. If `False`,
                do not highlight. Defaults to "max".

        Raises:
            TypeError: Raised if the highlighting strategy is supported.

        Returns:
            Styler or DataFrame: A dataframe or a highlighted Styler.
        """
        if not self._results:
            return pd.DataFrame([])
        results = pd.DataFrame(self._results)
        results = results.set_index("name")
        if highlight == "max":
            return results.style.highlight_max()
        if highlight == "min":
            return results.style.highlight_min()
        if highlight not in ("max", "min", False):
            raise TypeError("Highlight must be one of: max, min, None")
        return results
