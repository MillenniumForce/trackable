"""Module used to generate a minimal report to track ML models"""

from typing import List, Optional

from trackable import types


class Report:
    def __init__(
        self, X_test: types.Data, y_test: types.Data, metrics: List[types.Metric], sort_on: Optional[str] = None
    ) -> None:
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = metrics
        self.sort_on = sort_on
        self._models: List[dict] = []
        self._results: List[dict] = []

    def add_model(
        self,
        model: types.GenericModel,
        name: Optional[str] = None,
        X_test: Optional[types.Data] = None,
        y_test: Optional[types.Data] = None,
    ) -> None:
        X = X_test if X_test else self.X_test
        y = y_test if y_test else self.y_test
        name = name if name else model.__class__.__name__

        y_pred = model.predict(X)

        results = []
        for metric in self.metrics:
            results.append(metric(y, y_pred))

        self._results.append(dict(name=name, results=results))
        self._models.append(dict(name=name, model=model))

    def generate(self):
        pass
