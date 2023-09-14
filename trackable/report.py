"""Module used to generate a minimal report to track ML models."""

import os
import pickle
import shutil
import tempfile
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar, Union

import pandas as pd
from pandas.io.formats.style import Styler

from trackable import types
from trackable.exceptions import ArchiveAlreadyExistsError, ModelAlreadyExistsError, ModelDoesNotExistError

__all__ = ["Report"]

R = TypeVar("R", bound="Report")


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

    @classmethod
    def load(cls: Type[R], X_test: Any, y_test: Any, metrics: List[types.Metric], path: str = "report") -> R:
        """Loads a report from an archive. See the save report class method.

        The class method loads the stored models and results from the archive.
        The testing data and metrics must be provided separately.

        Args:
            X_test (Any): Testing data. Should be identical to the original report.
            y_test (Any): Ground truth prediction data. Should be identical to the original report.
            metrics (List[types.Metric]): List of metrics. Should be identical to the original report.
            path (str): Path to archived report. Don't append ".zip".
                Defaults to 'report' since this is the default path for saving.

        Returns:
            R: Loaded report class
        """
        archive_type = "zip"
        mode = "rb"
        results_dir = "trackable_results"
        models_sub_folder = "models"
        zip_path = f"{path}.{archive_type}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(zip_path, tmp_dir, format=archive_type)

            models_dir = os.path.join(tmp_dir, models_sub_folder)
            models = {}
            results = []
            for model_name in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_name)
                with open(model_path, mode) as f:
                    loaded_model = pickle.load(f)
                    models[model_name] = loaded_model

                results_path = os.path.join(tmp_dir, results_dir)
                with open(results_path, mode) as f:
                    results = pickle.load(f)

        report = cls(X_test, y_test, metrics)
        report._models = models
        report._results = results
        return report

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

    def get_model(self, name: str) -> types.GenericModel:
        """Get a model from a report given its name.

        Args:
            name (str): Name of a model

        Raises:
            ModelDoesNotExistError: Raised if the given model name does not exist in the report

        Returns:
            types.GenericModel: A generic model
        """
        try:
            return self._models[name]
        except KeyError:
            raise ModelDoesNotExistError(f"Model '{name}' does not exist.")

    def remove_model(self, name: str) -> types.GenericModel:
        """Removes a model given its name. The model is returned and then
        removed from the report.

        Args:
            name (str): Name of a model

        Raises:
            ModelDoesNotExistError: Raised if the given model name does not exist in the report

        Returns:
            types.GenericModel: A generic model
        """
        try:
            return self._models.pop(name)
        except KeyError:
            raise ModelDoesNotExistError(f"Model '{name}' does not exist.")

    def save(self, path: str = "report") -> None:
        """Saves report to an archive which can be loaded later.
        Saved reports will not be overwritten and will raise an exception.

        It is assumed that each model in the report can be pickled.

        Note: X_test, y_test and metrics are NOT saved.
        They should be saved separetely.

        Args:
            path (str, optional): Path to save archive.
            Defaults to "report" in current working directory.
            ".zip" is appended automatically.

        Raises:
            ArchiveAlreadyExistsError: Raised when a report archive with
            the same name already exists.
        """
        archive = "zip"
        mode = "wb"
        results = "trackable_results"
        models_sub_folder = "models"
        zip_path = f"{path}.{archive}"

        if os.path.isfile(zip_path):
            raise ArchiveAlreadyExistsError(
                f"{zip_path} already exists. Provide another path to avoid overwriting existing archives."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            models_dir = os.path.join(tmp_dir, models_sub_folder)
            os.mkdir(models_dir)

            for name, model in self._models.items():
                model_path = os.path.join(models_dir, name)
                with open(model_path, mode) as f:
                    pickle.dump(model, f)

            results_path = os.path.join(tmp_dir, results)
            with open(results_path, mode) as f:
                pickle.dump(self._results, f)

            shutil.make_archive(path, archive, tmp_dir)
