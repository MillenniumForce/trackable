#!/usr/bin/env python
"""Tests for `trackable` package."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler

from trackable import Report
from trackable.exceptions import ModelAlreadyExistsError


@pytest.fixture
def mock_report():
    """Simple fixture to mock a report."""
    X_test = np.array([])
    y_test = np.array([])
    metrics = [lambda x, y: 1.0]
    report = Report(X_test, y_test, metrics)
    return report


@pytest.fixture
def mock_report_complex():
    """Fixture to mock a complex report."""
    X_test = np.array([])
    y_test = np.array([])

    def mock_metric_1(x, y):
        return 1

    def mock_metric_2(x, y):
        return 2

    metrics = [mock_metric_1, mock_metric_2]
    report = Report(X_test, y_test, metrics)
    return report


@pytest.fixture
def mock_model():
    """Fixture to mock a dummy model."""
    model = Mock()
    model.predict = lambda x: x
    return model


def test_add_model_1(mock_report, mock_model):
    """Test add_model 1: add a single model."""
    mock_report.add_model(mock_model)
    print(mock_report._results, mock_report._models)
    assert mock_report._results == [{"<lambda>": 1, "name": "Mock"}]
    assert mock_report._models == {"Mock": mock_model}


def test_add_model_2(mock_report, mock_model):
    """Test add_model 2: add two models."""
    mock_report.add_model(mock_model, name="Mock A")
    mock_report.add_model(mock_model, name="Mock B")
    print(mock_report._results)
    assert mock_report._results == [{"<lambda>": 1, "name": "Mock A"}, {"<lambda>": 1, "name": "Mock B"}]
    assert mock_report._models == {"Mock A": mock_model, "Mock B": mock_model}


def test_add_model_3(mock_report_complex, mock_model):
    """Test add model 3: add model with multiple metrics."""
    mock_report_complex.add_model(mock_model)
    print(mock_report_complex._results, mock_report_complex._models)
    assert mock_report_complex._results == [{"mock_metric_1": 1, "mock_metric_2": 2, "name": "Mock"}]
    assert mock_report_complex._models == {"Mock": mock_model}


def test_add_model_with_exception(mock_report, mock_model):
    """Test add_model wit exception: add a duplicate model. Should raise an error."""
    mock_report.add_model(mock_model, name="Mock A")
    print(mock_report._models)
    with pytest.raises(ModelAlreadyExistsError):
        mock_report.add_model(mock_model, name="Mock A")


def test_generate_1(mock_report, mock_model):
    """Test generate 1: show correct results in dataframe."""
    mock_report.add_model(mock_model)
    report = mock_report.generate(False)
    correct = pd.DataFrame([{"<lambda>": 1.0, "name": "Mock"}]).set_index("name")
    print(report)
    print(correct)
    assert correct.equals(report)


def test_generate_2(mock_report, mock_model):
    """Test generate 2: show empty dataframe if no results."""
    report = mock_report.generate()
    assert pd.DataFrame([]).equals(report)


def test_generate_3(mock_report_complex, mock_model):
    """Test generate 3: multiple metrics, multiple models."""
    mock_report_complex.add_model(mock_model, name="Mock A")
    mock_report_complex.add_model(mock_model, name="Mock B")
    print(mock_report_complex._results)
    correct = pd.DataFrame(
        [
            {"mock_metric_1": 1, "mock_metric_2": 2, "name": "Mock A"},
            {"mock_metric_1": 1, "mock_metric_2": 2, "name": "Mock B"},
        ]
    ).set_index("name")
    assert correct.equals(mock_report_complex.generate(False))


def test_generate_4(mock_report, mock_model):
    """Test generate 4: test highlighting strategies."""
    mock_report.add_model(mock_model)
    report = mock_report.generate(highlight="max")
    assert isinstance(report, Styler)
    report = mock_report.generate(highlight="min")
    assert isinstance(report, Styler)
    report = mock_report.generate(highlight=False)
    assert isinstance(report, pd.DataFrame)
    with pytest.raises(TypeError):
        mock_report.generate(highlight="42")
