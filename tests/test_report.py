#!/usr/bin/env python
"""Tests for `trackable` package."""

from unittest.mock import Mock

import numpy as np
import pytest

from trackable import Report
from trackable.exceptions import ModelAlreadyExistsError


@pytest.fixture
def mock_report():
    """Simple fixture to mock a report"""
    X_test = np.array([])
    y_test = np.array([])
    metrics = [lambda x, y: 1.0]
    report = Report(X_test, y_test, metrics)
    return report


@pytest.fixture
def mock_model():
    """Fixture to mock a dummy model"""
    model = Mock()
    model.predict = lambda x: x
    return model


def test_add_model_1(mock_report, mock_model):
    """Test add_model 1: add a single model"""
    mock_report.add_model(mock_model)
    print(mock_report._results, mock_report._models)
    assert mock_report._results == [{'<lambda>': 1, 'name': 'Mock'}]
    assert mock_report._models == {'Mock': mock_model}


def test_add_model_2(mock_report, mock_model):
    """Test add_model 2: add two models"""
    mock_report.add_model(mock_model, name="Mock A")
    mock_report.add_model(mock_model, name="Mock B")
    print(mock_report._results)
    assert mock_report._results == [{'<lambda>': 1, 'name': 'Mock A'}, {'<lambda>': 1, 'name': 'Mock B'}]
    assert mock_report._models == {"Mock A": mock_model, "Mock B": mock_model}


def test_add_model_with_exception(mock_report, mock_model):
    """Test add_model 3: add a duplicate model. Should raise an error"""
    mock_report.add_model(mock_model, name="Mock A")
    print(mock_report._models)
    with pytest.raises(ModelAlreadyExistsError):
        mock_report.add_model(mock_model, name="Mock A")
