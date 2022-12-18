# trackable

[![pypi](https://img.shields.io/pypi/v/trackable.svg)](https://pypi.org/project/trackable/)
[![python](https://img.shields.io/pypi/pyversions/trackable.svg)](https://pypi.org/project/trackable/)
[![Build Status](https://github.com/MillenniumForce/trackable/actions/workflows/dev.yml/badge.svg)](https://github.com/MillenniumForce/trackable/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/MillenniumForce/trackable/branch/main/graphs/badge.svg)](https://codecov.io/github/MillenniumForce/trackable)

A minimalistic machine learning model tracker and reporting tool

* Documentation: <https://MillenniumForce.github.io/trackable>
* GitHub: <https://github.com/MillenniumForce/trackable>
* PyPI: <https://pypi.org/project/trackable/>
* Free software: MIT

`trackable` is a package focussed on users already familiar with machine learning in Python and aims to:

1. Provide a minimal model tracking tool with no frills
2. An intuitive and lightweight api

## Installation

The latest released version can be installed from [PyPI](https://pypi.org/project/trackable/) using:

```bash
# pip
pip install trackable
```

## Features

To start using `trackable` import the main reporting functionality via:

```python
from trackable import Report
```

It's simple to start using the package. The example below (although simplistic) shows how easy it
is to pick up the api:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from trackable import Report

X, y = make_classification()

lr = LogisticRegression().fit(X, y)
rf = RandomForestClassifier().fit(X, y)

# Instantiate the report...
report = Report(X, y, metrics = [accuracy_score, f1_score, roc_auc_score])

# Add models...
report.add_model(lr)
report.add_model(rf)

# Generate the report...
report.generate()
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
