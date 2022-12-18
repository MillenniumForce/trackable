# Usage

At its core, `trackable` consists of three elements:

1. Instantiating the `Report` class
2. Adding models with `.add_model(...)`
3. Generating the report `.generate()`

To get access to the core functionality use,

```python
from trackable import Report
```

In the following sections, assume that all code chunks are in the same python session.

## Creating a new report

To create a report, you need to have defined:

1. Testing data, `X_test` and `y_test`
2. A set of metrics, for example `accuracy_score` from scikit-learn.

Below is an example of what this might look like:

```python
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

report = Report(X, y, metrics = [accuracy_score, f1_score, roc_auc_score])
```

## Adding models

All models that are added to the report must contain a `.predict` method.
This is so that the report can calculate and compare metrics.

It's easy to add scikit-learn models, however, you may need to write a wrapper for more
complex models such as neural networks.

For exmaple, let's create a few models and add them to the report:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

lr = LogisticRegression().fit(X, y)
rf = RandomForestClassifier().fit(X, y)

report.add_model(lr)
report.add_model(rf)
```

## Generating the report

To generate the report call the `.generate` method.

By default, the report highlights the model with the maximum value for each metric.
However, this can easily be changed to the minimum value, or highlighting can be removed entirely.

Finally, to generate the report use:

```python
# Turn highlighting off unless you're in a Jupyter notebook
report.generate(highlight=False)
```

which outputs:

```python
                        accuracy_score  f1_score  roc_auc_score
name
LogisticRegression                0.91  0.909091           0.91
RandomForestClassifier            1.00  1.000000           1.00
```
