"""Internal module for typing extensions."""


from typing import Any, Callable, Protocol

Metric = Callable[[Any, Any], Any]


class GenericModel(Protocol):
    """Generic Sklearn Type model."""

    def predict(self, data: Any) -> Any:
        """Generic predict method."""
        ...
