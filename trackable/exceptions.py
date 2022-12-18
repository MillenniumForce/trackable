"""Internal module for exception classes."""


class ModelAlreadyExistsError(Exception):
    """Raise error if the model already exists. This is to avoid duplicates."""
