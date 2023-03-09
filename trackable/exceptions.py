"""Internal module for exception classes."""


class ModelAlreadyExistsError(Exception):
    """Raise error if the model already exists. This is to avoid duplicates."""


class ModelDoesNotExistError(KeyError):
    """Raises an error if the model does not exist. Used when fetching models."""
