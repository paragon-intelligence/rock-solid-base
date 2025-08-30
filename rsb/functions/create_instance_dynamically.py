import inspect
from typing import Any

from pydantic import BaseModel


def create_instance_dynamically[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
    """
    Creates an instance of the specified class using only valid constructor arguments.
    Handles Pydantic BaseModels specially for robustness.

    Args:
        cls: The class to instantiate
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        An instance of the specified class
    """
    if issubclass(cls, BaseModel):
        # For Pydantic v2, use model_fields to get valid field names
        valid_fields = cls.model_fields
        valid_kwargs = {name: kwargs[name] for name in valid_fields if name in kwargs}
        # Pydantic typically doesn't use positional args, so ignore *args here
        return cls(**valid_kwargs)  # type: ignore

    # Original logic for non-Pydantic classes
    constructor = cls.__init__
    signature = inspect.signature(constructor)
    parameters = signature.parameters

    valid_params: dict[str, inspect.Parameter] = {
        name: param for name, param in parameters.items() if name != "self"
    }

    valid_kwargs: dict[str, Any] = {}
    for name, _ in valid_params.items():
        if name in kwargs:
            valid_kwargs[name] = kwargs[name]

    valid_args: tuple[Any, ...] = args[: len(valid_params) - len(valid_kwargs)]

    return cls(*valid_args, **valid_kwargs)
