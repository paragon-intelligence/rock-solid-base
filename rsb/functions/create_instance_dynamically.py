import inspect
from typing import Any


def create_instance_dynamically[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
    """
    Creates an instance of the specified class using only valid constructor arguments.

    Args:
        cls: The class to instantiate
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        An instance of the specified class
    """
    # Get constructor signature
    constructor = cls.__init__
    signature = inspect.signature(constructor)
    parameters = signature.parameters

    # Filter out 'self' parameter
    valid_params: dict[str, inspect.Parameter] = {
        name: param for name, param in parameters.items() if name != "self"
    }

    # Create a dictionary of valid kwargs
    valid_kwargs: dict[str, Any] = {}
    for name, _ in valid_params.items():
        # Check if the parameter is in kwargs
        if name in kwargs:
            valid_kwargs[name] = kwargs[name]

    # For positional args, we only pass them if we have enough parameters
    valid_args: tuple[Any, ...] = args[: len(valid_params) - len(valid_kwargs)]

    # Create and return the instance
    return cls(*valid_args, **valid_kwargs)
