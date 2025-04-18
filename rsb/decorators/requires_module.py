from __future__ import annotations

import functools
import importlib
from typing import Callable, cast


def requires_module[**P, R](
    module_name: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator that checks if a module is installed before executing the function.

    Args:
        module_name: The name of the module to check.

    Returns:
        A decorator function that will check for the module when the
        decorated function is called.

    Example:
        @requires_module("numpy")
        def analyze_data(data: list[float]) -> list[float]:
            import numpy as np
            return list(np.array(data) * 2)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                importlib.import_module(module_name)
            except ImportError:
                raise ImportError(
                    f"The required module '{module_name}' is not installed."
                )
            return func(*args, **kwargs)

        return cast(Callable[P, R], wrapper)

    return decorator
