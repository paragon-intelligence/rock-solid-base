from __future__ import annotations


def abstractservice[T](cls: type[T]) -> type[T]:
    """
    Decorator to make a class both a dataclass and an abstract base class.

    Args:
        cls: The class to be decorated

    Returns:
        The decorated class, now a dataclass and inheriting from ABC
    """
    return cls
