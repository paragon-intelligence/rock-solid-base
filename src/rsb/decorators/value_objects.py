from __future__ import annotations


def valueobject[T](cls: type[T]) -> type[T]:
    return cls
