from __future__ import annotations

import abc
import copy
import json
from collections.abc import Iterator, Sequence
from typing import Any


class ReadonlyCollection[T](abc.ABC):
    """Abstract base class for read-only collections that behave like sequences."""

    elements: Sequence[T]

    def __init__(self, elements: Sequence[T]) -> None:
        super().__init__()
        self.elements = elements

    def copy(self) -> ReadonlyCollection[T]:
        return ReadonlyCollection(elements=copy.deepcopy(self.elements))

    def json(self) -> dict[str, object]:
        return json.loads(json.dumps(self.elements))

    def __len__(self) -> int:
        """Return the number of items in the collection.

        Returns:
            int: The count of elements in the collection.
        """
        return len(self.elements)

    def __getitem__(self, index: int | slice) -> T | Sequence[T]:
        """Access an element or slice of elements by index.

        Args:
            index: An integer index or slice object.

        Returns:
            Either a single element of type T or a sequence of elements.

        Raises:
            IndexError: If the index is out of range.
        """
        return self.elements[index]

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the collection elements.

        Returns:
            An iterator yielding elements of type T.
        """
        return iter(self.elements)

    def __contains__(self, item: Any) -> bool:
        """Check if the collection contains the specified item.

        Args:
            item: The item to check for.

        Returns:
            bool: True if the item is in the collection, False otherwise.
        """
        return item in self.elements

    def __repr__(self) -> str:
        """Return a string representation for debugging.

        Returns:
            str: A string representation of the collection.
        """
        return f"{self.__class__.__name__}({list(self)})"

    def count(self, value: Any) -> int:
        """Count occurrences of a value in the collection.

        Args:
            value: The value to count.

        Returns:
            int: The number of occurrences.
        """
        return sum(1 for item in self if item == value)

    def index(self, value: Any, start: int = 0, stop: int | None = None) -> int:
        """Find the index of value in the collection.

        Args:
            value: The value to find.
            start: The starting index to search from.
            stop: The ending index to search up to.

        Returns:
            int: The index of the first occurrence.

        Raises:
            ValueError: If the value is not present.
        """
        if stop is None:
            stop = len(self)

        for i in range(start, stop):
            try:
                if self[i] == value:
                    return i
            except IndexError:
                break

        raise ValueError(f"{value} is not in collection")
