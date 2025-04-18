from __future__ import annotations

import abc
from collections.abc import Iterator, MutableSequence, Sequence
from typing import overload


class WritableCollection[T](abc.ABC):
    """Abstract base class for writable collections that behave like mutable sequences.

    This extends ReadonlyCollection with methods for modifying the collection.
    """

    elements: MutableSequence[T]

    def __init__(self, elements: MutableSequence[T]) -> None:
        super().__init__()
        self.elements = elements

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

    def __contains__(self, item: T) -> bool:
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

    def count(self, value: T) -> int:
        """Count occurrences of a value in the collection.

        Args:
            value: The value to count.

        Returns:
            int: The number of occurrences.
        """
        return sum(1 for item in self if item == value)

    def index(self, value: T, start: int = 0, stop: int | None = None) -> int:
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

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Sequence[T]) -> None: ...

    def __setitem__(self, index: int | slice, value: T | Sequence[T]) -> None:
        """Set an element or slice of elements.

        Args:
            index: An integer index or slice object.
            value: The value(s) to set.

        Raises:
            IndexError: If the index is out of range.
            TypeError: If the value is not compatible.
        """
        self.elements[index] = value  # type: ignore

    def __delitem__(self, index: int | slice) -> None:
        """Delete an element or slice of elements.

        Args:
            index: An integer index or slice object.

        Raises:
            IndexError: If the index is out of range.
        """
        del self.elements[index]

    def insert(self, index: int, value: T) -> None:
        """Insert an element at the specified index.

        Args:
            index: The index at which to insert.
            value: The value to insert.
        """
        self.elements.insert(index, value)

    def append(self, value: T) -> None:
        """Add an element to the end of the collection.

        Args:
            value: The element to append.
        """
        self.elements.append(value)

    def extend(self, values: Sequence[T]) -> None:
        """Extend the collection with elements from an iterable.

        Args:
            values: The sequence of elements to add.
        """
        self.elements.extend(values)

    def pop(self, index: int = -1) -> T:
        """Remove and return an element at the given index.

        Args:
            index: The index of the element to remove (default: last element).

        Returns:
            The removed element.

        Raises:
            IndexError: If the collection is empty or index is out of range.
        """
        return self.elements.pop(index)

    def remove(self, value: T) -> None:
        """Remove the first occurrence of a value.

        Args:
            value: The value to remove.

        Raises:
            ValueError: If the value is not present.
        """
        self.elements.remove(value)

    def clear(self) -> None:
        """Remove all elements from the collection."""
        self.elements.clear()

    def reverse(self) -> None:
        """Reverse the elements in place."""
        self.elements.reverse()
