import abc
from collections.abc import Sequence
from typing import Any


class Reader[T = Any, I = str](abc.ABC):
    """
    Interface defining a reader for a single entity by its identifier.

    Parameters:
        T: The type of entity to be read
        I: The type of identifier used, defaults to string
    """

    @abc.abstractmethod
    def read(self, uid: I) -> T:
        """
        Read a single entity by its identifier.

        Args:
            uid: The unique identifier of the entity to read

        Returns:
            The entity of type T corresponding to the given identifier
        """
        ...


class BulkReader[T](abc.ABC):
    """
    Interface defining a reader for multiple entities matching specified filters.

    Parameters:
        T: The type of entity to be read
    """

    @abc.abstractmethod
    def read_all(self, filters: dict[str, object] | None = None) -> Sequence[T]:
        """
        Read multiple entities matching the specified filters.

        Args:
            filters: Optional dictionary of filter criteria to apply

        Returns:
            A readonly collection of entities matching the filters
        """
        ...


class AsyncReader[T = Any, I = str](abc.ABC):
    """
    Interface defining an asynchronous reader for a single entity by its identifier.

    Parameters:
        T: The type of entity to be read
        I: The type of identifier used, defaults to string
    """

    @abc.abstractmethod
    async def read(self, uid: I, filters: dict[str, object] | None = None) -> T:
        """
        Asynchronously read a single entity by its identifier.

        Args:
            uid: The unique identifier of the entity to read
            filters: Optional dictionary of filter criteria to apply

        Returns:
            The entity of type T corresponding to the given identifier
        """
        ...


class AsyncBulkReader[T](abc.ABC):
    """
    Interface defining an asynchronous reader for multiple entities matching specified filters.

    Parameters:
        T: The type of entity to be read
    """

    @abc.abstractmethod
    async def read_all(self, filters: dict[str, object] | None = None) -> Sequence[T]:
        """
        Asynchronously read multiple entities matching the specified filters.

        Args:
            filters: Optional dictionary of filter criteria to apply

        Returns:
            A readonly collection of entities matching the filters
        """
        ...
