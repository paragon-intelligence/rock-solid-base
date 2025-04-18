from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MaybeProtocol[T](
    Iterable[T],
    Protocol,
):
    def apply[U](self, function: MaybeProtocol[Callable[[T], U]]) -> MaybeProtocol[U]:
        """
        Applies a function wrapped in a container to the value in this container.

        Args:
            function: A container containing a function to apply

        Returns:
            A new container containing the result of applying the function
        """
        ...

    def bind[U](self, function: Callable[[T], MaybeProtocol[U]]) -> MaybeProtocol[U]:
        """
        Chains operations that return containers.

        Args:
            function: A function that takes a value and returns a container

        Returns:
            The result of applying the function to the contained value
        """
        ...

    @classmethod
    def do(
        cls,
        expr: Generator[T, None, None],
    ) -> MaybeProtocol[T]:
        """
        Enables working with unwrapped values in a safe way through generator expressions.

        This implements the 'do notation' pattern from functional programming.

        Args:
            expr: Generator expression using for-yield syntax

        Returns:
            A Maybe containing the result of the generator expression

        Examples:
            >>> from returns.maybe import Maybe, Some, Nothing
            >>> Maybe.do(
            ...     x + y
            ...     for x in Some(1)
            ...     for y in Some(2)
            ... )  # Returns Some(3)
            >>> Maybe.do(
            ...     x + y
            ...     for x in Some(1)
            ...     for y in Nothing
            ... )  # Returns Nothing
        """
        ...

    def failure(self) -> None:
        """
        Get failed value from failed container or raise exception from success.

        Returns:
            None for Nothing

        Raises:
            UnwrapFailedError: If this is Some

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> Nothing.failure()  # Returns None
            >>> Some(1).failure()  # Raises UnwrapFailedError
        """
        ...

    def lash(
        self,
        function: Callable[[Any], MaybeProtocol[T]],
    ) -> MaybeProtocol[T]:
        """
        Composes failed container with a function that returns a container.

        For a Some instance, returns the Some instance unchanged.
        For a Nothing instance, applies the function to None.

        Args:
            function: A function that takes None and returns a Maybe

        Returns:
            Original Some instance or the result of the function

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> def fallback(arg=None) -> Maybe[str]:
            ...     return Some('default')
            >>> Some('value').lash(fallback)  # Returns Some('value')
            >>> Nothing.lash(fallback)  # Returns Some('default')
        """
        ...

    def map[U](self, function: Callable[[T], U]) -> MaybeProtocol[U]:
        """
        Transforms the value inside a successful container using a pure function.

        For a Some instance, applies the function to the contained value.
        For a Nothing instance, returns Nothing unchanged.

        Args:
            function: A function to apply to the contained value

        Returns:
            A new Maybe containing the transformed value

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> def add_one(x: int) -> int:
            ...     return x + 1
            >>> Some(1).map(add_one)  # Returns Some(2)
            >>> Nothing.map(add_one)  # Returns Nothing
        """
        ...

    def bind_optional[U](self, function: Callable[[T], U | None]) -> MaybeProtocol[U]:
        """
        Binds a function returning an optional value over a container.

        For a Some instance, applies the function to the contained value and
        returns Some if the result is not None, or Nothing if it is None.
        For a Nothing instance, returns Nothing unchanged.

        Args:
            function: A function that takes a value and returns a value or None

        Returns:
            Some containing the result or Nothing

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> def get_length(s: str) -> int | None:
            ...     return len(s) if s else None
            >>> Some("hello").bind_optional(get_length)  # Returns Some(5)
            >>> Some("").bind_optional(get_length)  # Returns Nothing
        """
        ...

    @classmethod
    def from_optional(
        cls,
        inner_value: T | None,
    ) -> MaybeProtocol[T]:
        """
        Creates new instance of Maybe container based on an optional value.

        Args:
            inner_value: Value to wrap in Maybe or None

        Returns:
            Some containing the value if not None, otherwise Nothing

        Examples:
            >>> from returns.maybe import Maybe
            >>> Maybe.from_optional(1)  # Returns Some(1)
            >>> Maybe.from_optional(None)  # Returns Nothing
        """
        ...

    def unwrap(self) -> T | None:
        """
        Get value from the container

        Returns:
            The contained value or None

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> Some(1).unwrap()  # Returns 1
            >>> Nothing.unwrap()  # Raises UnwrapFailedError
        """
        ...

    @classmethod
    def from_value(
        cls,
        inner_value: T,
    ) -> MaybeProtocol[T]:
        """
        Creates new instance of Maybe container based on a value.

        Args:
            inner_value: Value to wrap in Maybe

        Returns:
            Some containing the value

        Examples:
            >>> from returns.maybe import Maybe
            >>> Maybe.from_value(1)  # Returns Some(1)
            >>> Maybe.from_value(None)  # Returns Some(None)
        """
        ...

    def value_or[U](
        self,
        default_value: U,
    ) -> T | U:
        """
        Get value from successful container or default value from failed one.

        Args:
            default_value: Value to return if this is Nothing

        Returns:
            The contained value or the default value

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> Some(1).value_or(0)  # Returns 1
            >>> Nothing.value_or(0)  # Returns 0
        """
        ...
