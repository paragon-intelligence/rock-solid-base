from __future__ import annotations

from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, ClassVar, cast

from rsb.contracts.maybe_protocol import MaybeProtocol


class UnwrapFailedError(Exception):
    """Exception raised when unwrapping a Maybe with no value."""

    def __init__(self) -> None:
        super().__init__("Attempted to unwrap a Maybe with no value")


class Maybe[T = object](MaybeProtocol[T]):
    """
    A class that safely handles optional chaining for Python objects, emulating the `?.` operator
    found in languages like JavaScript. This allows for safe access to attributes and methods
    of objects that may be `None`, preventing `AttributeError` exceptions.

    This class implements the MaybeProtocol, providing monadic operations for handling
    potentially None values with type safety.

    **Usage Patterns:**

    1. **Type Annotation with Instance Creation:**
       ```python
       user_instance = User("Alice")
       maybe_user: Maybe[User] = Maybe(user_instance)
       ```

    2. **Handling Optional Values:**
       ```python
       maybe_none_user: Maybe[User] = Maybe(None)
       ```

    3. **Error Handling Behavior:**
       ```python
       # Default behavior: errors in operations are caught and return Maybe(None)
       default_maybe = Maybe(obj)

       # Strict behavior: errors in operations are raised
       strict_maybe = Maybe(obj, ignore_errors=False)
       ```

    **Usage Examples:**

    ```python
    >>> # Type annotation with instance creation
    >>> user_instance = User("Alice")
    >>> maybe_user: Maybe[User] = Maybe(user_instance)
    >>> maybe_user.name.unwrap()
    'Alice'

    >>> # Using map to transform the wrapped value
    >>> maybe_number: Maybe[int] = Maybe(10)
    >>> maybe_double: Maybe[int] = maybe_number.map(lambda x: x * 2)
    >>> maybe_double.unwrap()
    20

    >>> # Using value_or to provide fallback
    >>> maybe_none: Maybe[str] = Maybe(None)
    >>> maybe_none.value_or("Default Value")
    'Default Value'

    >>> # Using bind for chaining
    >>> maybe_upper: Maybe[str] = maybe_user.bind(lambda user: Maybe(user.name.upper()))
    >>> maybe_upper.unwrap()
    'ALICE'
    ```
    """

    _obj: T | None
    _ignore_errors: bool
    empty: ClassVar[Maybe[object] | None] = None  # Will be set after class definition

    def __init__(self, obj: T | None = None, ignore_errors: bool = True) -> None:
        self._obj = obj
        self._ignore_errors = ignore_errors
        super().__init__()

    @property
    def obj(self) -> T | None:
        """Get the wrapped object."""
        return self._obj

    def __getattr__(self, attr: str) -> Maybe[object]:
        """
        Safely access an attribute of the wrapped object.

        Args:
            attr: The attribute name to access.

        Returns:
            A Maybe wrapping the attribute's value or Nothing.

        Examples:
            >>> class User:
            ...     def __init__(self, name):
            ...         self.name = name
            >>> user = User("Alice")
            >>> maybe_user: Maybe[User] = Maybe(user)
            >>> maybe_user.name.unwrap()
            'Alice'

            >>> maybe_none: Maybe[User] = Maybe(None)
            >>> maybe_none.name.unwrap()  # Raises UnwrapFailedError
        """
        if self._obj is None:
            return Maybe[object](None, self._ignore_errors)
        try:
            return Maybe[object](getattr(self._obj, attr), self._ignore_errors)
        except AttributeError:
            if self._ignore_errors:
                return Maybe[object](None, self._ignore_errors)
            raise

    def __call__(self, *args: object, **kwargs: object) -> Maybe[object]:
        """
        Safely call the wrapped object if it's callable.

        Args:
            *args: Positional arguments for the callable.
            **kwargs: Keyword arguments for the callable.

        Returns:
            A Maybe wrapping the result of the call or Nothing.

        Examples:
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>> maybe_greet: Maybe[Callable[[str], str]] = Maybe(greet)
            >>> maybe_greet("Alice").unwrap()
            'Hello, Alice!'
        """
        if self.obj is None or not callable(self.obj):
            return Maybe[object](None, self._ignore_errors)
        try:
            result: Any = self.obj(*args, **kwargs)
            return Maybe[object](cast(object, result), self._ignore_errors)
        except Exception:
            if self._ignore_errors:
                return Maybe[object](None, self._ignore_errors)
            raise

    def map[U](
        self, function: Callable[[T], U], ignore_errors: bool | None = None
    ) -> MaybeProtocol[U]:
        """
        Transforms the value inside a successful container using a pure function.

        For a Some instance, applies the function to the contained value.
        For a Nothing instance, returns Nothing unchanged.

        Args:
            function: A function to apply to the contained value
            ignore_errors: Optional override for instance error handling behavior.

        Returns:
            A new Maybe containing the transformed value

        Examples:
            >>> def add_one(x: int) -> int:
            ...     return x + 1
            >>> Some(1).map(add_one)  # Returns Some(2)
            >>> Nothing.map(add_one)  # Returns Nothing
        """
        effective_ignore_errors = (
            self._ignore_errors if ignore_errors is None else ignore_errors
        )

        if self.obj is None:
            return Maybe[U](None, effective_ignore_errors)
        try:
            return Maybe[U](function(self.obj), effective_ignore_errors)
        except Exception:
            if effective_ignore_errors:
                return Maybe[U](None, effective_ignore_errors)
            raise

    def bind[U](
        self,
        function: Callable[[T], MaybeProtocol[U]],
        ignore_errors: bool | None = None,
    ) -> MaybeProtocol[U]:
        """
        Chains operations that return Maybe containers.

        For a Some instance, applies the function to the contained value.
        For a Nothing instance, returns Nothing unchanged.

        Args:
            function: A function that takes a value and returns a Maybe
            ignore_errors: Optional override for instance error handling behavior.

        Returns:
            The result of applying the function to the contained value

        Examples:
            >>> def half(x: int) -> Maybe[float]:
            ...     return Some(x / 2) if x != 0 else Nothing
            >>> Some(4).bind(half)  # Returns Some(2.0)
            >>> Some(0).bind(half)  # Returns Nothing
        """
        effective_ignore_errors = (
            self._ignore_errors if ignore_errors is None else ignore_errors
        )

        if self.obj is None:
            return Maybe[U](None, effective_ignore_errors)
        try:
            result = function(self.obj)
            if isinstance(result, Maybe):
                result._ignore_errors = effective_ignore_errors
            return result
        except Exception:
            if effective_ignore_errors:
                return Maybe[U](None, effective_ignore_errors)
            raise

    def bind_optional[U](
        self, function: Callable[[T], U | None], ignore_errors: bool | None = None
    ) -> Maybe[U]:
        """
        Binds a function returning an optional value over a container.

        For a Some instance, applies the function to the contained value and
        returns Some if the result is not None, or Nothing if it is None.
        For a Nothing instance, returns Nothing unchanged.

        Args:
            function: A function that takes a value and returns a value or None
            ignore_errors: Optional override for instance error handling behavior.

        Returns:
            Some containing the result or Nothing

        Examples:
            >>> def get_length(s: str) -> int | None:
            ...     return len(s) if s else None
            >>> Some("hello").bind_optional(get_length)  # Returns Some(5)
            >>> Some("").bind_optional(get_length)  # Returns Nothing
        """
        effective_ignore_errors = (
            self._ignore_errors if ignore_errors is None else ignore_errors
        )

        if self.obj is None:
            return Maybe[U](None, effective_ignore_errors)
        try:
            result = function(self.obj)
            return Maybe[U](result, effective_ignore_errors)
        except Exception:
            if effective_ignore_errors:
                return Maybe[U](None, effective_ignore_errors)
            raise

    def apply[U](
        self,
        function: MaybeProtocol[Callable[[T], U]],
        ignore_errors: bool | None = None,
    ) -> MaybeProtocol[U]:
        """
        Applies a function wrapped in a Maybe to the value in this Maybe.

        Args:
            function: A Maybe containing a function to apply
            ignore_errors: Optional override for instance error handling behavior.

        Returns:
            A new Maybe containing the result of applying the function

        Examples:
            >>> from returns.maybe import Some, Nothing
            >>> Some(1).apply(Some(lambda x: x + 1))  # Returns Some(2)
            >>> Some(1).apply(Nothing)  # Returns Nothing
            >>> Nothing.apply(Some(lambda x: x + 1))  # Returns Nothing
        """
        effective_ignore_errors = (
            self._ignore_errors if ignore_errors is None else ignore_errors
        )

        if self.obj is None or function.obj is None:  # type: ignore
            return Maybe[U](None, effective_ignore_errors)
        try:
            return Maybe[U](function.obj(self.obj), effective_ignore_errors)  # type: ignore
        except Exception:
            if effective_ignore_errors:
                return Maybe[U](None, effective_ignore_errors)
            raise

    def lash(
        self,
        function: Callable[[Any], MaybeProtocol[T]],
        ignore_errors: bool | None = None,
    ) -> MaybeProtocol[T]:
        """
        Composes failed container with a function that returns a container.

        For a Some instance, returns the Some instance unchanged.
        For a Nothing instance, applies the function to None.

        Args:
            function: A function that takes None and returns a Maybe
            ignore_errors: Optional override for instance error handling behavior.

        Returns:
            Original Some instance or the result of the function

        Examples:
            >>> def fallback(arg=None) -> Maybe[str]:
            ...     return Some('default')
            >>> Some('value').lash(fallback)  # Returns Some('value')
            >>> Nothing.lash(fallback)  # Returns Some('default')
        """
        effective_ignore_errors = (
            self._ignore_errors if ignore_errors is None else ignore_errors
        )

        if self.obj is not None:
            return self
        try:
            result = function(None)
            if isinstance(result, Maybe):
                result._ignore_errors = effective_ignore_errors
            return result
        except Exception:
            if effective_ignore_errors:
                return Maybe[T](None, effective_ignore_errors)
            raise

    def unwrap(self) -> T | None:
        """
        Get value from successful container or raise exception for failed one.

        Returns:
            The contained value

        Raises:
            UnwrapFailedError: If this is Nothing

        Examples:
            >>> Some(1).unwrap()  # Returns 1
            >>> Nothing.unwrap()  # Raises UnwrapFailedError
        """
        return self.obj

    def failure(self) -> None:
        """
        Get failed value from failed container or raise exception from success.

        Returns:
            None for Nothing

        Raises:
            UnwrapFailedError: If this is Some

        Examples:
            >>> Nothing.failure()  # Returns None
            >>> Some(1).failure()  # Raises UnwrapFailedError
        """
        if self.obj is not None:
            raise UnwrapFailedError()
        return None

    def value_or[U](self, default_value: U) -> T | U:
        """
        Get value from successful container or default value from failed one.

        Args:
            default_value: Value to return if this is Nothing

        Returns:
            The contained value or the default value

        Examples:
            >>> Some(1).value_or(0)  # Returns 1
            >>> Nothing.value_or(0)  # Returns 0
        """
        return self.obj if self.obj is not None else default_value

    def or_else_call[U](self, function: Callable[[], U]) -> T | U:
        """
        Get value from successful container or call function for failed one.

        Similar to value_or but using a lazy value from a function call.

        Args:
            function: Function to call if this is Nothing

        Returns:
            The contained value or the result of calling the function

        Examples:
            >>> Some(1).or_else_call(lambda: 0)  # Returns 1
            >>> Nothing.or_else_call(lambda: 0)  # Returns 0
        """
        if self.obj is not None:
            return self.obj
        return function()

    def __iter__(self) -> Iterator[T]:  # type: ignore[reportIncompatibleMethodOverride]
        """
        Iterator interface for use with do-notation.

        Yields:
            The contained value

        Raises:
            UnwrapFailedError: If this is Nothing

        Examples:
            >>> list(iter(Some(5)))  # Returns [5]
            >>> list(iter(Nothing))  # Raises UnwrapFailedError
        """
        if self.obj is None:
            raise UnwrapFailedError()
        yield self.obj

    def __bool__(self) -> bool:
        """
        Allow `Maybe` instances to be used in boolean contexts.

        Returns:
            `True` if the wrapped object is not None; `False` otherwise.

        Examples:
            >>> bool(Some(5))
            True
            >>> bool(Nothing)
            False
        """
        return self.obj is not None

    def __eq__(self, other: object) -> bool:
        """
        Equality comparison between `Maybe` instances or with raw values.

        Args:
            other: Another `Maybe` instance or a raw value to compare with.

        Returns:
            `True` if both wrapped objects are equal; `False` otherwise.

        Examples:
            >>> Some(5) == Some(5)
            True
            >>> Some(5) == Some(10)
            False
            >>> Some(5) == 5
            True
            >>> Nothing == None
            True
        """
        if isinstance(other, Maybe):
            return self.obj == other.obj  # type: ignore[reportUnknownVariableType]
        return self.obj == other

    def __ne__(self, other: object) -> bool:
        """
        Non-equality comparison between `Maybe` instances or with raw values.

        Args:
            other: Another `Maybe` instance or a raw value to compare with.

        Returns:
            `True` if both wrapped objects are not equal; `False` otherwise.

        Examples:
            >>> Some(5) != Some(5)
            False
            >>> Some(5) != Some(10)
            True
            >>> Some(5) != 5
            False
            >>> Nothing != None
            False
        """
        return not self.__eq__(other)

    def __getitem__(self, key: object) -> Maybe[object]:
        """
        Safely access an item by key/index if the wrapped object supports indexing.

        Args:
            key: The key/index to access.

        Returns:
            A Maybe wrapping the item's value or Nothing.

        Examples:
            >>> maybe_dict = Maybe({"a": 1, "b": 2})
            >>> maybe_dict["a"].unwrap()
            1
            >>> maybe_dict["c"].unwrap()  # Raises UnwrapFailedError
        """
        if self.obj is None:
            return Maybe(None, self._ignore_errors)

        # Mapping type (dict-like)
        if isinstance(self.obj, Mapping):
            try:
                return Maybe(self.obj[key], self._ignore_errors)  # type: ignore[reportUnknownArgumentType]
            except (KeyError, TypeError):
                if self._ignore_errors:
                    return Maybe(None, self._ignore_errors)
                raise

        # Sequence type (list-like)
        elif isinstance(self.obj, Sequence):
            if not isinstance(key, (int, slice)):
                return Maybe(None, self._ignore_errors)
            try:
                return Maybe(self.obj[key], self._ignore_errors)  # type: ignore[reportUnknownArgumentType]
            except (IndexError, TypeError):
                if self._ignore_errors:
                    return Maybe(None, self._ignore_errors)
                raise

        # Any other type with __getitem__
        elif hasattr(self.obj, "__getitem__"):
            try:
                # Access __getitem__ directly to avoid type errors
                get_item_method = getattr(self.obj, "__getitem__")
                return Maybe(get_item_method(key), self._ignore_errors)
            except (IndexError, KeyError, TypeError, AttributeError):
                if self._ignore_errors:
                    return Maybe(None, self._ignore_errors)
                raise

        return Maybe(None, self._ignore_errors)

    @classmethod
    def from_optional(
        cls, inner_value: T | None, ignore_errors: bool = True
    ) -> MaybeProtocol[T]:
        """
        Creates new instance of Maybe container based on an optional value.

        Args:
            inner_value: Value to wrap in Maybe or None
            ignore_errors: Whether to ignore errors in operations (defaults to True)

        Returns:
            Some containing the value if not None, otherwise Nothing

        Examples:
            >>> Maybe.from_optional(1)  # Returns Some(1)
            >>> Maybe.from_optional(None)  # Returns Nothing
        """
        return cls(inner_value, ignore_errors)

    @classmethod
    def from_value(cls, inner_value: T, ignore_errors: bool = True) -> MaybeProtocol[T]:
        """
        Creates new instance of Maybe container based on a value.

        Args:
            inner_value: Value to wrap in Maybe
            ignore_errors: Whether to ignore errors in operations (defaults to True)

        Returns:
            Some containing the value

        Examples:
            >>> Maybe.from_value(1)  # Returns Some(1)
            >>> Maybe.from_value(None)  # Returns Some(None)
        """
        return cls(inner_value, ignore_errors)

    @classmethod
    def do(cls, expr: Generator[T, None, None]) -> MaybeProtocol[T]:
        """
        Enables working with unwrapped values in a safe way through generator expressions.

        This implements the 'do notation' pattern from functional programming.

        Args:
            expr: Generator expression using for-yield syntax

        Returns:
            A Maybe containing the result of the generator expression

        Examples:
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
        try:
            # Attempt to run the generator to get the final value
            result = None
            for value in expr:
                result = value
            return cls(result)
        except UnwrapFailedError:
            # If any Maybe in the chain is Nothing, return Nothing
            return cls(None)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: Callable[[Any], Any]
    ) -> Any:
        """
        Pydantic integration: returns a core schema for validating inputs into a Maybe.

        This implementation wraps any input value into a Maybe instance.
        """
        from pydantic_core import core_schema

        def validate_maybe(input_value: Any) -> Any:
            return cls(input_value)

        return core_schema.no_info_after_validator_function(
            validate_maybe, core_schema.any_schema()
        )


# Set the empty class variable after class definition
Maybe.empty = Maybe(None)

# Create convenience aliases similar to the returns.maybe module
Nothing = Maybe.empty
Some = Maybe.from_value
