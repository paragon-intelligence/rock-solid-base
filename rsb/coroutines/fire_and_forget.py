import asyncio
from collections.abc import Callable, Awaitable
from typing import Any, Union, overload

P = tuple[Any, ...]
K = dict[str, Any]


@overload
def fire_and_forget(
    async_func_or_coro: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
) -> None: ...


@overload
def fire_and_forget(
    async_func_or_coro: Awaitable[Any],
) -> None: ...


def fire_and_forget(
    async_func_or_coro: Union[Callable[..., Awaitable[Any]], Awaitable[Any]],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Schedules an async function or coroutine to run in the existing event loop if one is running.
    Otherwise, it creates a new event loop and runs the coroutine to completion.

    This function does not wait for the coroutine to finish if a loop is already
    running ("fire-and-forget"). If no loop is detected in the current thread,
    it will block just long enough to run the coroutine in a newly-created loop
    (which is closed immediately afterward).

    Errors in fire-and-forget tasks are caught and logged to stdout to prevent
    them from crashing the calling code.

    Usage:
        # With async function
        fire_and_forget(async_function, arg1, arg2, kwarg1=value)

        # With coroutine
        fire_and_forget(async_function(arg1, arg2, kwarg1=value))

    Args:
        async_func_or_coro: Either an async function or a coroutine object.
        *args: Positional arguments to pass to the function (ignored if coroutine is passed).
        **kwargs: Keyword arguments to pass to the function (ignored if coroutine is passed).
    """

    # Validate arguments early (before creating async wrapper)
    if not callable(async_func_or_coro) and (args or kwargs):
        raise TypeError("Cannot pass arguments when using an awaitable directly")

    async def _safe_wrapper() -> None:
        """Wrapper that safely executes the async function/coroutine with error handling."""
        try:
            if callable(async_func_or_coro):
                # We have a callable, use args/kwargs
                await async_func_or_coro(*args, **kwargs)
            else:
                # We have an awaitable
                await async_func_or_coro
        except Exception as e:
            # In fire-and-forget mode, we log errors but don't propagate them
            print(f"Fire-and-forget task failed: {type(e).__name__}: {e}")

    try:
        # Attempt to get a running loop in the current thread.
        loop = asyncio.get_running_loop()

        if loop.is_running():
            # We have a loop, and it's actively running. Schedule the coroutine
            # to run asynchronously (true fire-and-forget).
            loop.create_task(_safe_wrapper())
        else:
            # We have a loop object in this thread, but it's not actually running.
            # Run the coroutine to completion (blocking briefly).
            loop.run_until_complete(_safe_wrapper())
    except RuntimeError:
        # No event loop in the current thread -> create one and run the coroutine
        # immediately to completion, then close the loop.
        asyncio.run(_safe_wrapper())


# Example usage and tests
if __name__ == "__main__":
    import time

    async def example_async_func(message: str, delay: float = 0.1) -> None:
        """Example async function for testing."""
        await asyncio.sleep(delay)
        print(f"[{time.time():.2f}] {message}")

    async def error_func() -> None:
        """Function that raises an error to test error handling."""
        await asyncio.sleep(0.05)
        raise ValueError("This is a test error")

    def test_fire_and_forget() -> None:
        print("🧪 Testing enhanced fire_and_forget function...\n")

        # Test 1: Using async function with arguments
        print("✅ Test 1: Async function with arguments")
        fire_and_forget(example_async_func, "Hello from function call!", 0.1)
        time.sleep(0.2)  # Wait for completion

        # Test 2: Using coroutine directly
        print("\n✅ Test 2: Coroutine directly")
        fire_and_forget(example_async_func("Hello from coroutine!", 0.1))
        time.sleep(0.2)  # Wait for completion

        # Test 3: Multiple fire-and-forget calls
        print("\n✅ Test 3: Multiple concurrent calls")
        for i in range(3):
            fire_and_forget(example_async_func, f"Message {i}", 0.05)
        time.sleep(0.2)  # Wait for all to complete

        # Test 4: Error handling (errors should not crash the program)
        print("\n✅ Test 4: Error handling")
        fire_and_forget(error_func())
        print("Error function scheduled (should see error message but not crash)")
        time.sleep(0.2)  # Wait for error to occur and be handled

        # Test 5: Edge case - passing args to coroutine (should error)
        print("\n✅ Test 5: Edge case - args with coroutine")
        coro = example_async_func("test", 0.1)
        try:
            # Type ignore because we're intentionally testing wrong usage
            fire_and_forget(coro, "extra_arg")  # type: ignore[misc]
            print("❌ Should have raised TypeError!")
        except TypeError as e:
            print(f"✅ Caught expected error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            # Clean up the coroutine to avoid warnings
            coro.close()

        # Test 6: In async context (simulated)
        print("\n✅ Test 6: Testing in async context")

        async def test_in_async_context() -> None:
            print("   Inside async context:")
            fire_and_forget(example_async_func, "From async context (function)", 0.05)
            fire_and_forget(example_async_func("From async context (coroutine)", 0.05))
            await asyncio.sleep(0.1)  # Let them complete

        # Run the async test
        asyncio.run(test_in_async_context())

        print("\n🎉 All tests completed!")

    test_fire_and_forget()
