import asyncio
from collections.abc import Callable
from typing import Any, Coroutine


def fire_and_forget(
    async_func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
) -> None:
    """
    Schedules the async_func to run in the existing event loop if one is running.
    Otherwise, it creates a new event loop and runs the coroutine to completion.

    This function does not wait for the coroutine to finish if a loop is already
    running ("fire-and-forget"). If no loop is detected in the current thread,
    it will block just long enough to run `async_func()` in a newly-created loop
    (which is closed immediately afterward).

    Args:
        async_func: The asynchronous function (coroutine) to run.
        *args: Positional arguments to pass to the coroutine.
        **kwargs: Keyword arguments to pass to the coroutine.
    """
    try:
        # Attempt to get a running loop in the current thread.
        loop = asyncio.get_running_loop()

        if loop.is_running():
            # We have a loop, and it's actively running. Schedule the coroutine
            # to run asynchronously (true fire-and-forget).
            loop.create_task(async_func(*args, **kwargs))
        else:
            # We have a loop object in this thread, but it's not actually running.
            # Run the coroutine to completion (blocking briefly).
            loop.run_until_complete(async_func(*args, **kwargs))
    except RuntimeError:
        # No event loop in the current thread -> create one and run the coroutine
        # immediately to completion, then close the loop.
        asyncio.run(async_func(*args, **kwargs))
