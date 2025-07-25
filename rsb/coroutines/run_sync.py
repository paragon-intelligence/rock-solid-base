from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable


def run_sync[**P, _T](
    func: Callable[P, Awaitable[_T]],
    timeout: float | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T:
    """
    Runs a callable synchronously. If called from an async context in the main thread,
    it runs the callable in a new event loop in a separate thread. Otherwise, it
    runs the callable directly or using `run_coroutine_threadsafe`.

    Args:
        func: The callable to execute.
        timeout: Maximum time to wait for the callable to complete (in seconds).
                 None means wait indefinitely.
        *args: Positional arguments to pass to the callable.
        **kwargs: Keyword arguments to pass to the callable.

    Returns:
        The result of the callable.
    """

    async def _async_wrapper() -> _T:
        return await func(*args, **kwargs)

    # Try to get the running loop, but handle the case where there isn't one
    try:
        loop = asyncio.get_running_loop()
        loop_is_running = True
    except RuntimeError:
        # No running event loop - we're in a synchronous context
        loop = None
        loop_is_running = False

    # If there's no running loop, just use asyncio.run
    if not loop_is_running:
        return asyncio.run(_async_wrapper())

    # We have a running loop - need to handle this carefully
    if threading.current_thread() is threading.main_thread():
        # We're in the main thread with a running loop
        # We need to run in a separate thread to avoid blocking
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_async_wrapper())
            finally:
                new_loop.close()

        with ThreadPoolExecutor() as pool:
            future = pool.submit(run_in_new_loop)
            return future.result(timeout)
    else:
        # We're in a background thread with a running loop in the main thread
        # Use run_coroutine_threadsafe to schedule on the main loop
        assert loop is not None  # loop_is_running=True guarantees loop is not None
        return asyncio.run_coroutine_threadsafe(_async_wrapper(), loop).result(timeout)
