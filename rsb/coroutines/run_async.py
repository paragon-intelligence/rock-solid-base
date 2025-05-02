from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


async def run_async[**P, _T](
    func: Callable[P, _T],
    executor: ThreadPoolExecutor | None = None,
    timeout: float | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T:
    """
    Runs a synchronous callable asynchronously by executing it in a thread pool.

    This function is useful for calling blocking synchronous code from asynchronous
    code without blocking the event loop.

    Args:
        func: The synchronous callable to execute.
        executor: Optional custom thread pool executor to use.
                  If None, the default executor will be used.
        timeout: Maximum time to wait for the callable to complete (in seconds).
                 None means wait indefinitely.
        *args: Positional arguments to pass to the callable.
        **kwargs: Keyword arguments to pass to the callable.

    Returns:
        An awaitable that resolves to the result of the callable.

    Raises:
        asyncio.TimeoutError: If the operation takes longer than the specified timeout.
        RuntimeError: If called from a non-async context with no way to create a loop.
        Exception: Any exception raised by the callable will be propagated.
    """
    # Get or create a running event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop
        if threading.current_thread() is threading.main_thread():
            # We can create a loop in the main thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            # Cannot create a loop in a non-main thread with no existing loop
            raise RuntimeError(
                "Cannot create event loop in non-main thread with no existing loop. "
                "Either run in main thread, or ensure an event loop is running."
            )

    # Use partial to bind the arguments to the function
    bound_func = functools.partial(func, *args, **kwargs)

    # Create a future for cancellation handling
    future = None

    try:
        # Run the synchronous function in a thread pool
        future = loop.run_in_executor(executor, bound_func)

        # Wait for the result with optional timeout
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.CancelledError:
        # Handle cancellation by attempting to cancel the executor future
        if future and not future.done():
            # Note: This doesn't actually interrupt the thread, but it marks
            # the future as cancelled so we don't process the result
            future.cancel()
        raise
    except asyncio.TimeoutError:
        # Handle timeout by attempting to cancel the executor future
        if future and not future.done():
            future.cancel()
        raise
