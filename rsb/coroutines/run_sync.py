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

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_async_wrapper())

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(_async_wrapper())
        else:

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
        return asyncio.run_coroutine_threadsafe(_async_wrapper(), loop).result(timeout)
