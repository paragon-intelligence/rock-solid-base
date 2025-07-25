from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, Callable, TypeVar, ParamSpec

_T = TypeVar("_T")
P = ParamSpec("P")

# Persistent background loop setup
_loop_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_started = threading.Event()


def _start_background_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_started.set()
    _loop.run_forever()


def _ensure_background_loop_running():
    global _loop_thread
    if _loop_thread is None or not _loop_thread.is_alive():
        _loop_thread = threading.Thread(target=_start_background_loop, daemon=True)
        _loop_thread.start()
        _loop_started.wait()


def run_sync(
    func: Callable[P, Awaitable[_T]],
    timeout: float | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T:
    """
    Runs an async function synchronously using a shared background event loop.
    """

    async def _async_wrapper() -> _T:
        return await func(*args, **kwargs)

    try:
        running_loop = asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        running_loop = None
        in_async_context = False

    if in_async_context:
        if threading.current_thread() is threading.main_thread():
            # Offload to separate thread if we're in main thread async context
            def run_in_thread() -> _T:
                return run_sync(func, timeout, *args, **kwargs)

            fut: Future[_T] = Future()
            t = threading.Thread(target=lambda: fut.set_result(run_in_thread()))
            t.start()
            return fut.result(timeout)
        else:
            # In async context inside a thread — schedule in running loop
            assert running_loop is not None
            return asyncio.run_coroutine_threadsafe(
                _async_wrapper(), running_loop
            ).result(timeout)

    # We're in sync context — use background loop
    _ensure_background_loop_running()
    assert _loop is not None
    future = asyncio.run_coroutine_threadsafe(_async_wrapper(), _loop)
    return future.result(timeout)
