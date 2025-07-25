from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable
from contextvars import copy_context

# Global thread pool to avoid creating new threads repeatedly
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="run_sync")

# Track event loops to check if they're still running
_loop_refs: weakref.WeakValueDictionary[int, asyncio.AbstractEventLoop] = (
    weakref.WeakValueDictionary()
)


def run_sync[**P, _T](
    func: Callable[P, Awaitable[_T]],
    timeout: float | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T:
    """
    Runs an async function synchronously in a thread-safe manner.

    This function handles multiple scenarios:
    - No event loop: Creates a new one
    - Main thread with loop: Runs in a separate thread
    - Background thread: Uses the appropriate event loop

    Args:
        func: The async function to execute
        timeout: Maximum execution time in seconds
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        The result of the async function
    """

    async def _async_wrapper() -> _T:
        return await func(*args, **kwargs)

    # Check current thread and event loop status
    current_thread = threading.current_thread()
    is_main_thread = current_thread is threading.main_thread()

    # Try to get the current event loop
    try:
        current_loop = asyncio.get_running_loop()
        # Store a weak reference to check later if it's still valid
        _loop_refs[id(current_loop)] = current_loop
    except RuntimeError:
        current_loop = None

    # Case 1: No running event loop - create and run
    if current_loop is None:
        # Simple case - just run in a new event loop
        try:
            return asyncio.run(_async_wrapper())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # Fallback: we might be in a weird state, use thread approach
                return _run_in_thread_with_new_loop(_async_wrapper, timeout)
            raise

    # Case 2: We have a running loop
    # Check if the loop is still valid and not closed
    if current_loop.is_closed():
        # Loop is closed, need to create a new one
        return _run_in_thread_with_new_loop(_async_wrapper, timeout)

    # Case 3: Main thread with running loop
    if is_main_thread:
        # We're in the main thread with a running loop
        # Must run in a separate thread to avoid blocking
        return _run_in_thread_with_new_loop(_async_wrapper, timeout)

    # Case 4: Background thread with a running loop somewhere
    # Need to find which loop we should schedule on

    # First, check if there's a main thread loop
    main_loop = _get_main_thread_loop()

    if main_loop and not main_loop.is_closed():
        # Schedule on the main thread's loop
        try:
            future = asyncio.run_coroutine_threadsafe(_async_wrapper(), main_loop)
            return future.result(timeout)
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                # Loop was closed while we were trying to use it
                return _run_in_thread_with_new_loop(_async_wrapper, timeout)
            raise
    else:
        # No valid main loop, run in a new thread
        return _run_in_thread_with_new_loop(_async_wrapper, timeout)


def _run_in_thread_with_new_loop[_T](
    coro_func: Callable[[], Awaitable[_T]], timeout: float | None
) -> _T:
    """Run a coroutine in a new event loop in a separate thread."""
    # Capture the current context for context variables
    ctx = copy_context()

    def run_with_new_loop():
        # Create a fresh event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the coroutine in the new loop
            return loop.run_until_complete(coro_func())
        finally:
            # Clean up
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Run until all tasks are cancelled
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            finally:
                loop.close()
                # Clear the event loop for this thread
                asyncio.set_event_loop(None)

    # Submit to thread pool with context
    future = _executor.submit(ctx.run, run_with_new_loop)
    return future.result(timeout)


def _get_main_thread_loop() -> asyncio.AbstractEventLoop | None:
    """Get the event loop of the main thread if it exists and is running."""
    # This is a bit tricky - we need to check if the main thread has a loop
    # We'll use our weak references to find valid loops

    for _, loop in list(_loop_refs.items()):
        if loop is not None and not loop.is_closed():  # type: ignore
            # Check if this loop belongs to the main thread
            # This is a heuristic - in practice, the main thread usually has the first loop created
            try:
                # Try to determine if this is the main loop by checking if we can schedule on it
                # from a background thread (which should work for the main loop)
                if threading.current_thread() is not threading.main_thread():
                    # We're in a background thread, so we can test
                    test_future = asyncio.run_coroutine_threadsafe(
                        asyncio.sleep(0), loop
                    )
                    test_future.result(timeout=0.1)
                    return loop
            except Exception:
                # This loop doesn't work, continue
                continue

    return None
