from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable
from contextvars import copy_context

# Global thread pool to avoid creating new threads repeatedly
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="run_sync")

# Store the main thread's event loop specifically
_main_loop_ref: weakref.ReferenceType[asyncio.AbstractEventLoop] | None = None
_main_thread_id = threading.main_thread().ident


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
    is_main_thread = current_thread.ident == _main_thread_id

    # Try to get the current event loop
    try:
        current_loop = asyncio.get_running_loop()

        # If we're in the main thread, store a reference to its loop
        if is_main_thread:
            global _main_loop_ref
            _main_loop_ref = weakref.ref(current_loop)

    except RuntimeError:
        current_loop = None

    # Case 1: No running event loop - create and run
    if current_loop is None:
        try:
            return asyncio.run(_async_wrapper())
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                # Fallback: use thread approach
                return _run_in_thread_with_new_loop(_async_wrapper, timeout)
            raise

    # Case 2: Main thread with running loop
    if is_main_thread:
        # We're in the main thread with a running loop
        # Must run in a separate thread to avoid blocking
        return _run_in_thread_with_new_loop(_async_wrapper, timeout)

    # Case 3: Background thread - try to use main thread's loop if available
    main_loop = _get_main_thread_loop()

    if main_loop is not None:
        # Try to schedule on the main thread's loop
        try:
            future = asyncio.run_coroutine_threadsafe(_async_wrapper(), main_loop)
            return future.result(timeout)
        except Exception:
            # Main loop failed (closed, etc.), fall back to new thread
            pass

    # Case 4: No usable main loop, create new one in thread
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
    global _main_loop_ref

    if _main_loop_ref is None:
        return None

    loop = _main_loop_ref()
    if loop is None:
        # Loop was garbage collected
        _main_loop_ref = None
        return None

    # Double-check that the loop is still usable
    try:
        # Check if loop is closed
        if loop.is_closed():
            _main_loop_ref = None
            return None

        # Try a quick non-blocking test to see if we can schedule on it
        # We'll schedule a no-op coroutine and see if it works
        async def _test_coro():
            pass

        test_future = asyncio.run_coroutine_threadsafe(_test_coro(), loop)
        test_future.result(timeout=0.01)  # Very short timeout for quick test

        return loop

    except Exception:
        # Loop is not usable, clear the reference
        _main_loop_ref = None
        return None
