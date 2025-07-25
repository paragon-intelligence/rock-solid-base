from __future__ import annotations

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable
from contextvars import copy_context
import atexit

# Global thread pool to avoid creating new threads repeatedly
_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="run_sync")

# Store the main thread's event loop specifically
_main_loop_ref: weakref.ReferenceType[asyncio.AbstractEventLoop] | None = None
_main_thread_id = threading.main_thread().ident

# Cleanup the executor on exit
atexit.register(lambda: _executor.shutdown(wait=True))


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

    # Case 1: No running event loop - create and run with proper cleanup
    if current_loop is None:
        # Always use the thread approach to avoid cleanup issues
        return _run_in_thread_with_new_loop(_async_wrapper, timeout)

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
    """Run a coroutine in a new event loop in a separate thread with proper cleanup."""
    # Capture the current context for context variables
    ctx = copy_context()

    def run_with_new_loop():
        # Create a fresh event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the coroutine in the new loop
            result = loop.run_until_complete(coro_func())

            # Force cleanup of any remaining tasks before closing
            _cleanup_loop_tasks(loop)

            return result
        finally:
            # More aggressive cleanup
            try:
                # Give any pending callbacks a chance to run
                loop.call_soon(lambda: None)
                loop.run_until_complete(asyncio.sleep(0))

                # Final task cleanup
                _cleanup_loop_tasks(loop)

            except Exception:
                # Ignore cleanup errors
                pass
            finally:
                try:
                    loop.close()
                except Exception:
                    # Ignore close errors
                    pass
                finally:
                    # Clear the event loop for this thread
                    asyncio.set_event_loop(None)

    # Submit to thread pool with context
    future = _executor.submit(ctx.run, run_with_new_loop)
    return future.result(timeout)


def _cleanup_loop_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Aggressively clean up all tasks in the loop."""
    try:
        # Get all tasks
        pending = asyncio.all_tasks(loop)
        if not pending:
            return

        # Cancel all tasks
        for task in pending:
            if not task.done():
                task.cancel()

        # Wait for cancellation with a short timeout
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True), timeout=1.0
                )
            )
        except asyncio.TimeoutError:
            # If tasks don't cancel quickly, force them
            for task in pending:
                if not task.done():
                    try:
                        task.cancel()
                        # Don't wait - just cancel and move on
                    except Exception:
                        pass

    except Exception:
        # Ignore all cleanup errors - we're shutting down anyway
        pass


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
        async def _test_coro():
            pass

        test_future = asyncio.run_coroutine_threadsafe(_test_coro(), loop)
        test_future.result(timeout=0.01)  # Very short timeout for quick test

        return loop

    except Exception:
        # Loop is not usable, clear the reference
        _main_loop_ref = None
        return None
