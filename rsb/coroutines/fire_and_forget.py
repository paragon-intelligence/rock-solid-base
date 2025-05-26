import asyncio
import logging
from collections.abc import Callable
from typing import Any, Coroutine, Optional

logger = logging.getLogger(__name__)


def fire_and_forget(
    async_func: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any
) -> Optional[asyncio.Task[Any]]:
    """
    Schedules async_func to run without waiting for completion.

    If called from within an async context, creates a task and returns it.
    If called from sync context, runs the coroutine in a new event loop.

    Returns:
        The created Task if in async context, None otherwise.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context - create task with error handling
        task = loop.create_task(async_func(*args, **kwargs))

        # Add error handler to prevent silent failures
        def handle_exception(task: asyncio.Task[Any]) -> None:
            try:
                task.result()
            except Exception as e:
                logger.exception(f"Exception in fire-and-forget task: {e}")

        task.add_done_callback(handle_exception)
        return task

    except RuntimeError:
        # No running loop - we're in sync context
        # Run in a new event loop
        asyncio.run(async_func(*args, **kwargs))
        return None
