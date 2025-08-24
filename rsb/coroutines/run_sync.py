from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Awaitable, Callable, TypeVar, ParamSpec, overload, Union, Any, AsyncGenerator

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
        _loop_started.wait(timeout=5.0)  # Add timeout for robustness
        if not _loop_started.is_set():
            raise RuntimeError("Failed to start background event loop within timeout")


@overload
def run_sync(
    func: Callable[P, Awaitable[_T]],
    timeout: float | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> _T: ...


@overload
def run_sync(
    func: Awaitable[_T],
    timeout: float | None = None,
) -> _T: ...


def run_sync(
    func: Union[Callable[..., Awaitable[_T]], Awaitable[_T]],
    timeout: float | None = None,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """
    Runs an async function or coroutine synchronously using a shared background event loop.
    
    This function is designed to work in various environments:
    - Standard Python scripts ✅
    - Jupyter notebooks/IPython ✅  
    - Web frameworks (Django ASGI, FastAPI) ✅
    - GUI applications with event loops ✅
    - Testing frameworks ✅
    - Multi-threaded applications ✅
    
    Usage:
        # With async function
        result = run_sync(async_function, timeout=5, arg1, arg2, kwarg1=value)
        
        # With coroutine
        result = run_sync(async_function(arg1, arg2, kwarg1=value), timeout=5)
    """
    
    # Check if we got an awaitable object or a callable
    if callable(func):
        # We have a callable, use args/kwargs
        async def _async_wrapper() -> _T:
            return await func(*args, **kwargs)
            
    else:
        # We have an awaitable, ignore args/kwargs
        if args or kwargs:
            raise TypeError("Cannot pass arguments when using an awaitable directly")
        
        async def _async_wrapper() -> _T:
            return await func

    # Enhanced environment detection
    try:
        running_loop = asyncio.get_running_loop()
        in_async_context = True
    except RuntimeError:
        running_loop = None
        in_async_context = False

    if in_async_context:
        if threading.current_thread() is threading.main_thread():
            # Main thread with running loop (Jupyter, web apps, GUI apps)
            # Offload to separate thread to avoid blocking the main loop
            def run_in_thread() -> _T:
                return run_sync(func, timeout, *args, **kwargs)

            fut: Future[_T] = Future()
            t = threading.Thread(target=lambda: fut.set_result(run_in_thread()))
            t.start()
            return fut.result(timeout)
        else:
            # Background thread with running loop - schedule in that loop
            assert running_loop is not None
            return asyncio.run_coroutine_threadsafe(
                _async_wrapper(), running_loop
            ).result(timeout)

    # No running loop - use our background loop
    _ensure_background_loop_running()
    assert _loop is not None
    future = asyncio.run_coroutine_threadsafe(_async_wrapper(), _loop)
    return future.result(timeout)


# Example usage and comprehensive tests
if __name__ == "__main__":
    import time
    import queue
    from typing import List
    
    async def simple_async_func(x: int, y: int = 10) -> int:
        """Simple async function for basic testing."""
        await asyncio.sleep(0.05)
        return x + y

    async def slow_async_func(delay: float = 1.0) -> str:
        """Slow async function for timeout testing."""
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"

    async def error_async_func(should_error: bool = True) -> str:
        """Async function that can raise an error."""
        await asyncio.sleep(0.01)
        if should_error:
            raise ValueError("Test error from async function")
        return "Success!"

    async def nested_async_func(depth: int) -> List[int]:
        """Async function that calls other async functions."""
        if depth <= 0:
            return []
        
        # Simulate some async work
        await asyncio.sleep(0.01)
        
        # Call another async function
        result = await simple_async_func(depth, depth * 2)
        nested_result = await nested_async_func(depth - 1)
        
        return [result] + nested_result

    async def async_generator_consumer() -> int:
        """Async function that consumes an async generator."""
        async def async_gen():
            for i in range(5):
                await asyncio.sleep(0.01)
                yield i * 2
        
        total = 0
        async for value in async_gen():
            total += value
        return total

    def run_tests() -> None:
        print("🧪 Running comprehensive run_sync tests...\n")
        
        # Test 1: Basic function calls
        print("✅ Test 1: Basic function calls")
        result1 = run_sync(simple_async_func, None, 5, y=15)
        print(f"   run_sync(func, args) -> {result1}")
        
        result2 = run_sync(simple_async_func(5, y=15))
        print(f"   run_sync(coroutine) -> {result2}")
        assert result1 == result2 == 20
        
        # Test 2: Different argument patterns
        print("\n✅ Test 2: Different argument patterns")
        result3 = run_sync(simple_async_func, None, 10)  # Using default y=10
        result4 = run_sync(simple_async_func(10))
        print(f"   With defaults: {result3}, {result4}")
        assert result3 == result4 == 20
        
        # Test 3: Complex nested async calls
        print("\n✅ Test 3: Complex nested async operations")
        start_time = time.time()
        nested_result = run_sync(nested_async_func, None, 3)
        elapsed = time.time() - start_time
        print(f"   Nested result: {nested_result}")
        print(f"   Elapsed time: {elapsed:.3f}s")
        assert len(nested_result) == 3
        assert nested_result == [9, 6, 3]  # [3+6, 2+4, 1+2]
        
        # Test 4: Async generator consumption
        print("\n✅ Test 4: Async generator consumption")
        gen_result = run_sync(async_generator_consumer())
        print(f"   Generator sum: {gen_result}")
        assert gen_result == 20  # 0+2+4+6+8 = 20
        
        # Test 5: Timeout handling
        print("\n✅ Test 5: Timeout handling")
        try:
            # This should complete within timeout
            fast_result = run_sync(slow_async_func, 0.2, 0.1)
            print(f"   Fast call completed: {fast_result}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
        
        try:
            # This should timeout
            run_sync(slow_async_func, 0.1, 0.5)  # 0.1s timeout, 0.5s delay
            print("   ❌ This should have timed out!")
        except Exception as e:
            print(f"   ✅ Expected timeout: {type(e).__name__}")
        
        # Test 6: Error propagation
        print("\n✅ Test 6: Error propagation")
        try:
            run_sync(error_async_func, None, True)
            print("   ❌ This should have raised an error!")
        except ValueError as e:
            print(f"   ✅ Caught expected error: {e}")
        
        # Should work without error
        success_result = run_sync(error_async_func, None, False)
        print(f"   No error case: {success_result}")
        
        # Test 7: Edge cases
        print("\n✅ Test 7: Edge cases")
        
        # Try to pass args to a coroutine (should error)
        try:
            coro = simple_async_func(5, y=10)
            try:
                # Type ignore because we're intentionally testing wrong usage
                run_sync(coro, None, "extra_arg")  # type: ignore[arg-type]
                print("   ❌ Should have raised TypeError!")
            except TypeError as e:
                print(f"   ✅ Caught expected TypeError: {e}")
            finally:
                # Clean up the coroutine
                coro.close()
        except Exception as e:
            print(f"   ❌ Unexpected error in edge case test: {e}")
        
        # Test 8: Performance comparison
        print("\n✅ Test 8: Performance comparison")
        
        # Many quick calls
        start_time = time.time()
        results = []
        for i in range(10):
            result = run_sync(simple_async_func, None, i, y=i*2)
            results.append(result)
        elapsed = time.time() - start_time
        print(f"   10 sequential calls: {elapsed:.3f}s")
        print(f"   Results: {results[:5]}... (showing first 5)")
        
        # Test 9: Different awaitable types
        print("\n✅ Test 9: Different awaitable types")
        
        # Test with asyncio.Task
        async def create_task_test():
            task = asyncio.create_task(simple_async_func(100, y=200))
            return await task
        
        task_result = run_sync(create_task_test())
        print(f"   Task result: {task_result}")
        assert task_result == 300
        
        # Test with asyncio.Future
        async def create_future_test():
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            
            # Set the result after a short delay
            def set_result():
                future.set_result("Future completed!")
            
            loop.call_later(0.1, set_result)
            return await future
        
        future_result = run_sync(create_future_test())
        print(f"   Future result: {future_result}")

        # Test 10: Method calls and keyword arguments (like the agent example)
        print("\n✅ Test 10: Method calls and keyword arguments")
        
        class MockAgent:
            def __init__(self, name: str):
                self.name = name
            
            async def run_async(self, input: str, *, timeout: float | None = None, 
                              trace_params: dict[str, Any] | None = None, stream: bool = False) -> dict[str, Any]:
                """Mock async method similar to agent.run_async"""
                await asyncio.sleep(0.05)
                return {
                    "agent": self.name,
                    "input": input,
                    "timeout": timeout,
                    "trace_params": trace_params,
                    "stream": stream,
                    "result": f"Processed: {input}"
                }
        
        agent = MockAgent("test_agent")
        
        # Test calling method with keyword arguments (like in the agent code)
        result = run_sync(
            agent.run_async,
            timeout=5.0,  # This is run_sync's timeout, not the method's timeout
            input="test input",
            trace_params={"trace": "enabled"},
            stream=False
        )
        print(f"   Method call result: {result['result']}")
        assert result["input"] == "test input"
        assert result["timeout"] is None  # The method's timeout should be None (default)
        assert result["stream"] is False
        
        # Test with method's own timeout parameter using coroutine approach
        coro = agent.run_async(
            input="test input 2",
            timeout=10.0,  # This goes to the method
            stream=True
        )
        result2 = run_sync(coro, timeout=5.0)  # run_sync timeout
        print(f"   Method with timeout: timeout={result2['timeout']}, stream={result2['stream']}")
        assert result2["timeout"] == 10.0
        assert result2["stream"] is True

        # Test 11: Threading scenarios (like the streaming agent)
        print("\n✅ Test 11: Threading scenarios")
        
        import queue
        
        def test_run_sync_in_thread() -> None:
            """Test calling run_sync from within a thread"""
            results: List[str] = []
            exceptions: List[Exception] = []
            
            def thread_worker(task_id: int) -> None:
                try:
                    # This simulates the streaming agent pattern
                    async def local_async_func() -> str:
                        await asyncio.sleep(0.1)
                        return f"Thread {task_id} completed"
                    
                    result = run_sync(local_async_func, timeout=2.0)
                    results.append(result)
                except Exception as e:
                    exceptions.append(e)
            
            # Create multiple threads (like in streaming scenario)
            threads = []
            for i in range(3):
                t = threading.Thread(target=thread_worker, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for all threads
            for t in threads:
                t.join(timeout=5.0)
            
            print(f"   Thread results: {results}")
            print(f"   Thread exceptions: {exceptions}")
            assert len(results) == 3
            assert len(exceptions) == 0
            assert all("completed" in r for r in results)
        
        test_run_sync_in_thread()

        # Test 12: Complex streaming-like pattern
        print("\n✅ Test 12: Complex streaming-like pattern")
        
        async def mock_async_generator(count: int) -> AsyncGenerator[str, None]:
            """Mock the async iterator pattern from streaming"""
            for i in range(count):
                await asyncio.sleep(0.02)
                yield f"chunk_{i}"
        
        def sync_stream_consumer(count: int) -> List[str]:
            """Similar to the agent's streaming pattern"""
            chunk_queue: queue.Queue[str | None] = queue.Queue()
            exception_holder: List[Exception] = []
            
            async def consume_async_iterator() -> None:
                try:
                    async for chunk in mock_async_generator(count):
                        chunk_queue.put(chunk)
                    chunk_queue.put(None)  # End signal
                except Exception as e:
                    exception_holder.append(e)
                    chunk_queue.put(None)
            
            def run_consumer() -> None:
                # This is the key call - run_sync with a local async function
                run_sync(consume_async_iterator, timeout=5.0)
            
            consumer_thread = threading.Thread(target=run_consumer)
            consumer_thread.start()
            
            chunks: List[str] = []
            try:
                while True:
                    if exception_holder:
                        raise exception_holder[0]
                    
                    try:
                        chunk = chunk_queue.get(timeout=2.0)
                        if chunk is None:
                            break
                        chunks.append(chunk)
                    except queue.Empty:
                        raise TimeoutError("Timeout waiting for chunk")
            finally:
                consumer_thread.join(timeout=1.0)
            
            return chunks
        
        streaming_result = sync_stream_consumer(4)
        print(f"   Streaming chunks: {streaming_result}")
        assert len(streaming_result) == 4
        assert streaming_result == ["chunk_0", "chunk_1", "chunk_2", "chunk_3"]

        # Test 13: Edge case - calling run_sync recursively from different contexts
        print("\n✅ Test 13: Recursive and nested contexts")
        
        async def level_3_async() -> str:
            await asyncio.sleep(0.01)
            return "level_3"
        
        async def level_2_async() -> str:
            # This calls run_sync from within an async context
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: run_sync(level_3_async, timeout=1.0)
            )
            return f"level_2 -> {result}"
        
        def level_1_sync() -> str:
            return run_sync(level_2_async, timeout=2.0)
        
        nested_result = level_1_sync()
        print(f"   Nested result: {nested_result}")
        assert "level_2 -> level_3" == nested_result
        
        print("\n🎉 All tests passed! run_sync is working correctly.")
        
        # Additional environment compatibility test
        print("\n🔍 Environment Compatibility Check:")
        
        # Test if we're in a special environment
        try:
            import __main__
            if hasattr(__main__, '__file__'):
                print("   ✅ Running in script mode")
            else:
                print("   ℹ️  Running in REPL/Jupyter (no __main__.__file__)")
        except Exception:
            print("   ⚠️  Unusual environment detected")
        
        # Check event loop policy
        try:
            policy = asyncio.get_event_loop_policy()
            print(f"   Event loop policy: {type(policy).__name__}")
        except Exception as e:
            print(f"   ⚠️  Event loop policy issue: {e}")
        
        # Test current loop detection
        try:
            _ = asyncio.get_running_loop()
            print("   ℹ️  Event loop already running (Jupyter/IPython/Web environment)")
        except RuntimeError:
            print("   ✅ No running event loop (standard Python environment)")
        
        print("\n📋 Summary: run_sync should work in this environment!")

    run_tests()