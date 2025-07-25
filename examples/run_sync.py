import asyncio
import random
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List

from rsb.coroutines.run_sync import run_sync

# Context variable for testing context preservation
request_id: ContextVar[str] = ContextVar("request_id", default="none")


@dataclass
class TestResult:
    thread_name: str
    has_loop: bool
    result: Any
    execution_time: float
    context_value: str
    errors: List[str]


class ComplexAsyncService:
    """Simulates a complex async service with various async operations"""

    def __init__(self):
        self.db_pool = asyncio.Queue(maxsize=5)
        self.cache = {}
        self.lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the service with some async setup"""
        for i in range(5):
            await self.db_pool.put(f"connection_{i}")

    async def fetch_data(self, key: str, delay: float = 0.1) -> Dict[str, Any]:
        """Simulate fetching data with database connection pooling"""
        async with self.lock:
            if key in self.cache:
                return self.cache[key]

        # Get a connection from pool
        connection = await asyncio.wait_for(self.db_pool.get(), timeout=2.0)

        try:
            # Simulate database query
            await asyncio.sleep(delay)
            data = {
                "key": key,
                "value": random.randint(1, 1000),
                "timestamp": time.time(),
                "connection": connection,
                "context": request_id.get(),
            }

            # Cache the result
            async with self.lock:
                self.cache[key] = data

            return data
        finally:
            # Return connection to pool
            await self.db_pool.put(connection)

    async def batch_process(self, keys: List[str]) -> List[Dict[str, Any]]:
        """Process multiple keys concurrently"""
        tasks = [self.fetch_data(key, random.uniform(0.05, 0.2)) for key in keys]
        return await asyncio.gather(*tasks)

    async def streaming_data(self, count: int):
        """Async generator that yields data"""
        for i in range(count):
            await asyncio.sleep(0.01)
            yield {"item": i, "context": request_id.get()}

    async def complex_nested_operation(self, depth: int) -> Dict[str, Any]:
        """Recursively nested async operations"""
        if depth <= 0:
            return {"depth": 0, "context": request_id.get()}

        # Parallel nested calls
        tasks = [
            self.complex_nested_operation(depth - 1),
            self.fetch_data(f"nested_{depth}"),
        ]

        results = await asyncio.gather(*tasks)

        return {
            "depth": depth,
            "nested_result": results[0],
            "data": results[1],
            "context": request_id.get(),
        }


def test_scenario(scenario_name: str, test_func, *args, **kwargs) -> TestResult:
    """Run a test scenario and collect metrics"""
    start_time = time.time()
    thread_name = threading.current_thread().name

    try:
        has_loop = asyncio.get_running_loop() is not None
    except RuntimeError:
        has_loop = False

    errors = []
    result = None

    try:
        result = test_func(*args, **kwargs)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
        result = None

    execution_time = time.time() - start_time
    context_value = request_id.get()

    return TestResult(
        thread_name=thread_name,
        has_loop=has_loop,
        result=result,
        execution_time=execution_time,
        context_value=context_value,
        errors=errors,
    )


async def complex_test_suite():
    """Main test suite that exercises run_sync in various scenarios"""

    service = ComplexAsyncService()
    await service.initialize()

    print("🧪 Starting Complex run_sync Test Suite\n")

    # Test 1: Basic functionality from async context
    print("1️⃣ Testing from async context (should use thread)...")
    request_id.set("async_context_test")

    def sync_wrapper_basic():
        return run_sync(service.fetch_data, key="test1", timeout=5.0)

    result1 = test_scenario("basic_async_context", sync_wrapper_basic)
    print(f"   Result: {result1}")
    print()

    # Test 2: Nested run_sync calls (challenging!)
    print("2️⃣ Testing nested run_sync calls...")
    request_id.set("nested_test")

    async def nested_async_operation():
        # This will call run_sync from within an async function
        def inner_sync():
            return run_sync(service.fetch_data, key="nested", timeout=3.0)

        # Call sync function from async context
        sync_result = await asyncio.get_event_loop().run_in_executor(None, inner_sync)

        # Then do more async work
        async_result = await service.fetch_data("after_sync")

        return {"sync_part": sync_result, "async_part": async_result}

    def sync_wrapper_nested():
        return run_sync(nested_async_operation, timeout=10.0)

    result2 = test_scenario("nested_calls", sync_wrapper_nested)
    print(f"   Result: {result2}")
    print()

    # Test 3: High concurrency stress test
    print("3️⃣ Testing high concurrency stress...")
    request_id.set("stress_test")

    def concurrent_stress_test():
        def single_batch():
            keys = [f"stress_{i}" for i in range(20)]
            return run_sync(service.batch_process, keys, timeout=15.0)

        # Run multiple batches concurrently from different threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(single_batch) for _ in range(3)]
            results = [f.result() for f in futures]

        return {
            "batches": len(results),
            "total_items": sum(len(batch) for batch in results),
        }

    result3 = test_scenario("stress_test", concurrent_stress_test)
    print(f"   Result: {result3}")
    print()

    # Test 4: Complex nested operations
    print("4️⃣ Testing deeply nested async operations...")
    request_id.set("deep_nested")

    def deep_nested_test():
        return run_sync(service.complex_nested_operation, depth=5, timeout=20.0)

    result4 = test_scenario("deep_nested", deep_nested_test)
    print(f"   Result: {result4}")
    print()

    # Test 5: Async generator handling
    print("5️⃣ Testing async generator consumption...")
    request_id.set("generator_test")

    async def consume_generator():
        items = []
        async for item in service.streaming_data(10):
            items.append(item)
        return items

    def generator_test():
        return run_sync(consume_generator, timeout=5.0)

    result5 = test_scenario("generator_test", generator_test)
    print(f"   Result: {result5}")
    print()

    # Test 6: Timeout scenarios
    print("6️⃣ Testing timeout handling...")
    request_id.set("timeout_test")

    async def slow_operation():
        await asyncio.sleep(3.0)  # Longer than timeout
        return "should_not_reach_here"

    def timeout_test():
        return run_sync(slow_operation, timeout=1.0)  # Short timeout

    result6 = test_scenario("timeout_test", timeout_test)
    print(f"   Result: {result6}")
    print()

    # Test 7: Exception handling in complex scenarios
    print("7️⃣ Testing exception handling...")
    request_id.set("exception_test")

    async def failing_operation():
        await service.fetch_data("step1")  # This should work
        raise ValueError("Intentional test failure")
        await service.fetch_data("step2")  # Should not reach here

    def exception_test():
        return run_sync(failing_operation, timeout=5.0)

    result7 = test_scenario("exception_test", exception_test)
    print(f"   Result: {result7}")
    print()

    # Test 8: Context variable preservation across threads
    print("8️⃣ Testing context variable preservation...")
    request_id.set("context_preservation_test")

    async def context_dependent_operation():
        # This should preserve the context from the calling thread
        data = await service.fetch_data("context_test")
        return {
            "original_context": request_id.get(),
            "data_context": data.get("context"),
            "thread": threading.current_thread().name,
        }

    def context_test():
        return run_sync(context_dependent_operation, timeout=5.0)

    result8 = test_scenario("context_test", context_test)
    print(f"   Result: {result8}")
    print()

    print("🏁 Test suite completed!")

    return [result1, result2, result3, result4, result5, result6, result7, result8]


def run_from_sync_context():
    """Test run_sync from a purely synchronous context"""
    print("\n🔄 Testing from synchronous context (no event loop)...")
    request_id.set("sync_context_test")

    service = ComplexAsyncService()

    # Initialize service synchronously
    run_sync(service.initialize, timeout=5.0)

    # Run complex operations
    batch_result = run_sync(
        service.batch_process, keys=[f"sync_{i}" for i in range(5)], timeout=10.0
    )

    print(f"   Sync context result: {len(batch_result)} items processed")
    return batch_result


def run_mixed_scenario():
    """Mix of sync and async contexts with thread switching"""
    print("\n🔀 Testing mixed sync/async scenario...")

    async def main_async():
        results = await complex_test_suite()
        return results

    # Start with sync context
    sync_results = run_from_sync_context()

    # Switch to async context
    async_results = run_sync(main_async, timeout=60.0)

    print(f"\n📊 Final Summary:")
    print(f"   Sync results: {len(sync_results)} items")
    print(f"   Async test scenarios: {len(async_results)}")
    print(
        f"   Total errors across all tests: {sum(len(r.errors) for r in async_results)}"
    )

    return {"sync": sync_results, "async": async_results}


if __name__ == "__main__":
    # Run the complete test suite
    final_results = run_mixed_scenario()

    print("\n✅ All tests completed! Check the output above for any issues.")
