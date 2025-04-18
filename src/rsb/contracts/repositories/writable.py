from __future__ import annotations

import abc


class Writable[T](abc.ABC):
    @abc.abstractmethod
    def write(self, e: T) -> None: ...


class AsyncWritable[T](abc.ABC):
    @abc.abstractmethod
    async def write(self, e: T) -> None: ...
