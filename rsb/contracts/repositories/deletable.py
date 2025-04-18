from __future__ import annotations

import abc


class Deletable[I = str](abc.ABC):
    @abc.abstractmethod
    def delete(self, uid: I) -> None: ...


class AsyncDeletable[I = str](abc.ABC):
    @abc.abstractmethod
    async def delete(self, uid: I) -> None: ...
