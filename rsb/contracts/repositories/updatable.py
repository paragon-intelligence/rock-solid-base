from __future__ import annotations

import abc
from typing import Any


class Updatable[T = Any, I = str](abc.ABC):
    def update(self, uid: I, new: T) -> None: ...


class AsyncUpdatable[T = Any, I = str](abc.ABC):
    async def update(self, uid: I, new: T) -> None: ...
