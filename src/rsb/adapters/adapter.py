from __future__ import annotations

import abc


class Adapter[F, T](abc.ABC):
    @abc.abstractmethod
    def adapt(self, _f: F, /) -> T: ...
