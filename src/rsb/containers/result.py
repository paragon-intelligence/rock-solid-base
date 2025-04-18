from __future__ import annotations
import abc


class Result[T_Success, T_Failure: Exception](abc.ABC):
    value: T_Success | T_Failure

    def __init__(self, value: T_Success | T_Failure) -> None:
        self.value = value

    def raise_if_failure(self) -> None:
        if isinstance(self.value, Exception):
            raise self.value
