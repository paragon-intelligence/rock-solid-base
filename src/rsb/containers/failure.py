from src.infrastructure.containers.result import Result


class Failure[T_Failure: Exception](Result[Ellipsis, T_Failure]): ...
