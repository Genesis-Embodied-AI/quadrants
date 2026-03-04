from typing import Any, Generic, TypeVar

T = TypeVar("T")

class Template(Generic[T]):
    element_type: type[T]
    ndim: int | None
    def __init__(self, element_type: type[T] = ..., ndim: int | None = ...) -> None: ...
    def __getitem__(self, i: Any) -> T: ...

template = Template

class sparse_matrix_builder: ...
