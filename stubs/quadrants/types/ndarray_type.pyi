from typing import Any

class NdarrayType:
    dtype: Any
    ndim: int | None
    needs_grad: bool | None
    boundary: int

    def __init__(
        self,
        dtype: Any = ...,
        ndim: int | None = ...,
        element_dim: int | None = ...,
        element_shape: tuple[int, ...] | None = ...,
        field_dim: int | None = ...,
        needs_grad: bool | None = ...,
        boundary: str = ...,
    ) -> None: ...
    @classmethod
    def __class_getitem__(cls, args: Any) -> type[NdarrayType]: ...
    def __getitem__(self, i: Any) -> Any: ...
    def __setitem__(self, i: Any, v: Any) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

ndarray = NdarrayType
NDArray = NdarrayType
