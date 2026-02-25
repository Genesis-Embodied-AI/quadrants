from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Template(Generic[T]):
    """Type annotation for template kernel parameter.
    Useful for passing parameters to kernels by reference.

    See also https://docs.taichi-lang.org/docs/meta.

    Args:
        tensor (Any): unused
        dim (Any): unused

    Example::

        >>> a = 1
        >>>
        >>> @qd.kernel
        >>> def test():
        >>>     print(a)
        >>>
        >>> @qd.kernel
        >>> def test_template(a: qd.template()):
        >>>     print(a)
        >>>
        >>> test(a)  # will print 1
        >>> test_template(a)  # will also print 1
        >>> a = 2
        >>> test(a)  # will still print 1
        >>> test_template(a)  # will print 2
    """

    def __init__(self, element_type: type[T] = object, ndim: int | None = None):
        self.element_type = element_type
        self.ndim = ndim

    def __getitem__(self, i: Any) -> T:
        raise NotImplementedError()


template = Template
"""Alias for :class:`~quadrants.types.annotations.Template`.
"""


class sparse_matrix_builder:
    pass


__all__ = ["template", "sparse_matrix_builder", "Template"]
