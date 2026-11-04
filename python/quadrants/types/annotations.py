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


class RuntimeTyped:
    """A Python dataclass argument whose member types are inferred from the arriving instance.

    Use the class itself as a kernel or function parameter annotation: ``state: qd.RuntimeTyped``.
    """

    def __getattr__(self, name: str) -> Any:
        """Allow static type checkers to accept arbitrary member access on a RuntimeTyped parameter as Any.

        Callers pass actual dataclass objects, whose members supply runtime types; this method does not infer them.
        Accessing a missing member on an instance of this annotation marker still raises AttributeError.
        """
        raise AttributeError(name)


class sparse_matrix_builder:
    pass


__all__ = ["template", "sparse_matrix_builder", "Template", "RuntimeTyped"]
