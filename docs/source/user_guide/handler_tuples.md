# Tuples of handlers

A kernel can accept a Python tuple of handlers whose `__call__` methods are [`@qd.func`](sub_functions.md) functions. Quadrants discovers the functions and their captured data from the arriving objects. Python annotations such as `Protocol`, `Callable`, and `tuple[Handler, ...]` do not select the implementation.

```python
import dataclasses
import quadrants as qd

@dataclasses.dataclass(frozen=True)
class Data:
    increments: qd.Tensor

@qd.func
def add(i: int, value: float, data: Data) -> float:
    return value + data.increments[i]

@qd.data_oriented
class Bound:
    def __init__(self, func, *data):
        self.func = func
        self.data = data

    @qd.func
    def __call__(self, i: int, value: float) -> float:
        return self.func(i, value, *self.data)

@qd.kernel
def apply(values: qd.Tensor, handlers: tuple):
    for i in range(values.shape[0]):
        value = values[i]
        for handler in handlers:
            value = handler(i, value)
        values[i] = value
```

Pass a tuple such as `(Bound(add, first_data), Bound(add, second_data))` to `apply`. The handler loop is unrolled: the compiler emits each handler body in order, while the outer element loop runs on the selected backend. An empty tuple applies no handlers. Nested tuple loops and tuples of bare `qd.func` functions are also supported. Ordinary range and field loops retain their existing behavior.

The tuple determines its argument handling at runtime, even when its kernel parameter has another supported annotation. `tuple`, `tuple[Handler, ...]`, `qd.Template`, and `qd.RuntimeTyped` are accepted spellings. This does not enable arbitrary Python objects or callbacks on the device: callable objects must expose a `qd.func __call__` method.

## Captured values and changes between launches

Captured data may contain tuples, Python dataclasses, data-oriented objects, Quadrants ndarrays, Fields, and `qd.Tensor` wrappers around those arrays or Fields. Ordinary scalar dataclass members are runtime arguments, using the same [runtime type inference](compound_types.md#runtime-member-types) as direct dataclass arguments. `Final` members remain compile-time constants. Data-oriented configuration members keep their existing template semantics.

Changing tensor contents or an ordinary scalar dataclass value updates the next launch without requiring another compilation. Rebinding a tensor with the same element type, rank, and layout also reuses compiled code. Changing a selected function, handler order, tuple structure, captured runtime type, or compile-time constant selects a matching specialization. Equivalent fresh handler objects can reuse compiled code; object identity alone is not the specialization key. Sharing captured objects is tracked because it affects which kernel arguments can be shared.

`Bound.func` and `Bound.data` may be reassigned. Quadrants checks the captured structure on each launch, so this flexibility has Python dispatch cost proportional to the captured structure. Keep handler sequences small when launch overhead matters. Cycles and unsupported capture types produce an error identifying the capture path. Mutation during an active launch is not supported.

## Caching and limitations

Compiled specializations are reused within the current runtime. Handler tuple arguments currently disable [offline caching](fastcache.md): the disk cache does not yet describe selected functions and indexed capture paths completely. Other kernel arguments retain their existing caching behavior. Captured external NumPy/PyTorch arrays, arbitrary lists/dictionaries, and opaque application objects are not supported by this tuple path; use Quadrants tensors and supported containers.
