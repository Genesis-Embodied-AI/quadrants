"""
Torch tensor wrapper for the python backend.
"""

from functools import partial

import numpy as np

torch = None
try:
    import torch
except Exception:
    pass


class MyTorchTensor(torch.Tensor):
    # Expose zero-copy torch views of self so that downstream code
    # (e.g. Genesis qd_to_torch) can access the data without conversion.
    _tc = property(lambda self: self)
    _T_tc = property(lambda self: self)
    _np = property(lambda self: self.numpy())
    _T_np = property(lambda self: self.numpy())

    @classmethod
    def zeros(cls, *args, **kwargs):
        return cls(torch.zeros(*args, **kwargs))

    def fill(self, v):
        super().fill_(v)

    def from_numpy(self, numpy_tensor):
        self.copy_(torch.from_numpy(numpy_tensor))

    def to_numpy(self):
        return self.numpy()

    @property
    def x(self):
        return super().__getitem__(0)

    @property
    def y(self):
        return super().__getitem__(1)

    @property
    def z(self):
        return super().__getitem__(2)

    def __getitem__(self, key):
        if key == 0 and self.shape == ():
            return self.item()
        if isinstance(key, list) and len(key) == 1:
            key = key[0]
        return super().__getitem__(key)

    def __setitem__(self, key, v):
        if type(v) is np.ndarray:
            v = torch.from_numpy(v)
        elif isinstance(v, np.generic):
            v = v.item()
        try:
            super().__setitem__(key, v)
        except Exception as e:
            raise type(e)(f"MyTorchTensor.__setitem__({key!r}, {v!r} (type={type(v).__name__}))") from e

    def get_shape(self):
        return self.shape

    def outer_product(self, other):
        shape_x = self.shape
        shape_y = other.shape
        res = [[(self[i] * other[j]).item() for j in range(shape_y[0])] for i in range(shape_x[0])]
        return MyTorchTensor(res)

    def __getattr__(self, name):
        from . import matrix_ops

        fn = getattr(matrix_ops, name, None)
        if fn is not None and callable(fn):
            return partial(fn, self)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def create_tensor(shape, dtype):
    return MyTorchTensor.zeros(size=shape, dtype=dtype)
