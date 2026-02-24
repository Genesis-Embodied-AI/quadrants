"""
handle initializatin torhc tensor for python backend
"""

from functools import partial
from typing import Callable

import numpy as np

torch = None
try:
    import torch
except Exception:
    pass


def from_numpy(self: torch.Tensor, numpy_tensor):
    assert torch is not None
    self.copy_(torch.from_numpy(numpy_tensor))


def getitem_wrapper(self: torch.Tensor, key):
    print("get_item wrapper", self, key)
    if key == 0 and len(self.shape) == 0:
        return self.item()
    return self.super().__getitem__(key)


def init_py_tensor(t: torch.Tensor) -> None:
    t.fill = t.fill_  # type: ignore
    t.from_numpy = partial(from_numpy, t)
    t.to_numpy = t.numpy
    t.old_getter = t.__getitem__

    from . import matrix_ops
    for k, v in matrix_ops.__dict__.items():
        if not k.startswith("_") and isinstance(v, Callable):
            print(k, v)
            setattr(t, k, partial(v, t))
    # t.__getitem__ = None
    # t.__getitem__ = partial(getitem_wrapper, t)


class MyTorchTensor(torch.Tensor):
    @classmethod
    def zeros(cls, *args, **kwargs):
        print("args", args, "kwargs", kwargs)
        return cls(torch.zeros(*args, **kwargs))

    def fill(self, v):
        super().fill_(v)

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
        # else:
        try:
            super().__setitem__(key, v)
        except Exception as e:
            print("setitem", key, v)
            print("type(v)", type(v))
            raise e
    
    def get_shape(self):
        return self.shape
        
    def outer_product(self, other):
        # from . import matrix_ops
        # return matrix_ops.outer_product(self, other)
        shape_x = self.shape
        shape_y = other.shape
        vec_x = self
        vec_y = other
        print('self.shape', self.shape, 'other.shape', other.shape)
        res = [[(vec_x[i] * vec_y[j]).item() for j in range(shape_y[0])] for i in range(shape_x[0])]
        print('res', res)
        return MyTorchTensor(res)


def create_tensor(shape, dtype):
    res = MyTorchTensor.zeros(size=shape, dtype=dtype)
    init_py_tensor(res)
    return res
