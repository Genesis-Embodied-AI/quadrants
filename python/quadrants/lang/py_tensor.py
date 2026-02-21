"""
handle initializatin torhc tensor for python backend
"""

from functools import partial

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


def init_py_tensor(t: torch.Tensor) -> None:
    t.fill = t.fill_  # type: ignore
    t.from_numpy = partial(from_numpy, t)
    t.to_numpy = t.numpy
    t.old_getter = t.__getitem__
    # t.__getitem__ = None
    # t.__getitem__ = partial(getitem_wrapper, t)


class MyTorchTensor(torch.Tensor):
    @classmethod
    def zeros(cls, *args, **kwargs):
        print("args", args, "kwargs", kwargs)
        return cls(torch.zeros(*args, **kwargs))

    def __getitem__(self, key):
        if key == 0 and self.shape == ():
            return self.item()
        if isinstance(key, list) and len(key) == 1:
            key = key[0]
        return super().__getitem__(key)

    def __setitem__(self, key, v):
        print("setitem", key, v)
        print("type(v)", type(v))
        # if isinstance(v, np.NDArray):
        if type(v) is np.ndarray:
            v = torch.from_numpy(v)
        elif type(v) is np.float32:
            v = v.item()
        super().__setitem__(key, v)


def create_tensor(shape, dtype):
    res = MyTorchTensor.zeros(size=shape, dtype=dtype)
    init_py_tensor(res)
    return res
