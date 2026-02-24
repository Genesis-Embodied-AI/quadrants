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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        has_foreign = any(not issubclass(t, MyTorchTensor) and t is not torch.Tensor for t in types)
        if has_foreign:

            def unwrap(o):
                return torch.Tensor._make_subclass(torch.Tensor, o) if isinstance(o, MyTorchTensor) else o

            args = torch.utils._pytree.tree_map(unwrap, args)
            kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)
            return func(*args, **kwargs)
        return super().__torch_function__(func, types, args, kwargs)

    @property
    def shape(self):
        real = self.size()
        batch = getattr(self, "_batch_shape", None)
        return batch if batch is not None else real

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

    @staticmethod
    def _unpack_key(key):
        """Unpack MyTorchTensor indices into plain ints for multi-dim indexing."""
        if isinstance(key, MyTorchTensor) and len(key.size()) == 1:
            return tuple(int(key[i]) for i in range(key.size()[0]))
        if isinstance(key, list) and len(key) == 1:
            return key[0]
        if isinstance(key, tuple):
            if any(isinstance(k, torch.Tensor) for k in key):
                return tuple(int(k) if isinstance(k, torch.Tensor) and k.ndim == 0 else k for k in key)
        return key

    def __iter__(self):
        for i in range(self.size()[0]):
            val = super().__getitem__(i)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                item = val.item()
                if isinstance(item, float) and item.is_integer():
                    item = int(item)
                yield item
            else:
                yield val

    def __getitem__(self, key):
        key = MyTorchTensor._unpack_key(key)
        if not isinstance(key, tuple) and key == 0 and self.size() == ():
            return self.item()
        return super().__getitem__(key)

    def __setitem__(self, key, v):
        key = MyTorchTensor._unpack_key(key)
        if type(v) is np.ndarray:
            v = torch.from_numpy(v)
        elif isinstance(v, np.generic):
            v = v.item()
        try:
            super().__setitem__(key, v)
        except Exception as e:
            raise type(e)(f"MyTorchTensor.__setitem__({key!r}, {v!r} (type={type(v).__name__}))") from e

    def transpose(self, *args):
        if len(args) == 0:
            ndim = len(self.size())
            assert ndim == 2, f"transpose() with no args requires a 2D tensor, got {ndim}D"
            return super().transpose(0, 1)
        return super().transpose(*args)

    def get_shape(self):
        return self.size()

    def outer_product(self, other):
        shape_x = self.size()
        shape_y = other.size()
        res = [[(self[i] * other[j]).item() for j in range(shape_y[0])] for i in range(shape_x[0])]
        return MyTorchTensor(res)

    def __getattr__(self, name):
        from . import matrix_ops  # pylint: disable=C0415

        fn = getattr(matrix_ops, name, None)
        if fn is not None and callable(fn):
            return partial(fn, self)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def create_tensor(shape, dtype, batch_ndim=None):
    res = MyTorchTensor.zeros(size=shape, dtype=dtype)
    if batch_ndim is None:
        batch_ndim = len(shape)
    _setup_views(res, batch_ndim)
    return res


def _setup_views(tensor, batch_ndim):
    """Set _tc/_T_tc/_np/_T_np instance attributes matching the qd.Field/Ndarray convention."""
    tensor._tc = tensor
    tensor._T_tc = tensor.movedim(batch_ndim - 1, 0) if batch_ndim > 1 else tensor
    tensor._np = tensor.numpy()
    tensor._T_np = tensor._T_tc.numpy()
    tensor._batch_shape = tensor.size()[:batch_ndim]
    tensor.grad = None
    tensor.dual = None
