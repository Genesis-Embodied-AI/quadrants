"""
Torch tensor wrapper for the python backend.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch


class PyTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        has_foreign = any(not issubclass(t, PyTensor) and t is not torch.Tensor for t in types)
        if has_foreign:

            def unwrap(o):
                return torch.Tensor._make_subclass(torch.Tensor, o) if isinstance(o, PyTensor) else o

            args = torch.utils._pytree.tree_map(unwrap, args)  # type: ignore[attr-defined]
            kwargs = torch.utils._pytree.tree_map(unwrap, kwargs)  # type: ignore[attr-defined]
            return func(*args, **kwargs)
        return super().__torch_function__(func, types, args, kwargs)

    @property
    def shape(self):  # type: ignore[override]
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
        return super()[0]

    @property
    def y(self):
        return super()[1]

    @property
    def z(self):
        return super()[2]

    @staticmethod
    def _unpack_key(key):
        """Unpack PyTensor indices into plain ints for multi-dim indexing."""
        if isinstance(key, PyTensor) and len(key.size()) == 1:
            return tuple(int(key[i]) for i in range(key.size()[0]))
        if isinstance(key, list) and len(key) == 1:
            return key[0]
        if isinstance(key, tuple):
            if any(isinstance(k, torch.Tensor) for k in key):
                return tuple(
                    int(k) if isinstance(k, torch.Tensor) and k.ndim == 0 else k  # type: ignore[union-attr]
                    for k in key
                )
        return key

    def __iter__(self):  # type: ignore[override]
        for i in range(self.size()[0]):
            val = super().__getitem__(i)
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                item = val.item()
                if isinstance(item, float) and item.is_integer():
                    item = int(item)
                yield item
            else:
                yield val

    def __getitem__(self, key):  # type: ignore[override]
        key = PyTensor._unpack_key(key)
        # Scalar fields (shape ()) are still indexed with field[0] in kernels;
        # torch would raise IndexError on a 0-d tensor, so return the scalar.
        if not isinstance(key, tuple) and key == 0 and self.size() == ():
            return self.item()
        return super().__getitem__(key)

    def __setitem__(self, key, v):
        key = PyTensor._unpack_key(key)
        if type(v) is np.ndarray:
            v = torch.from_numpy(v)
        elif isinstance(v, np.generic):
            v = v.item()
        try:
            super().__setitem__(key, v)
        except Exception as e:
            raise type(e)(f"PyTensor.__setitem__({key!r}, {v!r} (type={type(v).__name__}))") from e

    def transpose(self, *args):  # type: ignore[override]
        if len(args) == 0:
            ndim = len(self.size())
            assert ndim == 2, f"transpose() with no args requires a 2D tensor, got {ndim}D"
            return super().transpose(0, 1)
        return super().transpose(*args)

    def get_shape(self):
        return self.size()

    def _setup_views(self, batch_ndim):
        """Set _tc/_T_tc/_np/_T_np instance attributes matching the qd.Field/Ndarray convention.

        Not in __init__/__new__ because: (1) torch.Tensor subclass construction
        goes through C++ __new__, so __init__ is not reliably called by torch
        internals; (2) PyTensor is also used as a plain wrapper (e.g. in
        Matrix.__new__) where these attributes are not needed; (3) batch_ndim
        is a Quadrants concept with no analogue in torch's constructor signature.
        """
        self._tc = self
        self._T_tc = self.movedim(batch_ndim - 1, 0) if batch_ndim > 1 else self
        self._np = self.numpy()
        self._T_np = self._T_tc.numpy()
        self._batch_shape = self.size()[:batch_ndim]
        self.grad = None
        self.dual = None

    def __getattr__(self, name):
        from . import matrix_ops  # pylint: disable=C0415

        fn = getattr(matrix_ops, name, None)
        if fn is not None and callable(fn):
            return partial(fn, self)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def create(cls, shape, dtype, batch_ndim=None):
        """Factory for Quadrants field/ndarray tensors with view attributes."""
        res = cls.zeros(size=shape, dtype=dtype)
        if batch_ndim is None:
            batch_ndim = len(shape)
        res._setup_views(batch_ndim)
        return res


# Keep module-level alias for existing callers.
create_tensor = PyTensor.create
