"""
handle initializatin torhc tensor for python backend
"""

from functools import partial

torch = None
try:
    import torch
except Exception:
    pass


def from_numpy(self: torch.Tensor, numpy_tensor):
    assert torch is not None
    self.copy_(torch.from_numpy(numpy_tensor))


def init_py_tensor(t: torch.Tensor) -> None:
    t.fill = t.fill_  # type: ignore
    t.from_numpy = partial(from_numpy, t)
