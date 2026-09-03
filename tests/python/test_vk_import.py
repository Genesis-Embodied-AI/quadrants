from __future__ import annotations

import os

import pytest

import quadrants as qd
from quadrants.interop import VkImport


def valid_kwargs(handle: int) -> dict:
    return {
        "handle": handle,
        "allocation_size": 16,
        "offset": 0,
        "size": 12,
        "shape": (1, 3),
        "dtype": "float32",
    }


@pytest.mark.parametrize(
    ("field", "value", "error", "match"),
    [
        ("handle", -1, ValueError, "invalid Linux opaque FD"),
        ("allocation_size", 8, ValueError, "exceeds allocation"),
        ("offset", -1, ValueError, "exceeds allocation"),
        ("size", 8, ValueError, "requires 12 bytes"),
        ("shape", (), ValueError, "invalid VkImport shape"),
        ("dtype", "complex64", TypeError, "unsupported VkImport dtype"),
    ],
)
def test_vk_import_validates_metadata_before_consuming_fd(field, value, error, match):
    read_fd, write_fd = os.pipe()
    kwargs = valid_kwargs(read_fd)
    kwargs[field] = value
    try:
        with pytest.raises(error, match=match):
            VkImport(**kwargs)
        os.fstat(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_vk_import_requires_initialized_gpu_backend(monkeypatch):
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(qd, "cfg", None)
    try:
        with pytest.raises(RuntimeError, match=r"qd\.init"):
            VkImport(**valid_kwargs(read_fd))
        os.fstat(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_vk_import_is_exported_from_interop_module():
    from quadrants import interop

    assert interop.VkImport is VkImport
    assert "VkImport" in interop.__all__
