"""Linux Vulkan opaque-FD import for API review and validation.

The implementation uses ctypes and device-wide synchronization. It demonstrates the constructor, ownership, DLPack, and
lifecycle contract; the synchronization FIXME is documented in the interop user guide.
"""

from __future__ import annotations

import ctypes
import os
import sys
from math import prod
from typing import Any


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DL_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DL_DELETER),
]


@_DL_DELETER
def _noop_dlpack_deleter(_managed: ctypes.POINTER(_DLManagedTensor)) -> None:
    # The Genesis plugin retains VkImport beside its tensor view.
    return


class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("reserved", ctypes.c_void_p * 2)]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


_DTYPE: dict[str, tuple[int, int]] = {
    "int8": (0, 8),
    "int16": (0, 16),
    "int32": (0, 32),
    "int64": (0, 64),
    "uint8": (1, 8),
    "uint16": (1, 16),
    "uint32": (1, 32),
    "uint64": (1, 64),
    "float16": (2, 16),
    "float32": (2, 32),
    "float64": (2, 64),
}


class VkImport:
    """Import a Linux Vulkan opaque FD into the active CUDA or AMDGPU backend.

    The constructor is transactional: it imports a duplicate of ``handle`` and
    consumes the caller's original FD only after import and mapping both succeed.
    """

    def __init__(
        self,
        *,
        handle: int,
        allocation_size: int,
        offset: int,
        size: int,
        shape: tuple[int, ...],
        dtype: str,
    ) -> None:
        self._runtime: Any = None
        self._external = ctypes.c_void_p()
        self._ptr = ctypes.c_void_p()
        self._closed = False
        self._backend = ""
        self._device_id = 0

        if sys.platform != "linux":
            raise NotImplementedError("VkImport prototype currently supports Linux opaque FDs only")

        dtype = str(dtype).removeprefix("torch.")
        if dtype not in _DTYPE:
            raise TypeError(f"unsupported VkImport dtype: {dtype}")
        shape = tuple(int(extent) for extent in shape)
        if not shape or any(extent <= 0 for extent in shape):
            raise ValueError(f"invalid VkImport shape: {shape}")
        allocation_size = int(allocation_size)
        offset = int(offset)
        size = int(size)
        if allocation_size <= 0 or offset < 0 or size <= 0 or offset + size > allocation_size:
            raise ValueError("logical external-memory range exceeds allocation")
        _, bits = _DTYPE[dtype]
        expected_size = prod(shape) * (bits // 8)
        if expected_size != size:
            raise ValueError(f"shape/dtype requires {expected_size} bytes, got {size}")
        if not (0 <= int(handle) <= 0x7FFFFFFF):
            raise ValueError(f"invalid Linux opaque FD: {handle}")

        import quadrants as qd  # Imported lazily to avoid package initialization cycles.

        if qd.cfg is None:
            raise RuntimeError("VkImport requires qd.init(arch=qd.cuda or qd.amdgpu) first")
        arch = qd.cfg.arch
        if arch == qd.cuda:
            self._initialize_cuda(int(handle), allocation_size, offset, size)
        elif arch == qd.amdgpu:
            self._initialize_hip(int(handle), allocation_size, offset, size)
        else:
            raise RuntimeError(f"VkImport requires the CUDA or AMDGPU backend, got {arch}")

        strides: list[int] = []
        stride = 1
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= extent
        self._shape_array = (ctypes.c_int64 * len(shape))(*shape)
        self._strides_array = (ctypes.c_int64 * len(shape))(*strides)
        code, bits = _DTYPE[dtype]
        device_type = 2 if self._backend == "cuda" else 10
        self._managed = _DLManagedTensor(
            _DLTensor(
                self._ptr,
                _DLDevice(device_type, self._device_id),
                len(shape),
                _DLDataType(code, bits, 1),
                self._shape_array,
                self._strides_array,
                0,
            ),
            None,
            _noop_dlpack_deleter,
        )

    @staticmethod
    def _close_if_open(fd: int) -> None:
        try:
            os.close(fd)
        except OSError:
            pass

    def _initialize_cuda(self, handle: int, allocation_size: int, offset: int, size: int) -> None:
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGet.restype = ctypes.c_int
        retain = getattr(cuda, "cuDevicePrimaryCtxRetain_v2", cuda.cuDevicePrimaryCtxRetain)
        retain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        retain.restype = ctypes.c_int
        cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cuda.cuCtxSetCurrent.restype = ctypes.c_int
        cuda.cuCtxSynchronize.argtypes = []
        cuda.cuCtxSynchronize.restype = ctypes.c_int
        cuda.cuImportExternalMemory.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalMemoryHandleDesc),
        ]
        cuda.cuImportExternalMemory.restype = ctypes.c_int
        cuda.cuExternalMemoryGetMappedBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.POINTER(_ExternalMemoryBufferDesc),
        ]
        cuda.cuExternalMemoryGetMappedBuffer.restype = ctypes.c_int
        cuda.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        cuda.cuDestroyExternalMemory.restype = ctypes.c_int
        free = getattr(cuda, "cuMemFree_v2", cuda.cuMemFree)
        free.argtypes = [ctypes.c_uint64]
        free.restype = ctypes.c_int
        self._free_mapped = free

        self._check(cuda.cuInit(0), "cuInit")
        # Quadrants CUDA uses device zero in the current single-device test setup.
        device = ctypes.c_int()
        self._check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        context = ctypes.c_void_p()
        self._check(retain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
        self._check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")

        import_fd = os.dup(handle)
        hd = _ExternalMemoryHandleDesc()
        hd.type = 1  # CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
        hd.handle.fd = import_fd
        hd.size = allocation_size
        rc = cuda.cuImportExternalMemory(ctypes.byref(self._external), ctypes.byref(hd))
        if rc:
            self._close_if_open(import_fd)
            raise RuntimeError(f"cuImportExternalMemory={rc}; Vulkan and CUDA must select the same physical GPU")
        bd = _ExternalMemoryBufferDesc()
        bd.offset = offset
        bd.size = size
        ptr = ctypes.c_uint64()
        rc = cuda.cuExternalMemoryGetMappedBuffer(ctypes.byref(ptr), self._external, ctypes.byref(bd))
        if rc:
            cuda.cuDestroyExternalMemory(self._external)
            self._external = ctypes.c_void_p()
            raise RuntimeError(f"cuExternalMemoryGetMappedBuffer={rc}")

        os.close(handle)
        self._runtime = cuda
        self._ptr = ctypes.c_void_p(ptr.value)
        self._device_id = int(device.value)
        self._backend = "cuda"

    def _initialize_hip(self, handle: int, allocation_size: int, offset: int, size: int) -> None:
        hip = ctypes.CDLL("libamdhip64.so")
        hip.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDevice.restype = ctypes.c_int
        hip.hipDeviceSynchronize.argtypes = []
        hip.hipDeviceSynchronize.restype = ctypes.c_int
        hip.hipImportExternalMemory.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_ExternalMemoryHandleDesc),
        ]
        hip.hipImportExternalMemory.restype = ctypes.c_int
        hip.hipExternalMemoryGetMappedBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(_ExternalMemoryBufferDesc),
        ]
        hip.hipExternalMemoryGetMappedBuffer.restype = ctypes.c_int
        hip.hipDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        hip.hipDestroyExternalMemory.restype = ctypes.c_int
        hip.hipFree.argtypes = [ctypes.c_void_p]
        hip.hipFree.restype = ctypes.c_int
        self._free_mapped = hip.hipFree

        device = ctypes.c_int()
        self._check(hip.hipGetDevice(ctypes.byref(device)), "hipGetDevice")
        import_fd = os.dup(handle)
        hd = _ExternalMemoryHandleDesc()
        hd.type = 1  # hipExternalMemoryHandleTypeOpaqueFd
        hd.handle.fd = import_fd
        hd.size = allocation_size
        rc = hip.hipImportExternalMemory(ctypes.byref(self._external), ctypes.byref(hd))
        if rc:
            self._close_if_open(import_fd)
            raise RuntimeError(f"hipImportExternalMemory={rc}; Vulkan and HIP must select the same physical GPU")
        bd = _ExternalMemoryBufferDesc()
        bd.offset = offset
        bd.size = size
        ptr = ctypes.c_void_p()
        rc = hip.hipExternalMemoryGetMappedBuffer(ctypes.byref(ptr), self._external, ctypes.byref(bd))
        if rc:
            hip.hipDestroyExternalMemory(self._external)
            self._external = ctypes.c_void_p()
            raise RuntimeError(f"hipExternalMemoryGetMappedBuffer={rc}")

        os.close(handle)
        self._runtime = hip
        self._ptr = ptr
        self._device_id = int(device.value)
        self._backend = "hip"

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code:
            raise RuntimeError(f"{operation} failed with error {code}")

    def __dlpack_device__(self) -> tuple[int, int]:
        return (2 if self._backend == "cuda" else 10, self._device_id)

    def __dlpack__(self, stream: int | None = None, max_version: tuple[int, int] | None = None):  # noqa: ARG002
        if self._closed:
            raise RuntimeError("VkImport is closed")
        capsule_new = ctypes.pythonapi.PyCapsule_New
        capsule_new.restype = ctypes.py_object
        capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return capsule_new(ctypes.addressof(self._managed), b"dltensor", None)

    def release_to_vulkan(self) -> None:
        """Complete compute writes before Vulkan consumes the allocation."""
        if self._backend == "cuda":
            self._check(self._runtime.cuCtxSynchronize(), "cuCtxSynchronize")
        else:
            self._check(self._runtime.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def acquire_from_vulkan(self) -> None:
        """Wait before compute consumes the Vulkan-written allocation."""
        # Nyx currently flushes its Vulkan queue before returning. Synchronizing
        # the active compute device keeps this prototype's ownership boundary explicit.
        self.release_to_vulkan()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._runtime is None:
            return
        if self._ptr.value:
            self._free_mapped(self._ptr if self._backend == "hip" else int(self._ptr.value))
            self._ptr = ctypes.c_void_p()
        if self._external.value:
            (
                self._runtime.cuDestroyExternalMemory(self._external)
                if self._backend == "cuda"
                else self._runtime.hipDestroyExternalMemory(self._external)
            )
            self._external = ctypes.c_void_p()

    def __enter__(self) -> "VkImport":
        if self._closed:
            raise RuntimeError("VkImport is closed")
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
