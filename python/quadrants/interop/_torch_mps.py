"""Extract PyTorch MPS's Metal command queue for shared-queue interop with Quadrants.

Usage::

    import quadrants as qd
    from quadrants.interop import get_mps_command_queue

    queue = get_mps_command_queue()
    qd.init(arch=qd.metal, external_metal_command_queue=queue, external_metal_command_queue_is_torch_queue=True)
"""

from __future__ import annotations

import ctypes
import os
import sys


def get_mps_command_queue() -> int:
    """Extract PyTorch MPS's ``MTLCommandQueue*`` as a raw pointer (Python int).

    Returns the pointer value on success, or 0 if extraction fails (e.g. PyTorch not installed, non-macOS platform,
    or unsupported PyTorch build).

    The returned pointer is borrowed — it remains valid for the lifetime of the PyTorch MPS runtime (i.e. until process
    exit or an explicit ``torch._C._mps_emptyCache()``).
    """
    if sys.platform != "darwin":
        return 0

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return 0

    if torch.__file__ is None:
        return 0

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return 0

    # Ensure MPS runtime is initialised (creates the device and default stream).
    torch.zeros(1, device="mps")

    try:
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cpu.dylib")
        handle = ctypes.CDLL(torch_lib)._handle

        libdl = ctypes.CDLL(None)
        dlsym = libdl.dlsym
        dlsym.restype = ctypes.c_void_p
        dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        stream_fn = dlsym(handle, b"_ZN2at3mps19getDefaultMPSStreamEv")
        if not stream_fn:
            return 0
        stream_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p)(stream_fn)()

        cb_fn = dlsym(handle, b"_ZN2at3mps9MPSStream13commandBufferEv")
        if not cb_fn:
            return 0
        cb_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(cb_fn)(stream_ptr)

        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        sel_reg = objc.sel_registerName
        sel_reg.restype = ctypes.c_void_p
        sel_reg.argtypes = [ctypes.c_char_p]
        msg_send = objc.objc_msgSend
        msg_send.restype = ctypes.c_void_p
        msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        queue_ptr = msg_send(cb_ptr, sel_reg(b"commandQueue"))
        return queue_ptr or 0
    except (OSError, AttributeError):
        return 0
