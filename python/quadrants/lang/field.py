# pyright: reportPrivateImportUsage=false
# Reason: torch.zeros is public torch API, but pyright 1.1.409+ flags it as
# private because torch's stubs don't re-export it via __all__.
import ctypes
from typing import TYPE_CHECKING, cast

import quadrants.lang
from quadrants._lib import core as _qd_core
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._logging import warn
from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.lang.util import (
    in_python_scope,
    python_scope,
    to_numpy_type,
    to_pytorch_type,
)
from quadrants.types.primitive_types import (
    f32,
    f64,
    i32,
    i64,
    u1,
)


# ---------------------------------------------------------------------------
# DLPack canonical-view patch for layout-tagged fields.
#
# The C++ ``field_to_dlpack`` derives the physical-memory axis order from the SNode chain. For layout-tagged fields
# built via the new flat-rank-N allocation path (see ``lang/impl.py::_field``), the SNode chain is in natural order and
# C++ reports the *permuted physical* shape with identity strides. We patch the returned DLManagedTensor in place here
# so consumers (``torch.utils.dlpack.from_dlpack`` et al.) observe the canonical shape and matching permuted strides —
# byte-identical to the view layout-tagged ndarrays produce via the ``ndarray_to_dlpack`` ``layout=`` parameter.
#
# The struct layouts below match the DLPack v1 ABI exactly. Only the fields we need to read (``ndim``) and mutate
# (``shape``, ``strides``) are referenced, so additions to DLTensor / DLManagedTensor that land after this writes in
# newer DLPack versions won't affect correctness.
# ---------------------------------------------------------------------------
class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


def _patch_field_dlpack_canonical(capsule, layout):
    """Mutate the DLManagedTensor inside *capsule* so its shape/strides expose a canonical view of the
    permuted-physical field buffer.

    Invariants (input):
      * ``shape[i]`` = physical shape along axis ``i`` of the SNode (which is the *permuted* shape the field was
        allocated at).
      * ``strides[i]`` = physical stride along axis ``i`` (row-major over the permuted shape).

    Invariants (output):
      * ``shape[a]`` = canonical-axis-``a`` extent = input ``shape[invperm[a]]``.
      * ``strides[a]`` = canonical-axis-``a`` stride = input ``strides[invperm[a]]``.

    ``invperm`` inverts the ``_qd_layout`` permutation by the convention that ``layout[p]`` is the canonical axis at
    physical nesting position ``p``, so ``invperm[canonical_axis] = physical_axis``.
    """
    ndim = len(layout)
    if ndim == 0:
        return
    if tuple(layout) == tuple(range(ndim)):
        return  # identity layout — nothing to do
    _PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
    _PyCapsule_GetPointer.restype = ctypes.c_void_p
    _PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    raw = _PyCapsule_GetPointer(capsule, b"dltensor")
    if not raw:
        raise RuntimeError("field_to_dlpack returned a capsule without the expected 'dltensor' name")
    mt = ctypes.cast(raw, ctypes.POINTER(_DLManagedTensor)).contents
    t = mt.dl_tensor
    if t.ndim < ndim:
        raise RuntimeError(f"field_to_dlpack returned ndim={t.ndim} but layout has rank {ndim}; cannot patch")
    # Only the first ``ndim`` axes carry the canonical permutation; any trailing axes (element dims for VectorField /
    # MatrixField) are already in innermost identity position and must not be permuted.
    shape_phys = [int(t.shape[i]) for i in range(ndim)]
    strides_phys = [int(t.strides[i]) for i in range(ndim)]
    invperm = [0] * ndim
    for p, a in enumerate(layout):
        invperm[a] = p
    for a in range(ndim):
        t.shape[a] = shape_phys[invperm[a]]
        t.strides[a] = strides_phys[invperm[a]]


if TYPE_CHECKING:
    from quadrants.lang.expr import Expr

_ARCH_METAL = _qd_core.Arch.metal
_ARCH_VULKAN = _qd_core.Arch.vulkan
_ARCH_CPU = frozenset({_qd_core.Arch.x64, _qd_core.Arch.arm64})

_DLPACK_SUPPORTED_DTYPES = frozenset({f32, f64, i32, i64, u1})


def _compute_torch_mps_supports_dlpack_bytes_offset() -> bool:
    try:
        import torch  # pylint: disable=C0415
    except ImportError:
        return False
    parts = torch.__version__.replace("+", ".").split(".")[:3]
    try:
        return tuple(map(int, parts)) > (2, 9, 1)
    except ValueError:
        return False


_TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET = _compute_torch_mps_supports_dlpack_bytes_offset()


def _is_aos_struct_member(field: "Field") -> bool:
    """True when *field* is a member of a multi-member StructField with AOS layout.

    AOS struct members have interleaved memory (stride = sizeof(cell)), but the C++ DLPack export emits contiguous
    strides at the member dtype size, so a zero-copy view would silently read neighbouring members' bytes as garbage.

    SNode.place flattens vec/mat field components directly under the struct cell (no intermediate matrix SNode), so
    both ScalarField and MatrixField members sit as direct children of the struct cell dense SNode. For a ScalarField,
    num_ch != 1 means it's not a standalone field. For a MatrixField with n*m components, num_ch != n*m means either
    SOA layout (num_ch == 1, each component in its own subtree -- DLPack strides are wrong) or AOS struct member
    (num_ch > n*m, interleaved with other struct members). Only num_ch == n*m (standalone AOS) is safe for zero-copy.
    """
    try:
        from quadrants.lang.matrix import MatrixField  # pylint: disable=C0415

        parent_snode = field.parent()._snode.ptr
        if isinstance(field, MatrixField):
            return parent_snode.get_num_ch() != field.n * field.m
        return parent_snode.get_num_ch() != 1
    except Exception:
        return False


def _can_zerocopy_field(field: "Field", *, is_scalar: bool = False) -> bool:
    """Check whether zero-copy DLPack export is available for this field on the current backend."""
    dtype = field.dtype
    if dtype not in _DLPACK_SUPPORTED_DTYPES:
        return False
    arch = impl.current_cfg().arch
    if arch == _ARCH_VULKAN:
        return False
    if arch == _ARCH_METAL and not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
        return False
    if is_scalar and not field.shape:
        return False
    if _is_aos_struct_member(field):
        return False
    return True


def _try_zerocopy_torch(field: "Field", *, copy, device=None, is_scalar: bool = False):
    """Try to return a zero-copy torch tensor via DLPack.

    Only called when ``copy is False``. Returns the tensor on success. Raises ``ValueError`` when zero-copy is not
    available. Does NOT call ``torch.mps.synchronize()`` -- the caller is expected to handle MPS sync for the
    copy=True (kernel-copy) path instead.
    """
    if not _can_zerocopy_field(field, is_scalar=is_scalar):
        raise ValueError(f"Zero-copy not available for arch={impl.current_cfg().arch.name}, dtype={field.dtype}")

    import torch  # pylint: disable=C0415

    tc = torch.utils.dlpack.from_dlpack(field.to_dlpack())
    if impl.current_cfg().arch == _ARCH_METAL:
        impl.get_runtime().sync()

    if device is not None:
        requested = torch.device(device)
        type_mismatch = tc.device.type != requested.type
        index_mismatch = (
            requested.index is not None and tc.device.index is not None and tc.device.index != requested.index
        )
        if type_mismatch or index_mismatch:
            raise ValueError(
                f"copy=False is incompatible with device transfer (data on {tc.device}, requested {device})"
            )

    return tc


def _mps_sync_if_metal():
    """Call ``torch.mps.synchronize()`` when running on the Metal backend, no-op otherwise.

    Quadrants and PyTorch MPS use separate Metal command queues, so ``qd.sync()`` only guarantees Quadrants writes are
    complete. A subsequent ``.clone()`` or kernel copy is queued on the MPS stream and may execute *after* the next
    Quadrants kernel overwrites the source buffer. We must also synchronize MPS after the copy.
    """
    # FIXME: DLPack may return old values on Apple Metal if sync is not systematically called manually.
    if impl.current_cfg().arch == _ARCH_METAL:
        import torch  # pylint: disable=C0415

        torch.mps.synchronize()


class _DLPackV1Adapter:
    """Wraps a DLPack v0 PyCapsule into a v1-compatible object for ``np.from_dlpack``.

    Quadrants' C++ ``to_dlpack`` returns raw PyCapsules (v0 protocol). NumPy >= 1.23 requires the v1 protocol -- an
    object exposing ``__dlpack__`` and ``__dlpack_device__``.
    """

    __slots__ = ("_capsule",)

    def __init__(self, capsule):
        self._capsule = capsule

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        return (1, 0)  # kDLCPU = 1


def _try_zerocopy_numpy(field: "Field", *, copy, is_scalar: bool = False):
    """Try to return a zero-copy numpy array via DLPack.

    Returns the array on success, or ``None`` when zero-copy is unsupported and ``copy`` is not ``False``.
    Raises ``ValueError`` when ``copy=False`` but zero-copy is not available.
    """
    if impl.current_cfg().arch not in _ARCH_CPU:
        if copy is False:
            raise ValueError("Zero-copy numpy requires a CPU backend (numpy arrays cannot reference GPU memory)")
        return None
    if not _can_zerocopy_field(field, is_scalar=is_scalar):
        if copy is False:
            raise ValueError(f"Zero-copy not available for dtype={field.dtype}")
        return None

    import numpy as np  # pylint: disable=C0415

    arr = np.from_dlpack(_DLPackV1Adapter(field.to_dlpack()))
    if copy is True:
        arr = arr.copy()
    return arr


class Field:
    """Quadrants field class.

    A field is constructed by a list of field members.
    For example, a scalar field has 1 field member, while a 3x3 matrix field has 9 field members.
    A field member is a Python Expr wrapping a C++ FieldExpression.

    Args:
        vars (List[Expr]): Field members.
    """

    def __init__(self, _vars):
        assert all(_vars)
        self.vars = _vars
        self.host_accessors = None
        self.grad = None
        self.dual = None
        self._shape: tuple[int, ...] | None = None
        self._dtype: DataTypeCxx | None = None
        self.__name: str | None = None

    # TODO: why do we have snode and _snode, that return the same thing?
    @property
    def snode(self):
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return self._snode

    @property
    def _snode(self) -> "quadrants.lang.snode.SNode":  # type: ignore
        """Gets representative SNode for info purposes.

        Returns:
            SNode: Representative SNode (SNode of first field member).
        """
        return quadrants.lang.snode.SNode(self.vars[0].ptr.snode())  # type: ignore

    @property
    def shape(self) -> tuple[int, ...]:
        if not self._shape:
            phys = cast(tuple[int, ...], self._snode.shape)
            # For layout-tagged fields the SNode is allocated at the permuted *physical* shape; surface the
            # caller-facing *canonical* shape (inverse permutation) to match the ``Tensor`` / ``Ndarray`` contract
            # where ``.shape`` is always what was passed into the factory.
            layout = getattr(self, "_qd_layout", None)
            if layout is not None:
                layout_t = tuple(layout)
                if layout_t != tuple(range(len(layout_t))) and len(phys) == len(layout_t):
                    # ``phys[p] = canonical[layout[p]]`` ⇒ ``canonical[a] = phys[inv_layout[a]]``.
                    inv = [0] * len(layout_t)
                    for p, a in enumerate(layout_t):
                        inv[a] = p
                    phys = tuple(phys[inv[a]] for a in range(len(layout_t)))
            self._shape = phys
        return self._shape

    @property
    def layout(self):
        """Canonical-axis-permutation tuple, or ``None`` for identity.

        Mirrors :attr:`Ndarray.layout`: returns the same value the caller passed to ``qd.tensor(..., layout=...)``
        (or ``None`` if that kwarg was omitted / was the identity permutation). Lets downstream code introspect the
        physical layout without having to know which backend produced the tensor.

        Fields constructed directly via ``qd.field(..., order=...)`` also report their layout here — the ``order=``
        axis-string is translated into an integer permutation and stashed on the same ``_qd_layout`` attribute that
        layout-tagged ndarrays use.
        """
        layout = getattr(self, "_qd_layout", None)
        if layout is None:
            return None
        layout = tuple(layout)
        if layout == tuple(range(len(layout))):
            return None
        return layout

    @property
    def dtype(self) -> DataTypeCxx:
        if not self._dtype:
            self._dtype = cast(DataTypeCxx, self._snode._dtype)
        return self._dtype

    @property
    def _name(self) -> str:
        if not self.__name:
            self.__name = cast(str, self._snode._name)
        return self.__name

    def parent(self, n=1):
        """
        n (int): the number of levels going up from the representative SNode.
        """
        return self.snode.parent(n)

    def _get_field_members(self) -> list["Expr"]:
        return self.vars

    def _loop_range(self):
        """Gets SNode of representative field member for loop range info.

        Returns:
            quadrants_python.SNode: SNode of representative (first) field member.
        """
        return self.vars[0].ptr.snode()

    def _set_grad(self, grad: "Field") -> None:
        """Sets corresponding grad field (reverse mode)."""
        self.grad = grad

    def _set_dual(self, dual: "Field") -> None:
        """Sets corresponding dual field (forward mode)."""
        self.dual = dual

    @python_scope
    def fill(self, val: int | float) -> None:
        raise NotImplementedError()

    @python_scope
    def to_numpy(self, dtype: DataTypeCxx | None = None, *, copy=True):
        """Converts `self` to a numpy array.

        Args:
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def to_torch(self, device=None, *, copy=True):
        """Converts `self` to a torch tensor.

        Args:
            device (torch.device, optional): The desired device of returned tensor.
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises.

        Returns:
            torch.tensor: The result torch tensor.
        """
        raise NotImplementedError()

    @python_scope
    def from_numpy(self, arr):
        """Loads all elements from a numpy array.

        The shape of the numpy array needs to be the same as `self`.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        raise NotImplementedError()

    @python_scope
    def _from_external_arr(self, arr):
        raise NotImplementedError()

    @python_scope
    def from_torch(self, arr):
        """Loads all elements from a torch tensor.

        The shape of the torch tensor needs to be the same as `self`.

        Args:
            arr (torch.tensor): The source torch tensor.
        """
        self._from_external_arr(arr.contiguous())

    @python_scope
    def copy_from(self, other: "Field") -> None:
        """Copies all elements from another field.

        The shape of the other field needs to be the same as `self`.
        """
        if not isinstance(other, Field):
            raise TypeError("Cannot copy from a non-field object")
        if self.shape != other.shape:
            raise ValueError(f"qd.field shape {self.shape} does not match" f" the source field shape {other.shape}")
        from quadrants._kernels import tensor_to_tensor  # pylint: disable=C0415

        tensor_to_tensor(self, other)

    @python_scope
    def __setitem__(self, key: list[int] | int | None, value: int | float) -> None:
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key: list[int] | int | None) -> int | float:
        raise NotImplementedError()

    def __str__(self) -> str:
        if quadrants.lang.impl.inside_kernel():
            return self.__repr__()  # make pybind11 happy, see Matrix.__str__
        if self._snode.ptr is None:
            return "<Field: Definition of this field is incomplete>"
        return str(self.to_numpy())

    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key,)

        if len(key) != len(self.shape):
            raise AssertionError("Slicing is not supported on qd.field")

        # For layout-tagged fields (``_qd_layout = (a0, a1, ..., a_{N-1})`` canonical-axis permutation), the user writes
        # indices in canonical order but the underlying flat rank-N SNode is allocated at the permuted physical shape
        # and expects physical-order indices. Mirrors ``Tensor._permute_key`` in ``_tensor_wrapper.py``. Uses the same
        # convention as :func:`build_Subscript`: physical index at nesting level ``p`` is ``canonical[layout[p]]``.
        layout = getattr(self, "_qd_layout", None)
        if layout is not None:
            layout_t = tuple(layout)
            if layout_t != tuple(range(len(layout_t))):
                key = tuple(key[layout_t[p]] for p in range(len(layout_t)))

        return tuple(key) + ((0,) * (_qd_core.get_max_num_indices() - len(key)))  # type: ignore

    def _initialize_host_accessors(self):
        if self.host_accessors:
            return
        quadrants.lang.impl.get_runtime().materialize()
        self.host_accessors = [SNodeHostAccessor(e.ptr.snode()) for e in self.vars]

    def _host_access(self, key):
        return [SNodeHostAccess(e, key) for e in self.host_accessors]  # type: ignore

    def to_dlpack(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError("Struct for is only available in Quadrants scope.")


class ScalarField(Field):
    """Quadrants scalar field with SNode implementation.

    Args:
        var (Expr): Field member.
    """

    def __init__(self, var):
        super().__init__([var])

    def to_dlpack(self):
        """
        Note: caller is responsible for calling qd.sync() between modifying the field, and reading it.
        """
        impl.get_runtime().materialize()
        try:
            capsule = impl.get_runtime().prog.field_to_dlpack(self._snode.ptr, 0, 0, 0)
        except ModuleNotFoundError:
            # The C++ ``field_to_dlpack`` calls ``torch_supports_byte_offset`` which unconditionally does ``import
            # torch``. Guard here so callers that don't need torch (e.g. raw DLPack consumers) get a clear error
            # instead of a worker crash.
            raise ModuleNotFoundError(
                "field.to_dlpack() requires torch to be installed "
                "(the C++ layer checks torch version for DLPack byte_offset support)"
            ) from None
        # For layout-tagged fields the underlying dense SNode is allocated at the *permuted physical* shape; the C++
        # ``field_to_dlpack`` reports that physical shape because the SNode hierarchy no longer encodes the
        # canonical-axis permutation (which now lives on the Python-side ``_qd_layout`` attribute instead of nested
        # rank-1 SNodes). Patch the DLManagedTensor in place so consumers see the canonical shape with permuted
        # strides — the same view contract that layout-tagged ndarrays produce.
        layout = getattr(self, "_qd_layout", None)
        if layout is not None:
            _patch_field_dlpack_canonical(capsule, tuple(layout))
        return capsule

    def fill(self, val):
        """Fills this scalar field with a specified value."""
        if in_python_scope():
            from quadrants._kernels import fill_field  # pylint: disable=C0415

            fill_field(self, val)
        else:
            from quadrants._funcs import (  # pylint: disable=C0415
                field_fill_quadrants_scope,  # pylint: disable=C0415
            )

            field_fill_quadrants_scope(self, val)

    @python_scope
    def to_numpy(self, dtype=None, *, copy=True):
        """Converts this field to a ``numpy.ndarray``.

        Args:
            dtype: Optional target numpy dtype.
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises.
        """
        if copy is False:
            arr = _try_zerocopy_numpy(self, copy=False, is_scalar=True)
            if arr is not None:
                if dtype is not None and arr.dtype != dtype:
                    raise ValueError(f"copy=False is incompatible with dtype conversion ({arr.dtype} -> {dtype})")
                return arr

        if self.parent()._snode.ptr.type == _qd_core.SNodeType.dynamic:
            warn(
                "You are trying to convert a dynamic snode to a numpy array, be aware that inactive items in the snode will be converted to zeros in the resulting array."
            )
        if dtype is None:
            dtype = to_numpy_type(self.dtype)
        import numpy as np  # pylint: disable=C0415

        arr = np.zeros(shape=self.shape, dtype=dtype)  # type: ignore
        from quadrants._kernels import tensor_to_ext_arr  # pylint: disable=C0415

        tensor_to_ext_arr(self, arr)
        quadrants.lang.runtime_ops.sync()  # type: ignore
        return arr

    @python_scope
    def to_torch(self, device=None, *, copy=True):
        """Converts this field to a ``torch.Tensor``.

        Args:
            device: Optional torch device for the returned tensor.
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises.
        """
        if copy is False:
            return _try_zerocopy_torch(self, copy=copy, device=device, is_scalar=True)

        import torch  # pylint: disable=C0415

        arr = torch.zeros(size=self.shape, dtype=to_pytorch_type(self.dtype), device=device)
        from quadrants._kernels import tensor_to_ext_arr  # pylint: disable=C0415

        tensor_to_ext_arr(self, arr)
        quadrants.lang.runtime_ops.sync()  # type: ignore
        _mps_sync_if_metal()
        return arr

    @python_scope
    def _from_external_arr(self, arr):
        if len(self.shape) != len(arr.shape):
            raise ValueError(f"qd.field shape {self.shape} does not match" f" the numpy array shape {arr.shape}")
        for i, _ in enumerate(self.shape):
            if self.shape[i] != arr.shape[i]:
                raise ValueError(f"qd.field shape {self.shape} does not match" f" the numpy array shape {arr.shape}")
        from quadrants._kernels import ext_arr_to_tensor  # pylint: disable=C0415

        ext_arr_to_tensor(arr, self)
        # TODO: can we remove .runtime_ops here?
        quadrants.lang.runtime_ops.sync()  # type: ignore

    @python_scope
    def from_numpy(self, arr):
        """Copies the data from a `numpy.ndarray` into this field."""
        if not arr.flags.c_contiguous:
            import numpy as np  # pylint: disable=C0415

            arr = np.ascontiguousarray(arr)
        self._from_external_arr(arr)

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessors()
        self.host_accessors[0].setter(value, *self._pad_key(key))  # type: ignore

    @python_scope
    def __getitem__(self, key):
        self._initialize_host_accessors()
        # Check for potential slicing behaviour
        # for instance: x[0, :]
        padded_key = self._pad_key(key)
        import numpy as np  # pylint: disable=C0415

        for key in padded_key:
            if not isinstance(key, (int, np.integer)):
                raise TypeError(
                    f"Detected illegal element of type: {type(key)}. "
                    f"Please be aware that slicing a qd.field is not supported so far."
                )
        return self.host_accessors[0].getter(*padded_key)  # type: ignore

    def __repr__(self):
        # make interactive shell happy, prevent materialization
        return "<qd.field>"


class SNodeHostAccessor:
    def __init__(self, snode):
        if _qd_core.is_real(snode.data_type()):
            write_func = snode.write_float
            read_func = snode.read_float
        else:

            def write_func(key, value):
                if value >= 0:
                    snode.write_uint(key, value)
                else:
                    snode.write_int(key, value)

            if _qd_core.is_signed(snode.data_type()):
                read_func = snode.read_int
            else:
                read_func = snode.read_uint

        def getter(*key):
            assert len(key) == _qd_core.get_max_num_indices()
            return read_func(key)

        def setter(value, *key):
            assert len(key) == _qd_core.get_max_num_indices()
            write_func(key, value)
            # same as above
            if (
                impl.get_runtime().target_tape
                and impl.get_runtime().target_tape.grad_checker  # type: ignore
                and not impl.get_runtime().grad_replaced
            ):
                for x in impl.get_runtime().target_tape.grad_checker.to_check:  # type: ignore
                    assert snode != x.snode.ptr, "Overwritten is prohibitive when doing grad check."
                impl.get_runtime().target_tape.insert(write_func, (key, value))  # type: ignore

        self.getter = getter
        self.setter = setter


class SNodeHostAccess:
    def __init__(self, accessor, key):
        self.accessor = accessor
        self.key = key


class BitpackedFields:
    """Quadrants bitpacked fields, where fields with quantized types are packed together.

    Args:
        max_num_bits (int): Maximum number of bits all fields inside can occupy in total. Only 32 or 64 is allowed.
    """

    def __init__(self, max_num_bits):
        self.fields = []
        self.bit_struct_type_builder = _qd_core.BitStructTypeBuilder(max_num_bits)

    def place(self, *args, shared_exponent=False):
        """Places a list of fields with quantized types inside.

        Args:
            *args (List[Field]): A list of fields with quantized types to place.
            shared_exponent (bool): Whether the fields have a shared exponent.
        """
        if shared_exponent:
            self.bit_struct_type_builder.begin_placing_shared_exponent()
        count = 0
        for arg in args:
            assert isinstance(arg, Field)
            for var in arg._get_field_members():
                self.fields.append((var.ptr, self.bit_struct_type_builder.add_member(var.ptr.get_dt())))
                count += 1
        if shared_exponent:
            self.bit_struct_type_builder.end_placing_shared_exponent()
            if count <= 1:
                raise QuadrantsSyntaxError("At least 2 fields need to be placed when shared_exponent=True")


__all__ = ["BitpackedFields", "Field", "ScalarField"]
