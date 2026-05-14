# type: ignore

from typing import TYPE_CHECKING, Union

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants.lang import _ndarray_pickle, impl
from quadrants.lang._metal_interop import mps_sync_if_metal

# Cache enum value at module level for fast lookup in hot paths
_arch_metal = _qd_core.Arch.metal

from quadrants.lang.exception import QuadrantsIndexError
from quadrants.lang.util import (
    cook_dtype,
    get_traceback,
    python_scope,
    to_numpy_type,
)
from quadrants.types import primitive_types
from quadrants.types.enums import Layout
from quadrants.types.ndarray_type import NdarrayTypeMetadata
from quadrants.types.utils import is_real, is_signed

if TYPE_CHECKING:
    from quadrants.lang.matrix import MatrixNdarray, VectorNdarray

    TensorNdarray = Union["ScalarNdarray", VectorNdarray, MatrixNdarray]


def _invert_layout(layout):
    """Return the inverse permutation of ``layout`` as a tuple.

    ``layout`` lists the *canonical* axis index at each successive physical-memory axis (outermost first). The inverse
    maps each canonical axis back to the physical-memory axis it lives on, which is exactly what numpy's
    ``transpose(axes=)`` and DLPack's ``shape`` / ``strides`` arrays consume.
    """
    n = len(layout)
    inv = [0] * n
    for src, dst in enumerate(layout):
        inv[dst] = src
    return tuple(inv)


def _is_identity_layout(layout):
    """``True`` if ``layout`` is ``None`` or the identity permutation."""
    return layout is None or layout == tuple(range(len(layout)))


class Ndarray:
    """Quadrants ndarray class.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the Ndarray.
    """

    # PERF-CRITICAL: This class attribute lets hotpath code access ``arg._qd_layout`` directly instead of ``getattr(arg,
    # "_qd_layout", None)``. Direct attribute access is measurably faster on the per-kernel-arg path in
    # _template_mapper_hotpath._extract_arg. Instance-level _qd_layout (set during layout-tagged allocation) shadows
    # this default.
    _qd_layout: tuple[int, ...] | None = None

    def __init__(self):
        self.host_accessor = None
        # `_physical_shape` is the underlying storage shape (matches the C++ ndarray buffer). `shape` is exposed
        # as a property: when a layout tag (`_qd_layout`) is present it returns the *canonical* shape the user
        # indexes inside kernels; otherwise it returns the physical shape (which is the same thing).
        self._physical_shape = None
        self.element_type = None
        self.dtype = None
        self.arr = None
        self.grad: "TensorNdarray | None" = None
        # We register with runtime, in order to enable reset to work later
        impl.get_runtime().ndarrays.add(self)

    def __reduce__(self):
        """Pickle support. Gradients (``.grad``) are not preserved because they are considered transient computation
        state."""
        return _ndarray_pickle.unpickle, (_ndarray_pickle.serialize(self),)

    def __del__(self):
        if impl is not None and impl.get_runtime is not None and impl.get_runtime() is not None:
            arr = getattr(self, "arr")
            if arr is not None:
                prog = impl.get_runtime()._prog
                if prog is not None:
                    prog.delete_ndarray(arr)

    def to_dlpack(self, versioned=False):
        """Export this ndarray as a DLPack capsule.

        Args:
            versioned: If True, emit a DLPack v1 capsule (writable numpy arrays). If False (default), emit v0
                (required by ``torch.utils.dlpack.from_dlpack``). See :meth:`ScalarField.to_dlpack`.

        The returned capsule carries the *canonical* shape and a permuted strides array on layout-tagged ndarrays, so
        consumers (`torch.utils.dlpack.from_dlpack`, etc.) see a transposed view of the physical buffer with no data
        movement. On untagged ndarrays this is byte-identical to the legacy export.

        Mirrors the field-backend behaviour: ``to_dlpack()`` on a field allocated via ``qd.tensor(...,
        backend=qd.Backend.FIELD, layout=...)`` (translated to ``order=...``) likewise returns a canonical view via
        permuted strides — see the C++ ``field_to_dlpack`` SNode-walk path.
        """
        if impl.current_cfg().arch == _arch_metal:
            impl.get_runtime().sync()
        layout = getattr(self, "_qd_layout", None)
        if _is_identity_layout(layout):
            return impl.get_runtime().prog.ndarray_to_dlpack(self, self.arr, versioned=versioned)
        return impl.get_runtime().prog.ndarray_to_dlpack(self, self.arr, list(layout), versioned=versioned)

    def _reset(self):
        """
        Called by runtime, when we call qd.reset()
        """
        self.arr = None
        self.grad = None
        self.host_accessor = None
        self._physical_shape = None
        self.element_type = None
        self.dtype = None

    @property
    def layout(self):
        """Canonical-axis-permutation tuple, or ``None`` for identity.

        Mirrors :attr:`Field.layout`: returns the same value the caller passed to ``qd.tensor(..., layout=...)`` (or
        ``None`` if that kwarg was omitted / was the identity permutation). Lets downstream code introspect the
        physical layout without having to know which backend produced the tensor.
        """
        layout = getattr(self, "_qd_layout", None)
        if _is_identity_layout(layout):
            return None
        return tuple(layout)

    @property
    def shape(self):
        """Canonical shape the user sees and indexes inside kernels.

        On a layout-tagged ndarray (``_qd_layout`` set), the underlying buffer is allocated at the *physical* (permuted)
        shape; this property inverts the layout permutation so callers see the canonical shape they passed to
        ``qd.tensor(..., shape=)``.

        On an untagged ndarray (no layout, or identity layout) physical and canonical coincide and this returns the
        physical shape.

        ``to_numpy()`` / ``to_torch()`` / ``to_dlpack()`` also return canonical views (with ``_qd_layout`` reflected
        as a non-identity stride pattern on the DLPack side); the layout is purely an internal performance hint.
        """
        phys = self._physical_shape
        layout = getattr(self, "_qd_layout", None)
        if phys is None or layout is None:
            return phys
        inv = _invert_layout(layout)
        return tuple(phys[inv[i]] for i in range(len(phys)))

    def get_type(self):
        return NdarrayTypeMetadata(self.element_type, self._physical_shape, self.grad is not None)

    @property
    def element_shape(self):
        """Gets ndarray element shape.

        Returns:
            Tuple[Int]: Ndarray element shape.
        """
        raise NotImplementedError()

    @python_scope
    def __setitem__(self, key, value):
        """Sets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.
            value (element type): Value to set.
        """
        raise NotImplementedError()

    @python_scope
    def __getitem__(self, key):
        """Gets ndarray element in Python scope.

        Args:
            key (Union[List[int], int, None]): Coordinates of the ndarray element.

        Returns:
            element type: Value retrieved.
        """
        raise NotImplementedError()

    @python_scope
    def fill(self, val):
        """Fills ndarray with a specific scalar value.

        Args:
            val (Union[int, float]): Value to fill.
        """
        if impl.current_cfg().arch != _qd_core.Arch.cuda and impl.current_cfg().arch != _qd_core.Arch.x64:
            self._fill_by_kernel(val)
        elif _qd_core.is_tensor(self.element_type):
            self._fill_by_kernel(val)
        elif self.dtype == primitive_types.f32:
            impl.get_runtime().prog.fill_float(self.arr, val)
        elif self.dtype == primitive_types.i32:
            impl.get_runtime().prog.fill_int(self.arr, val)
        elif self.dtype == primitive_types.u32:
            impl.get_runtime().prog.fill_uint(self.arr, val)
        else:
            self._fill_by_kernel(val)

    @python_scope
    def _ndarray_to_numpy(self):
        """Converts ndarray to a numpy array.

        Returns the *canonical* view: the output array has ``self.shape`` (the user-facing shape passed to
        ``qd.tensor(..., shape=)``) and is filled by a kernel whose canonical iteration is mapped to the underlying
        physical buffer through the AST layout-permutation. Untagged ndarrays see canonical == physical and pay no extra
        cost.

        Returns:
            numpy.ndarray: The result numpy array, in canonical axis order.
        """
        arr = np.zeros(shape=tuple(self.shape), dtype=to_numpy_type(self.dtype))
        from quadrants._kernels import ndarray_to_ext_arr  # pylint: disable=C0415

        ndarray_to_ext_arr(self, arr)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_matrix_to_numpy(self, as_vector):
        """Converts matrix ndarray to a numpy array.

        Returns:
            numpy.ndarray: The result numpy array.
        """
        arr = np.zeros(shape=self.arr.total_shape(), dtype=to_numpy_type(self.dtype))
        from quadrants._kernels import (  # pylint: disable=C0415
            ndarray_matrix_to_ext_arr,  # pylint: disable=C0415
        )

        layout_is_aos = 1
        ndarray_matrix_to_ext_arr(self, arr, layout_is_aos, as_vector)
        impl.get_runtime().sync()
        return arr

    @python_scope
    def _ndarray_from_numpy(self, arr):
        """Loads all values from a numpy array.

        ``arr.shape`` is validated against the *canonical* shape (what ``self.shape`` reports). The
        ``ext_arr_to_ndarray`` kernel iterates ``arr`` canonically and writes through ``ndarray[I]``, so on
        layout-tagged destinations the AST permutation routes canonical positions into the underlying physical buffer
        without any python-side transpose.

        Args:
            arr (numpy.ndarray): The source numpy array, in canonical axis order.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        canonical_shape = tuple(self.shape)
        if canonical_shape != tuple(arr.shape):
            raise ValueError(f"Mismatch shape: {canonical_shape} expected, but {tuple(arr.shape)} provided")
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        from quadrants._kernels import ext_arr_to_ndarray  # pylint: disable=C0415

        ext_arr_to_ndarray(arr, self)
        impl.get_runtime().sync()

    @python_scope
    def _ndarray_matrix_to_torch(self, as_vector, device=None):
        """Mirror of ``_ndarray_matrix_to_numpy`` that fills a torch tensor instead of a numpy array.

        Allocates a ``torch.zeros`` of ``self.arr.total_shape()`` (the canonical-major shape including the trailing
        element-axis extents) and dispatches the same ``ndarray_matrix_to_ext_arr`` bridge kernel — torch tensors
        expose the same external-array interface to the kernel as numpy arrays do.
        """
        import torch  # pylint: disable=C0415

        from quadrants.lang.util import to_pytorch_type  # pylint: disable=C0415

        out = torch.zeros(size=tuple(self.arr.total_shape()), dtype=to_pytorch_type(self.dtype), device=device)
        from quadrants._kernels import (  # pylint: disable=C0415
            ndarray_matrix_to_ext_arr,  # pylint: disable=C0415
        )

        layout_is_aos = 1
        ndarray_matrix_to_ext_arr(self, out, layout_is_aos, as_vector)
        impl.get_runtime().sync()
        mps_sync_if_metal()
        return out

    @python_scope
    def _ndarray_matrix_from_torch(self, arr, as_vector):
        """Mirror of ``_ndarray_matrix_from_numpy`` that ingests a torch tensor. ``.contiguous()`` is forced so the
        bridge kernel sees a tightly-packed external array."""
        contig = arr.contiguous()
        if tuple(self.arr.total_shape()) != tuple(contig.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.total_shape())} expected, but {tuple(contig.shape)} provided"
            )
        from quadrants._kernels import (  # pylint: disable=C0415
            ext_arr_to_ndarray_matrix,  # pylint: disable=C0415
        )

        layout_is_aos = 1
        ext_arr_to_ndarray_matrix(contig, self, layout_is_aos, as_vector)
        impl.get_runtime().sync()

    @python_scope
    def _ndarray_matrix_from_numpy(self, arr, as_vector):
        """Loads all values from a numpy array.

        Args:
            arr (numpy.ndarray): The source numpy array.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{np.ndarray} expected, but {type(arr)} provided")
        if tuple(self.arr.total_shape()) != tuple(arr.shape):
            raise ValueError(
                f"Mismatch shape: {tuple(self.arr.total_shape())} expected, but {tuple(arr.shape)} provided"
            )
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        from quadrants._kernels import (  # pylint: disable=C0415
            ext_arr_to_ndarray_matrix,  # pylint: disable=C0415
        )

        layout_is_aos = 1
        ext_arr_to_ndarray_matrix(arr, self, layout_is_aos, as_vector)
        impl.get_runtime().sync()

    @python_scope
    def _get_element_size(self):
        """Returns the size of one element in bytes.

        Returns:
            Size in bytes.
        """
        return self.arr.element_size()

    @python_scope
    def _get_nelement(self):
        """Returns the total number of elements.

        Returns:
            Total number of elements.
        """
        return self.arr.nelement()

    @python_scope
    def copy_from(self, other):
        """Copies all elements from another ndarray.

        The shape of the other ndarray needs to be the same as `self`.

        Args:
            other (Ndarray): The source ndarray.
        """
        assert isinstance(other, Ndarray)
        assert tuple(self.shape) == tuple(
            other.shape
        ), f"copy_from shape mismatch: destination {tuple(self.shape)} vs source {tuple(other.shape)}"
        from quadrants._kernels import ndarray_to_ndarray  # pylint: disable=C0415

        ndarray_to_ndarray(self, other)
        impl.get_runtime().sync()

    def _set_grad(self, grad: "TensorNdarray"):
        """Sets the gradient ndarray.

        Args:
            grad (Ndarray): The gradient ndarray.
        """
        self.grad = grad

    def has_grad(self) -> bool:
        """Whether this ndarray has a gradient companion ndarray allocated.

        ``True`` iff the ndarray was allocated with ``needs_grad=True``.
        """
        return self.grad is not None

    def has_dual(self) -> bool:
        """Whether this ndarray has a dual companion ndarray allocated.

        ``Ndarray`` does not support dual storage; always returns ``False``.
        """
        return False

    def __deepcopy__(self, memo=None):
        """Copies all elements to a new ndarray.

        Returns:
            Ndarray: The result ndarray.
        """
        raise NotImplementedError()

    def _fill_by_kernel(self, val):
        """Fills ndarray with a specific scalar value using a qd.kernel.

        Args:
            val (Union[int, float]): Value to fill.
        """
        raise NotImplementedError()

    def _slice_to_buffer_view(self, key: slice):
        """Convert a slice key into a BufferView over this 1D ndarray.

        Called from ``__getitem__`` when the key is a ``slice``.
        Supports start/stop with negative indices; step must be 1.
        """
        from quadrants.lang.buffer_view import BufferView  # pylint: disable=C0415  # noqa: I001

        if len(self.shape) != 1:
            raise TypeError(f"ndarray slice to BufferView requires a 1D array, got shape {self.shape}")
        start, stop, step = key.indices(self.shape[0])
        if step != 1:
            raise ValueError(f"BufferView slice requires step=1, got step={step}")
        size = max(stop - start, 0)
        return BufferView(self, start, size)

    @python_scope
    def _pad_key(self, key):
        if key is None:
            key = ()
        if not isinstance(key, (tuple, list)):
            key = (key,)
        if len(key) != len(self.arr.total_shape()):
            raise QuadrantsIndexError(f"{len(self.arr.total_shape())}d ndarray indexed with {len(key)}d indices: {key}")
        return key

    @python_scope
    def _initialize_host_accessor(self):
        if self.host_accessor:
            return
        impl.get_runtime().materialize()
        self.host_accessor = NdarrayHostAccessor(self.arr)


class ScalarNdarray(Ndarray):
    """Quadrants ndarray with scalar elements.

    Args:
        dtype (DataType): Data type of each value.
        shape (Tuple[int]): Shape of the ndarray.
    """

    def __init__(self, dtype, arr_shape):
        super().__init__()
        self.dtype = cook_dtype(dtype)
        if impl.is_python_backend():
            import torch  # pylint: disable=C0415

            from quadrants.lang.util import (  # pylint: disable=C0415
                dtype_to_torch_dtype,
            )

            self.arr = torch.zeros(shape=arr_shape, dtype=dtype_to_torch_dtype(dtype))
        else:
            self.arr = impl.get_runtime().prog.create_ndarray(
                self.dtype, arr_shape, layout=Layout.NULL, zero_fill=True, dbg_info=_qd_core.DebugInfo(get_traceback())
            )
        self._physical_shape = tuple(self.arr.shape)
        self.element_type = dtype

    @property
    def element_shape(self):
        return ()

    @python_scope
    def __setitem__(self, key, value):
        self._initialize_host_accessor()
        self.host_accessor.setter(value, *self._pad_key(key))

    @python_scope
    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._slice_to_buffer_view(key)
        self._initialize_host_accessor()
        return self.host_accessor.getter(*self._pad_key(key))

    @python_scope
    def to_numpy(self, dtype=None, *, copy=True):
        """Return a canonical-view NumPy array.

        Args:
            dtype: Optional numpy dtype to cast the result to. ``None`` keeps the native ndarray dtype.
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises,
                ``None`` uses zero-copy when available and falls back to a copy otherwise.
        """
        if copy is not True:
            from quadrants.lang.field import (  # pylint: disable=C0415
                _try_zerocopy_numpy,
            )

            arr = _try_zerocopy_numpy(self, copy=copy, is_ndarray=True)
            if arr is not None:
                if dtype is not None and arr.dtype != dtype:
                    if copy is False:
                        raise ValueError(f"copy=False is incompatible with dtype conversion ({arr.dtype} -> {dtype})")
                    # copy=None: fall through to the copy path for dtype conversion
                else:
                    return arr

        arr = self._ndarray_to_numpy()
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    @python_scope
    def from_numpy(self, arr):
        self._ndarray_from_numpy(arr)

    @python_scope
    def to_torch(self, device=None, *, copy=True):
        """Return a canonical-view torch tensor.

        Mirrors :meth:`Field.to_torch`. The destination is a ``torch.zeros(self.shape, ...)`` allocation that the
        bridge kernel writes into via the canonical iteration path, so layout-tagged ndarrays produce a canonical
        view just like ``to_numpy()`` does.

        Args:
            copy: ``True`` (default) returns an independent copy, ``False`` requires zero-copy or raises,
                ``None`` uses zero-copy when available and falls back to a copy otherwise.
        """
        if copy is not True:
            from quadrants.lang.field import (  # pylint: disable=C0415
                _try_zerocopy_torch,
            )

            result = _try_zerocopy_torch(self, copy=copy, device=device, is_ndarray=True)
            if result is not None:
                return result

        import torch  # pylint: disable=C0415

        from quadrants.lang.util import to_pytorch_type  # pylint: disable=C0415

        canonical_shape = tuple(self.shape)
        out = torch.zeros(size=canonical_shape, dtype=to_pytorch_type(self.dtype), device=device)
        from quadrants._kernels import ndarray_to_ext_arr  # pylint: disable=C0415

        ndarray_to_ext_arr(self, out)
        impl.get_runtime().sync()
        mps_sync_if_metal()
        return out

    @python_scope
    def from_torch(self, arr):
        """Load all values from a torch tensor.

        The torch tensor must have the same canonical shape as ``self``; a non-contiguous source is materialised via
        ``.contiguous()`` so the bridge kernel sees a tightly packed external array.
        """
        contig = arr.contiguous()
        canonical_shape = tuple(self.shape)
        if canonical_shape != tuple(contig.shape):
            raise ValueError(f"Mismatch shape: {canonical_shape} expected, but {tuple(contig.shape)} provided")
        from quadrants._kernels import ext_arr_to_ndarray  # pylint: disable=C0415

        ext_arr_to_ndarray(contig, self)
        impl.get_runtime().sync()

    def __deepcopy__(self, memo=None):
        ret_arr = ScalarNdarray(self.dtype, self._physical_shape)
        if self._qd_layout is not None:
            ret_arr._qd_layout = self._qd_layout
        ret_arr.copy_from(self)
        return ret_arr

    def _fill_by_kernel(self, val):
        from quadrants._kernels import fill_ndarray  # pylint: disable=C0415

        fill_ndarray(self, val)

    def __repr__(self):
        return "<qd.ndarray>"


class NdarrayHostAccessor:
    def __init__(self, ndarray):
        dtype = ndarray.element_data_type()
        if is_real(dtype):

            def getter(*key):
                return ndarray.read_float(key)

            def setter(value, *key):
                ndarray.write_float(key, value)

        else:
            if is_signed(dtype):

                def getter(*key):
                    return ndarray.read_int(key)

            else:

                def getter(*key):
                    return ndarray.read_uint(key)

            def setter(value, *key):
                ndarray.write_int(key, value)

        self.getter = getter
        self.setter = setter


class NdarrayHostAccess:
    """Class for accessing VectorNdarray/MatrixNdarray in Python scope.
    Args:
        arr (Union[VectorNdarray, MatrixNdarray]): See above.
        indices_first (Tuple[Int]): Indices of first-level access (coordinates in the field).
        indices_second (Tuple[Int]): Indices of second-level access (indices in the vector/matrix).
    """

    def __init__(self, arr, indices_first, indices_second):
        self.ndarr = arr
        self.arr = arr.arr
        self.indices = indices_first + indices_second

        def getter():
            self.ndarr._initialize_host_accessor()
            return self.ndarr.host_accessor.getter(*self.ndarr._pad_key(self.indices))

        def setter(value):
            self.ndarr._initialize_host_accessor()
            self.ndarr.host_accessor.setter(value, *self.ndarr._pad_key(self.indices))

        self.getter = getter
        self.setter = setter


__all__ = ["Ndarray", "ScalarNdarray"]
