import dataclasses
import enum
import numbers
import time
from typing import Any, Sequence

import numpy as np

from quadrants import _logging, _tensor_wrapper
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants._tensor_wrapper import Tensor as _TensorWrapper
from quadrants.types.annotations import Template

from .._ndarray import ScalarNdarray
from .._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from ..field import ScalarField
from ..kernel_arguments import ArgMetadata
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented, is_dataclass_instance
from .hash_utils import hash_iterable_strings

_FIELD_TYPES = (ScalarField, MatrixField)

try:
    import torch

    torch_type = torch.Tensor
except ImportError:
    torch_type = ()


g_num_calls = 0
g_num_args = 0
g_hashing_time = 0
g_repr_time = 0
g_num_ignored_calls = 0


FIELD_METADATA_CACHE_VALUE = "add_value_to_cache_key"

_DC_REPR_NONE = object()


# Set by ``stringify_obj_type`` when it encounters a recognised-but-unsupported tensor-like type (``Field`` /
# ``MatrixField``) anywhere in the traversal — including nested under a dataclass or another data_oriented object.
# The ``stable_members=True`` data_oriented walker uses this to differentiate two reasons a child returned ``None``:
# truly-opaque metadata (``RigidSolver._uid: UID``, etc.) which is inert and can be skipped, vs a tensor-like type
# whose value affects kernel codegen and must invalidate fastcache for the whole call. Reset at the top of each
# ``hash_args``; snapshotted/restored around each nested ``stringify_obj_type`` call inside the data_oriented walker.
_hit_recognised_unsupported = False


def _mark_hit_recognised_unsupported() -> None:
    global _hit_recognised_unsupported  # pylint: disable=global-statement
    _hit_recognised_unsupported = True


class FastcacheSkip(enum.Enum):
    """Why fastcache does not apply to this call."""

    FIELD_VIA_TENSOR = "field_via_tensor"
    WARN = "warn"


# Set when the fastcache skip is something callers should warn about (as opposed to a Field arriving through a
# qd.Tensor annotation, which is a normal silent path). Reset at the start of each hash_args call.
_should_warn = False


# Counter set by the data_oriented walker when entering a ``_qd_stable_members`` object. While nonzero, the
# unknown-type branch of ``stringify_obj_type`` returns ``None`` silently instead of logging
# ``[FASTCACHE][PARAM_INVALID]``. ``stable_members=True`` is the user's promise that the class's member set / types
# don't change after construction — under that promise, opaque members like ``RigidSolver._uid`` (a
# ``genesis.utils.uid.UID``) don't affect kernel codegen so they can be skipped silently rather than killing
# fastcache for the whole call. Single-threaded by construction (the hasher only runs during JIT compile).
_skip_unknown_warn_depth = 0


def _mark_warn_if_not_tensor_annotation(arg_meta: ArgMetadata | None) -> None:
    """Flag that a warning is needed if the Field didn't arrive through a qd.Tensor annotation."""
    global _should_warn  # pylint: disable=global-statement
    if arg_meta is not None and arg_meta.annotation is not _TensorWrapper:
        _should_warn = True


def _mark_should_warn() -> None:
    global _should_warn  # pylint: disable=global-statement
    _should_warn = True


def dataclass_to_repr(raise_on_templated_floats: bool, path: tuple[str, ...], arg: Any) -> str | None:
    # PERF: For frozen dataclasses, the repr never changes. Cache it on the instance to avoid repeated
    # ``dataclasses.fields()`` calls (which are slow due to extra runtime checks — see _template_mapper_hotpath.py
    # module docstring). The cache is stored as ``_qd_dc_repr`` via ``object.__setattr__`` to bypass frozen guards.
    # A cached ``None`` is stored as the sentinel ``_DC_REPR_NONE`` to distinguish "not yet computed" from
    # "computed but not fast-cacheable".
    is_frozen = type(arg).__hash__ is not None
    if is_frozen:
        cached = getattr(arg, "_qd_dc_repr", None)
        if cached is _DC_REPR_NONE:
            return None
        if cached is not None:
            return cached
    repr_l = []
    for field in dataclasses.fields(arg):
        child_value = getattr(arg, field.name)
        _repr = stringify_obj_type(raise_on_templated_floats, path + (field.name,), child_value, arg_meta=None)
        if _repr is None:
            if isinstance(child_value, _FIELD_TYPES) and field.type is not _TensorWrapper:
                _mark_should_warn()
            if is_frozen:
                try:
                    object.__setattr__(arg, "_qd_dc_repr", _DC_REPR_NONE)
                except AttributeError:
                    pass
            return None
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    result = "[" + ",".join(repr_l) + "]"
    if is_frozen:
        try:
            object.__setattr__(arg, "_qd_dc_repr", result)
        except AttributeError:
            pass
    return result


def _is_template(arg_meta: ArgMetadata | None) -> bool:
    if arg_meta is None:
        return False
    annot = arg_meta.annotation
    return annot is Template or isinstance(annot, Template)


def stringify_obj_type(
    raise_on_templated_floats: bool, path: tuple[str, ...], obj: object, arg_meta: ArgMetadata | None
) -> str | None:
    """
    Convert an object into a string representation that only depends on its type.

    String should somehow represent the type of obj. Doesnt have to be hashed, nor does it have
    to be the actual python type string, just a string that is representative of the type, and won't collide
    with different (allowed) types. String should be non-empty.

    Note that fields are not included in fast cache.

    arg_meta should only be non-None for the top level arguments and for data oriented objects. It is
    used currently to determine whether a value is added to the cache key, as well as the name. eg
    - at the top level, primitive types have their values added to the cache key if their annotation is qd.Template,
      since they are baked into the kernel
    - in data oriented objects, the values of all primitive types are added to the cache key, since they are baked
      into the kernel, and require a kernel recompilation, when they change
    """
    # ``qd.Tensor`` wrappers passed as struct fields. The top-level kernel-arg unwrap hook in ``Kernel.__call__`` strips
    # wrappers off positional / keyword args before the fastcache hasher sees them, but the dataclass / data-oriented
    # walkers below (``dataclass_to_repr`` and the ``is_data_oriented`` branch) do raw ``getattr`` to fetch struct
    # fields, so a wrapper stored as a struct field arrives here un-stripped. Without this branch the hasher falls
    # through to the ``[FASTCACHE][PARAM_INVALID]`` warning and disables the fast path for the whole call. See
    # ``perso_hugh/doc/quadrants-tensor.md`` §8.14.
    # ``qd.Tensor`` wrappers: unwrap to the bare impl so the type checks below match. After unwrap, ``_qd_layout`` (if
    # any) is on the impl.
    #
    # PERF-CRITICAL: The _any_tensor_constructed guard makes this check zero-cost when no qd.Tensor has been created.
    # ``type(obj) in _TENSOR_WRAPPER_TYPES`` is used instead of ``isinstance`` because it is a pointer comparison (~10
    # ns) vs an MRO walk (~100–200 ns). Do not replace with isinstance or remove the guard.
    if (
        _tensor_wrapper._any_tensor_constructed and type(obj) in _TENSOR_WRAPPER_TYPES
    ):  # pyright: ignore[reportOptionalMemberAccess]
        obj = obj._unwrap()  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = type(obj)
    _layout = getattr(obj, "_qd_layout", None)
    _layout_tag = "" if _layout is None else f"-L{_layout!r}"
    if isinstance(obj, ScalarNdarray):
        return f"[nd-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, VectorNdarray):
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, ScalarField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        _mark_hit_recognised_unsupported()
        return None
    if isinstance(obj, MatrixNdarray):
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, torch_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"  # type: ignore
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        _mark_hit_recognised_unsupported()
        return None
    if is_dataclass_instance(obj):
        return dataclass_to_repr(raise_on_templated_floats, path, obj)
    if is_data_oriented(obj):
        # ``@qd.data_oriented(stable_members=True)``: the class promises its member *set* and *types* don't change
        # after construction. Under that contract, unrecognised member types (e.g. Genesis's ``RigidSolver._uid`` of
        # type ``genesis.utils.uid.UID``, or any other opaque metadata) are treated as inert from fastcache's
        # perspective: they don't affect kernel codegen so they can be skipped silently rather than killing fastcache
        # for the whole call. Without this, migrating a kernel from a standalone ``@qd.kernel`` function to a method
        # on a ``@qd.data_oriented`` class disables fastcache the moment the class holds any opaque metadata, even
        # though the kernel's compiled output would be identical.
        stable_members = bool(type(obj).__dict__.get("_qd_stable_members"))
        child_repr_l = ["da"]
        _dict = {}
        try:
            # pyright is ok with this approach
            _asdict = getattr(obj, "_asdict")
            _dict = _asdict()
        except AttributeError:
            _dict = obj.__dict__
        global _skip_unknown_warn_depth  # pylint: disable=global-statement
        if stable_members:
            _skip_unknown_warn_depth += 1
        try:
            for k, v in _dict.items():
                # Skip Quadrants method-descriptor cache entries. ``QuadrantsCallable.__get__``
                # stashes the per-instance ``BoundQuadrantsCallable`` on ``instance.__dict__`` so
                # that subsequent ``instance.method`` lookups skip the descriptor allocation;
                # those entries are not data and must not invalidate the fastcache key.
                v_type = type(v)
                if v_type is QuadrantsCallable or v_type is BoundQuadrantsCallable:
                    continue
                # Snapshot the recognised-but-unsupported flag around the recursive call so we can tell whether
                # *this child's* subtree hit a ``Field`` / ``MatrixField`` (in which case we must fail fastcache
                # even under ``stable_members``).
                global _hit_recognised_unsupported  # pylint: disable=global-statement
                _hit_recognised_unsupported = False
                _child_repr = stringify_obj_type(raise_on_templated_floats, (*path, k), v, ArgMetadata(Template, ""))
                child_hit_field = _hit_recognised_unsupported
                if _child_repr is None:
                    # Differentiate two reasons ``stringify_obj_type`` returns None:
                    #
                    #   (a) RECOGNISED-BUT-UNSUPPORTED: ``Field`` / ``MatrixField`` somewhere in this child's
                    #       subtree. These are *known* tensor-like types whose values affect kernel codegen but
                    #       which fastcache doesn't yet handle. Killing fastcache for the whole call is the
                    #       intended contract — ``test_num_envs[False-...]`` pins this behaviour for the field
                    #       backend.
                    #   (b) TRULY-OPAQUE: anything that falls through to the ``[FASTCACHE][PARAM_INVALID]``
                    #       warning at the bottom of ``stringify_obj_type`` (``RigidSolver._uid`` of type
                    #       ``UID``, etc.). For ``stable_members=True`` containers, opaque metadata is inert by
                    #       the user's contract and can be skipped without invalidating the hash for the rest
                    #       of the members.
                    if stable_members and not child_hit_field:
                        continue
                    if _should_warn:
                        _logging.warn(
                            f"A kernel that has been marked as eligible for fast cache was passed 1 or more "
                            f"parameters that are not, in fact, eligible for fast cache: one of the parameters was a "
                            f"@qd.data_oriented object, and one of its children was not eligible. The data oriented "
                            f"object was of type {type(obj)} and the child {k}={type(v)} was not eligible. For "
                            f"information, the path of the value was {path}."
                        )
                    return None
                child_repr_l.append(f"{k}: {_child_repr}")
        finally:
            if stable_members:
                _skip_unknown_warn_depth -= 1
        return ", ".join(child_repr_l)
    if issubclass(arg_type, (numbers.Number, np.number)):
        if _is_template(arg_meta):
            if raise_on_templated_floats and isinstance(obj, float):
                raise ValueError("Floats should not be used in template parameters.")
            # cache value too
            return f"{arg_type}={obj}"
        return str(arg_type)
    if arg_type is np.bool_:
        # np is deprecating bool. Treat specially/carefully
        if _is_template(arg_meta):
            # cache value too
            return f"np.bool_={obj}"
        return "np.bool_"
    if isinstance(obj, enum.Enum):
        return f"enum-{obj.name}-{obj.value}"
    if _skip_unknown_warn_depth > 0:
        # Inside a ``stable_members=True`` data_oriented walk: opaque members are tolerated by contract, so don't log
        # the per-member ``[FASTCACHE][PARAM_INVALID]`` warning. The data_oriented walker reads the returned ``None``
        # and skips this member.
        return None
    _mark_should_warn()
    # The bit in caps should not be modified without updating corresponding test
    # The rest of free text can be freely modified
    # (will probably formalize this in more general doc / contributor guidelines at some point)
    _logging.warn(
        f"[FASTCACHE][PARAM_INVALID] Parameter with path {path} and type {arg_type} not allowed by fast cache."
    )
    return None


def hash_args(
    raise_on_templated_floats: bool, args: Sequence[Any], arg_metas: Sequence[ArgMetadata | None]
) -> str | FastcacheSkip:
    """Return the args hash string, or a HashFailure explaining why hashing failed."""
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls, _should_warn, _hit_recognised_unsupported  # pylint: disable=line-too-long
    _should_warn = False
    _hit_recognised_unsupported = False
    g_num_calls += 1
    g_num_args += len(args)
    hash_l = []
    if len(args) != len(arg_metas):
        raise RuntimeError(
            f"Number of args passed in {len(args)} doesnt match number of declared args {len(arg_metas)}"
        )
    for i_arg, arg in enumerate(args):
        start = time.time()
        _hash = stringify_obj_type(raise_on_templated_floats, (str(i_arg),), arg, arg_metas[i_arg])
        g_repr_time += time.time() - start
        if not _hash:
            g_num_ignored_calls += 1
            return FastcacheSkip.WARN if _should_warn else FastcacheSkip.FIELD_VIA_TENSOR
        hash_l.append(_hash)
    start = time.time()
    res = hash_iterable_strings(hash_l)
    g_hashing_time += time.time() - start
    return res


def dump_stats() -> None:
    print("args hasher dump stats")
    print("total calls", g_num_calls)
    print("ignored calls", g_num_ignored_calls)
    print("total args", g_num_args)
    print("hashing time", g_hashing_time)
    print("arg representation time", g_repr_time)
