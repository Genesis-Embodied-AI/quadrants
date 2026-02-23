import dataclasses
import enum
import numbers
import time
from typing import Any, Sequence

import numpy as np

from quadrants import _logging
from quadrants.types.annotations import Template

from .._ndarray import ScalarNdarray
from ..field import ScalarField
from ..kernel_arguments import ArgMetadata
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented
from .hash_utils import hash_iterable_strings

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


def dataclass_to_repr(raise_on_templated_floats: bool, path: tuple[str, ...], arg: Any) -> str:
    repr_l = []
    for field in dataclasses.fields(arg):
        child_value = getattr(arg, field.name)
        _repr = stringify_obj_type(raise_on_templated_floats, path + (field.name,), child_value, arg_meta=None)
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    return "[" + ",".join(repr_l) + "]"


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
    arg_type = type(obj)
    if isinstance(obj, ScalarNdarray):
        return f"[nd-{obj.dtype}-{len(obj.shape)}]"
    if isinstance(obj, VectorNdarray):
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}]"
    if isinstance(obj, ScalarField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        return None
    if isinstance(obj, MatrixNdarray):
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}]"
    if isinstance(obj, torch_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"  # type: ignore
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        return None
    if dataclasses.is_dataclass(obj):
        return dataclass_to_repr(raise_on_templated_floats, path, obj)
    if is_data_oriented(obj):
        child_repr_l = ["da"]
        _dict = {}
        try:
            # pyright is ok with this approach
            _asdict = getattr(obj, "_asdict")
            _dict = _asdict()
        except AttributeError:
            _dict = obj.__dict__
        for k, v in _dict.items():
            _child_repr = stringify_obj_type(raise_on_templated_floats, (*path, k), v, ArgMetadata(Template, ""))
            if _child_repr is None:
                _logging.warn(
                    f"""A kernel that has been marked as eligible for fast cache was passed 1 or more parameters that are not, in fact, eligible for fast cache: one of the parameters was a @qd.data_oriented objects, and one of its children was not eligible.
The data oriented object was of type {type(obj)} and the child {k}={type(v)} was not eligible. For information, the path of the value was {path}."""
                )
                return None
            child_repr_l.append(f"{k}: {_child_repr}")
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
    # The bit in caps should not be modified without updating corresponding test
    # The rest of free text can be freely modified
    # (will probably formalize this in more general doc / contributor guidelines at some point)
    _logging.warn(
        f"[FASTCACHE][PARAM_INVALID] Parameter with path {path} and type {arg_type} not allowed by fast cache."
    )
    return None


def hash_args(
    raise_on_templated_floats: bool, args: Sequence[Any], arg_metas: Sequence[ArgMetadata | None]
) -> str | None:
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls
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
            return None
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
