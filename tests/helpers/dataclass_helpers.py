import dataclasses
from typing import Any, cast

import quadrants as qd


def _make_child_obj(obj_type: Any) -> Any:
    if isinstance(obj_type, qd.types.NDArray):
        ndarray_type = cast(qd.types.ndarray, obj_type)
        assert ndarray_type.ndim is not None
        shape = tuple([10] * ndarray_type.ndim)
        child_obj = qd.ndarray(ndarray_type.dtype, shape=shape)
    elif dataclasses.is_dataclass(obj_type):
        child_obj = build_struct(obj_type)
    elif isinstance(obj_type, qd.Template) or obj_type == qd.Template:
        child_obj = qd.field(qd.i32, (10,))
    else:
        raise Exception("unknown type ", obj_type)
    return child_obj


def build_struct(struct_type: Any) -> Any:
    member_objects = {}
    for field in dataclasses.fields(struct_type):
        child_obj = _make_child_obj(field.type)
        member_objects[field.name] = child_obj
    dataclass_object = struct_type(**member_objects)
    return dataclass_object


def build_obj_tuple_from_type_dict(name_to_type: dict[str, Any]) -> tuple[Any, ...]:
    obj_l = []
    for _name, param_type in name_to_type.items():
        child_obj = _make_child_obj(param_type)
        obj_l.append(child_obj)
    return tuple(obj_l)
