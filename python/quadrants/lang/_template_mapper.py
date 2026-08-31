from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType

from quadrants.lang import impl
from quadrants.lang.impl import Program
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import is_data_oriented
from quadrants.types import ndarray_type

from .._test_tools import warnings_helper
from ._kernel_types import ArgsHash
from ._optional_annotation import OPTIONAL_ABSENT
from ._template_mapper_hotpath import (
    _extract_arg,
    _primitive_types,
    _struct_nd_paths_for,
    annotation_has_final_subtree,
)

# Whether an arg of this class contributes to the args_hash ndarray-id walk in ``TemplateMapper.lookup``. Only the
# decision is per class; the paths themselves must stay per instance, since a @qd.data_oriented class can have a
# different attribute structure per instance (Genesis ``DataManager``).
_arg_disposition: dict[type, object] = {}
_SKIP = object()
_PER_INSTANCE = object()


def _classify_disposition(arg: Any) -> object:
    """Return ``_SKIP`` (no per-call walk for this class) or ``_PER_INSTANCE``.

    ``_qd_stable_members`` promises ndarray members are never reassigned, so the walk can be skipped. It is a
    launch-time hint only and has no bearing on fastcache keys."""
    if not is_data_oriented(arg):
        return _SKIP
    if type(arg).__dict__.get("_qd_stable_members"):
        return _SKIP
    return _PER_INSTANCE


Key: TypeAlias = tuple[Any, ...]


def _destroy_callback(template_mapper_ref: ReferenceType["TemplateMapper"], ref: ReferenceType):
    maybe_template_mapper = template_mapper_ref()
    if maybe_template_mapper is not None:
        maybe_template_mapper._mapping_cache.clear()
        maybe_template_mapper._mapping_cache_tracker.clear()
        maybe_template_mapper._prog_weakref = None


class TemplateMapper:
    """
    This should probably be renamed to sometihng like FeatureMapper, or
    FeatureExtractor, since:
    - it's not specific to templates
    - it extracts what are later called 'features', for example for ndarray this includes:
        - element type
        - number dimensions
        - needs grad (or not)
    - these are returned as a heterogeneous tuple, whose contents depends on the type
    """

    def __init__(self, arguments: list[ArgMetadata], template_slot_locations: list[int]) -> None:
        self.arguments: list[ArgMetadata] = arguments
        self.num_args: int = len(arguments)
        self.template_slot_locations: list[int] = template_slot_locations
        self.mapping: dict[Key, int] = {}
        self._mapping_cache: dict[ArgsHash, tuple[int, Key]] = {}
        self._mapping_cache_tracker: dict[ArgsHash, list[ReferenceType | None]] = {}
        self._prog_weakref: ReferenceType[Program] | None = None
        # Whether a Final-bearing argument disables the instance-keyed ``_mapping_cache`` (see
        # ``annotation_has_final_subtree``). Computed lazily on first ``lookup`` so its ``Final`` validation runs then.
        self._mapping_cache_disabled: bool | None = None

    def extract(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> Key:
        # Optional ndarray + None gets its own key: _extract_arg's NdarrayType branch reads .shape and raises on None.
        return tuple(
            [
                (
                    OPTIONAL_ABSENT
                    if arg is None and kernel_arg.optional and type(kernel_arg.annotation) is ndarray_type.NdarrayType
                    else _extract_arg(raise_on_templated_floats, arg, kernel_arg.annotation, kernel_arg.name)
                )
                for arg, kernel_arg in zip(args, self.arguments)
            ]
        )

    def lookup(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[int, Key]:
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        # Keep track of quadrants runtime to automatically clear cache if destroyed
        if self._prog_weakref is None:
            prog = impl.get_runtime().prog
            assert prog is not None
            self._prog_weakref = ReferenceType(prog, partial(_destroy_callback, ReferenceType(self)))
        else:
            # Since we already store a weak reference to quadrants program, it is much faster to use it rather than
            # paying the overhead of calling pybind11 functions (~200ns vs 5ns).
            prog = self._prog_weakref()
        assert prog is not None

        # Note that it is not necessary to handle primitive types separately here because primitive types are
        # immutable and therefore identical primitive values usually reuse the same addresses for efficiency unless
        # extra effort is made to do otherwise (this behavior is referring to as "interning"). Avoiding special
        # branching for primitive types dramatically improve performance of hash computation.
        mapping_cache_tracker: list[ReferenceType | None] | None = None
        args_hash: ArgsHash = tuple([id(arg) for arg in args])
        # A ``@qd.data_oriented`` container's member ndarrays can be reassigned between calls on the same instance
        # (``state.x = other_ndarray``), which ``id(arg)`` cannot see, so the spec-key cache would serve an entry
        # compiled for the old dtype/ndim. Fold the reachable ndarray ids in as well.
        #
        # Only ``template_slot_locations`` is iterated: a data_oriented container can only appear at a ``qd.template()``
        # position (a typed-dataclass arg carries a dataclass type, and data_oriented classes are never dataclasses).
        #
        # PERF: the ``arg.__dict__["_qd_nd_paths"]`` lookup is inlined rather than left to ``_struct_nd_paths_for``,
        # which costs ~60ns/call at 4 template args - ~15% of this loop. The call remains for the cold-miss cases.
        nd_ids: list = []
        for i in self.template_slot_locations:
            arg = args[i]
            cls = type(arg)
            disposition = _arg_disposition.get(cls)
            if disposition is None:
                disposition = _classify_disposition(arg)
                _arg_disposition[cls] = disposition
            if disposition is _SKIP:
                continue
            try:
                paths = arg.__dict__["_qd_nd_paths"]
            except (AttributeError, KeyError):
                paths = _struct_nd_paths_for(arg)
            if not paths:
                continue
            for chain in paths:
                v = arg
                for a in chain:
                    v = getattr(v, a)
                nd_ids.append(id(v))
        if nd_ids:
            args_hash = args_hash + tuple(nd_ids)
        # A Final-bearing argument disables this cache: serving a prior ``(count, key)`` would skip ``extract()`` and
        # its per-launch ``final_scalar_key`` revalidation. Final-free mappers are unaffected.
        cache_disabled = self._mapping_cache_disabled
        if cache_disabled is None:
            cache_disabled = self._mapping_cache_disabled = any(
                annotation_has_final_subtree(kernel_arg.annotation) for kernel_arg in self.arguments
            )

        if not cache_disabled:
            try:
                mapping_cache_tracker = self._mapping_cache_tracker[args_hash]
            except KeyError:
                pass
            if mapping_cache_tracker:
                return self._mapping_cache[args_hash]

        key = self.extract(raise_on_templated_floats, args)
        try:
            count = self.mapping[key]
        except KeyError:
            count = self.mapping[key] = len(self.mapping)

        # Skip the store too when disabled, so the read above always misses and every launch revalidates.
        if not cache_disabled:
            # Note that it is important to prepend the cache tracker with 'None' to avoid misclassifying no argument with
            # expired cache entry caused by deallocated argument.
            mapping_cache_tracker_: list[ReferenceType | None] = [None]

            # Clear the tracker (original invalidation) and also remove the stale
            # dict entries so they do not accumulate indefinitely.
            def _evict_callback(ref, _tracker=mapping_cache_tracker_, _self=self, _hash=args_hash):
                _tracker.clear()
                _self._mapping_cache.pop(_hash, None)
                _self._mapping_cache_tracker.pop(_hash, None)

            try:
                # Note that it is necessary to handle primitive types separately because it does not make sense to use
                # these arguments to track the lifetime of the corresponding cache entry and taking weakref of primitive
                # types if forbidden anyway.
                mapping_cache_tracker_ += [
                    ReferenceType(arg, _evict_callback) for arg in args if type(arg) not in _primitive_types
                ]
                self._mapping_cache_tracker[args_hash] = mapping_cache_tracker_
                self._mapping_cache[args_hash] = (count, key)
            except TypeError as e:
                warnings_helper.warn_once(f"{e}. Template mapper caching disabled.")

        return (count, key)
