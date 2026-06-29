from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType

from quadrants.lang import impl
from quadrants.lang.impl import Program
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import is_data_oriented

from .._test_tools import warnings_helper
from ._kernel_types import ArgsHash
from ._template_mapper_hotpath import (
    _extract_arg,
    _primitive_types,
    _struct_nd_paths_for,
)

# Per-class disposition for the args_hash ndarray-id walk in ``TemplateMapper.lookup``: one of ``_SKIP`` (this class
# never contributes — non-data_oriented, or ``@qd.data_oriented(stable_members=True)``) or ``_PER_INSTANCE`` (delegate
# to ``_struct_nd_paths_for`` for a per-instance walk). The disposition depends only on type (data_oriented?
# stable_members?), so caching by class is correct. The *actual* path list is per-instance because @qd.data_oriented
# classes can have polymorphic attribute structure across instances (Genesis ``DataManager`` is the motivating case).
_arg_disposition: dict[type, object] = {}
_SKIP = object()
_PER_INSTANCE = object()


def _classify_disposition(arg: Any) -> object:
    """First-sighting per-class disposition for the args_hash walk. Returns ``_SKIP`` (no per-call walk for this
    class) or ``_PER_INSTANCE`` (delegate to ``_struct_nd_paths_for`` for a per-instance walk).

    ``_qd_stable_members`` here is a *launch-time perf hint only* (see ``@qd.data_oriented(stable_members=...)``).
    It promises that ndarray members are never reassigned, which lets us skip the per-call walk entirely. It does
    not affect fastcache key derivation."""
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

    def extract(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> Key:
        return tuple(
            [
                _extract_arg(raise_on_templated_floats, arg, kernel_arg.annotation, kernel_arg.name)
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
        # ``@qd.data_oriented`` containers can have their member ndarrays reassigned between calls on the same instance
        # (``state.x = other_ndarray``). The id(arg) alone does not capture that, so the spec-key cache below would
        # serve a stale entry and the new ndarray's dtype/ndim would be wrong. Fold the reachable ndarray ids into the
        # hash for the (small) set of arg positions that need it.
        #
        # The kernel's ``template_slot_locations`` already gives us the subset of arg positions annotated as
        # ``qd.template()`` — the only positions where a data_oriented container could appear (typed-dataclass args
        # carry a specific dataclass type by construction and a data_oriented class is never a dataclass). So we only
        # iterate ``template_slot_locations`` instead of all args (Genesis main kernel_step_1: 4 template positions
        # of 16 args; Genesis branch step_1/step_2: 4 of 4).
        #
        # For each candidate, ``_arg_disposition`` caches the per-class decision (skip vs walk-per-instance) and the
        # actual paths come from ``_struct_nd_paths_for`` (per-instance, stashed on ``arg._qd_nd_paths``). Per-instance
        # path caching is load-bearing for correctness — @qd.data_oriented classes can have polymorphic attribute
        # structure across instances (Genesis ``DataManager`` only allocates adjoint-cache members when
        # ``requires_grad=True``); a per-class cache populated from one instance can't safely be reused for another.
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
