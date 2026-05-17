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


def _collect_data_oriented_nd_ids(arg: Any, out: list) -> None:
    """Append ``id(ndarray)`` for every ndarray reachable from ``arg``, using the per-class path cache in
    ``_template_mapper_hotpath._struct_nd_paths_for`` so the first call walks ``vars(arg)`` once and subsequent calls
    are just ``getattr`` chains. Empty path list short-circuits with zero work — critical for genesis's
    ``@qd.data_oriented`` Solver passed as ``self`` to every kernel.
    """
    for chain in _struct_nd_paths_for(arg):
        v = arg
        for a in chain:
            v = getattr(v, a)
        out.append(id(v))


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

        # Keep track of taichi runtime to automatically clear cache if destroyed
        if self._prog_weakref is None:
            prog = impl.get_runtime().prog
            assert prog is not None
            self._prog_weakref = ReferenceType(prog, partial(_destroy_callback, ReferenceType(self)))
        else:
            # Since we already store a weak reference to taichi program, it is much faster to use it rather than
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
        # hash. No-op for data_oriented containers that hold no ndarrays — the walker returns an empty list. See
        # ``_collect_data_oriented_nd_ids``.
        nd_ids: list = []
        for arg in args:
            if is_data_oriented(arg):
                # Opt-out: classes that promise their ndarray members never reassign between calls
                # (set ``_qd_stable_members = True`` on the class, or use
                # ``@qd.data_oriented(stable_members=True)``) skip the per-call walk. The spec key
                # then falls back to weakref(arg) alone — see _extract_arg's data_oriented branch.
                # Saves ~1.1-1.5 us per kernel call on Genesis-style containers. Reassigning a
                # member on a stable-marked instance is silently undefined behaviour: the cached
                # kernel for the prior shape will be reused.
                if type(arg).__dict__.get("_qd_stable_members"):
                    continue
                _collect_data_oriented_nd_ids(arg, nd_ids)
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
