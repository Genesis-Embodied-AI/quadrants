"""Single source of truth for user-facing qd.init / CompileConfig options.

Read this as a declarative table: every attribute of :class:`Options` is one
option. The attribute *name* is the option name (and the qd.init keyword, and
the ``QD_<UPPER>`` environment variable); the *annotation* is its type; the
assigned *value* is the default; and the *docstring* directly beneath it is the
end-user documentation. Nothing here imports ``quadrants`` or anything
third-party, so it is safe to import both at C++ build time (before the
extension exists) and at docs build time.

Type rules: annotate with a normal Python type (``bool``, ``int``, ``str``,
``float``). The C++ type is derived from it (bool->bool, int->int, str->
std::string, float->double). Python has a single ``int`` but C++ has several
integer widths, so when a field needs a specific one, override just the C++ side
with ``Annotated[int, Cpp("std::size_t")]`` -- the Python type stays ``int``.

SPIKE SCOPE: only the commonly-tuned options are listed here. Phase 1 of the
migration (see the B1 design doc,
perso_hugh/doc/config_self_documenting_design.md) extends this to every current
CompileConfig field.
"""

from typing import Annotated


class Cpp:
    """Annotation metadata: force a specific C++ type for an option.

    Use inside ``Annotated`` when the default Python->C++ mapping is not the one
    you want (typically to select an exact integer width).
    """

    def __init__(self, cpp_type: str):
        self.cpp_type = cpp_type


class Computed:
    """A default that is not a compile-time literal.

    The field is declared without an in-class initializer and assigned
    ``cpp_expr`` in the generated constructor fragment (mirroring how
    compile_config.cpp sets these by hand today). ``doc`` is the human-readable
    description of the resulting default, shown in the docs table.
    """

    def __init__(self, cpp_expr: str, doc: str):
        self.cpp_expr = cpp_expr
        self.doc = doc


class Options:
    """User-facing qd.init / CompileConfig options (see the module docstring)."""

    offline_cache: bool = True
    """Whether compiled-kernel caches persist on disk and are reused across
    separate Python processes."""

    offline_cache_file_path: str = Computed(
        'get_repo_dir() + "qdcache"',
        "<cache dir>/qdcache (e.g. ~/.cache/quadrants/qdcache)",
    )
    """Directory that holds the on-disk compilation cache."""

    cfg_optimization: bool = True
    """Run the control-flow-graph optimization, an internal compile-time pass
    that simplifies your kernel's branches and loops. Disabling it makes
    compilation much faster at a small runtime cost."""

    fast_math: bool = True
    """Allow IEEE-relaxed floating-point optimizations (e.g. fused
    multiply-add). Faster, but drops strict NaN/inf/signed-zero guarantees."""

    num_compile_threads: int = 4
    """Number of host threads used to compile kernels."""

    debug: bool = False
    """Turn on the full suite of correctness checks (field bounds, assertions,
    adstack overflow). Considerably slower; intended for development."""

    check_out_of_bound: bool = False
    """Enable the field out-of-bounds check on tensor indexing without turning
    on the rest of debug mode."""

    ad_stack_experimental_enabled: bool = False
    """Enable the reverse-mode autodiff pipeline for kernels with
    runtime-bounded loops (the adstack)."""

    ad_stack_size: int = 0
    """Force every autodiff stack to exactly this many slots. 0 lets the
    launch-time sizer choose automatically."""

    ad_stack_sparse_threshold_bytes: Annotated[int, Cpp("std::size_t")] = 100 * 1024 * 1024
    """Byte cutoff below which the sparse adstack sizing path is skipped in favor
    of eager heap allocation."""

    external_metal_command_queue: Annotated[int, Cpp("uint64_t")] = 0
    """An MTLCommandQueue pointer (as an integer) to dispatch on instead of
    creating a new Metal queue. 0 means create a new queue."""

    external_metal_command_queue_is_torch_queue: bool = False
    """Set True when external_metal_command_queue is PyTorch MPS's own queue, to
    skip redundant interop synchronization."""
