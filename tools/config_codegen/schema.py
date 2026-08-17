"""Single source of truth for user-facing qd.init / CompileConfig options.

This module is PURE DATA: it imports nothing from ``quadrants`` and nothing
third-party, so it can be imported (a) by the C++ build-time code generator
(which runs before the quadrants extension exists) and (b) by the Sphinx docs
build. Each :class:`Option` co-locates an option's name, C++ type, Python type,
default value, and end-user description in exactly one place. Everything else
(the C++ struct fields and defaults, the nanobind bindings, and the user-guide
reference table) is generated from this list.

SPIKE SCOPE: only the commonly-tuned options are listed here. Phase 1 of the
migration (see DESIGN.md) extends this to every current CompileConfig field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Computed:
    """A default that is not a compile-time literal.

    The field is declared in the struct without an in-class initializer and is
    assigned ``cpp_expr`` in the generated constructor fragment, mirroring how
    ``compile_config.cpp`` sets these by hand today.
    """

    cpp_expr: str  # C++ expression assigned in the constructor
    doc: str  # human-readable description of the resulting default


@dataclass(frozen=True)
class Option:
    name: str  # field name == qd.init kwarg == QD_<UPPER> env var
    cpp_type: str  # e.g. "bool", "int", "std::size_t", "uint64_t", "std::string"
    py_type: str  # what the user passes: "bool", "int", "str"
    default: Any  # a literal (bool/int/float/str) or a Computed(...)
    doc: str  # the single end-user description

    def default_display(self) -> str:
        """Human-readable default for docs."""
        if isinstance(self.default, Computed):
            return self.default.doc
        if isinstance(self.default, bool):
            return "True" if self.default else "False"
        if isinstance(self.default, str):
            return '"' + self.default + '"'
        return str(self.default)


# The single source of truth. Order here is the order emitted everywhere.
OPTIONS: list[Option] = [
    Option(
        "offline_cache",
        "bool",
        "bool",
        True,
        "Whether compiled-kernel caches persist on disk and are reused across "
        "separate Python processes.",
    ),
    Option(
        "offline_cache_file_path",
        "std::string",
        "str",
        Computed(
            'get_repo_dir() + "qdcache"',
            "<cache dir>/qdcache (e.g. ~/.cache/quadrants/qdcache)",
        ),
        "Directory that holds the on-disk compilation cache.",
    ),
    Option(
        "cfg_optimization",
        "bool",
        "bool",
        True,
        "Run the control-flow-graph optimization, an internal compile-time pass "
        "that simplifies your kernel's branches and loops. Disabling it makes "
        "compilation much faster at a small runtime cost.",
    ),
    Option(
        "fast_math",
        "bool",
        "bool",
        True,
        "Allow IEEE-relaxed floating-point optimizations (e.g. fused "
        "multiply-add). Faster, but drops strict NaN/inf/signed-zero guarantees.",
    ),
    Option(
        "num_compile_threads",
        "int",
        "int",
        4,
        "Number of host threads used to compile kernels.",
    ),
    Option(
        "debug",
        "bool",
        "bool",
        False,
        "Turn on the full suite of correctness checks (field bounds, assertions, "
        "adstack overflow). Considerably slower; intended for development.",
    ),
    Option(
        "check_out_of_bound",
        "bool",
        "bool",
        False,
        "Enable the field out-of-bounds check on tensor indexing without turning "
        "on the rest of debug mode.",
    ),
    Option(
        "ad_stack_experimental_enabled",
        "bool",
        "bool",
        False,
        "Enable the reverse-mode autodiff pipeline for kernels with "
        "runtime-bounded loops (the adstack).",
    ),
    Option(
        "ad_stack_size",
        "int",
        "int",
        0,
        "Force every autodiff stack to exactly this many slots. 0 lets the "
        "launch-time sizer choose automatically.",
    ),
    Option(
        "ad_stack_sparse_threshold_bytes",
        "std::size_t",
        "int",
        100 * 1024 * 1024,
        "Byte cutoff below which the sparse adstack sizing path is skipped in "
        "favor of eager heap allocation.",
    ),
    Option(
        "external_metal_command_queue",
        "uint64_t",
        "int",
        0,
        "An MTLCommandQueue pointer (as an integer) to dispatch on instead of "
        "creating a new Metal queue. 0 means create a new queue.",
    ),
    Option(
        "external_metal_command_queue_is_torch_queue",
        "bool",
        "bool",
        False,
        "Set True when external_metal_command_queue is PyTorch MPS's own queue, "
        "to skip redundant interop synchronization.",
    ),
]
