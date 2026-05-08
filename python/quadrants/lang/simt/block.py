# type: ignore

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang import ops as _ops
from quadrants.lang.expr import make_expr_group
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.util import quadrants_scope
from quadrants.types.primitive_types import i32 as _i32


def arch_uses_spv(arch):
    return arch == _qd_core.vulkan or arch == _qd_core.metal


def sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.shared_array is not supported for arch {arch}")


def sync_all_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        # Hardware-fused barrier+reduction on NVPTX (`barrier.cta.red.and.aligned.all.sync`).
        return impl.call_internal("block_barrier_and_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        # AMDGPU and SPIR-V (Vulkan / Metal) emulate via shared memory + 2 barriers + an
        # atomic; see `_block_reduce_*_emulated` below for the pattern.
        return _block_reduce_all_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_all_nonzero is not supported for arch {arch}")


def sync_any_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_or_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        return _block_reduce_any_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_any_nonzero is not supported for arch {arch}")


def sync_count_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_count_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        return _block_reduce_count_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_count_nonzero is not supported for arch {arch}")


def mem_fence():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_mem_fence", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.mem_fence is not supported for arch {arch}")


def mem_sync():
    warnings.warn(
        "qd.simt.block.mem_sync() is deprecated; use qd.simt.block.mem_fence() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mem_fence()


def thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_thread_idx", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("localInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.thread_idx is not supported for arch {arch}")


def global_thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.get_runtime().compiling_callable.ast_builder().insert_thread_idx_expr()
    if arch_uses_spv(arch):
        return impl.call_internal("globalInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.global_thread_idx is not supported for arch {arch}")


class SharedArray:
    _is_quadrants_class = True

    def __init__(self, shape, dtype):
        if isinstance(shape, int):
            self.shape = (shape,)
        elif (isinstance(shape, tuple) or isinstance(shape, list)) and all(isinstance(s, int) for s in shape):
            self.shape = shape
        else:
            raise ValueError(
                f"qd.simt.block.shared_array shape must be an integer or a tuple of integers, but got {shape}"
            )
        if isinstance(dtype, impl.MatrixType):
            dtype = dtype.tensor_type
        self.dtype = dtype
        self.shared_array_proxy = impl.expr_init_shared_array(self.shape, dtype)

    @quadrants_scope
    def subscript(self, *indices):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return impl.Expr(
            ast_builder.expr_subscript(
                self.shared_array_proxy,
                make_expr_group(*indices),
                _qd_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )


# Shared-memory emulation of CUDA's hardware-fused barrier-with-reduction ops, used on backends
# that lack a direct equivalent (AMDGPU has no NVPTX `barrier.cta.red.*` analog; SPIR-V's
# `OpGroupNonUniform*` only operate at subgroup scope reliably across Vulkan + Metal).
#
# Pattern: lane 0 zeroes a 1-element shared `i32` -> block.sync() -> every thread atomically
# folds its predicate into the slot -> block.sync() -> every thread reads the broadcasted
# result. Costs 2 barriers + 1 atomic (vs. CUDA's hardware fast path of 1 barrier+reduction).
# Slower than the CUDA path but functionally equivalent and portable. Each call-site allocates
# a fresh `SharedArray` so multiple calls in the same kernel do not alias each other.
#
# All three emulations use `atomic_add` rather than `atomic_or` / `atomic_and`. We originally
# used the bitwise atomics on the `any` / `all` paths because the contributions are 0 / 1 and
# OR is conceptually cleaner, but Metal (via MoltenVK / SPIRV-Cross) silently no-ops
# `OpAtomicOr` on threadgroup memory in some configurations -- the SPIR-V op translates to MSL
# `atomic_fetch_or_explicit` on a `threadgroup atomic_int` and the resulting shader does not
# update the slot, so every thread reads back the initialiser value. `OpAtomicIAdd` does not
# hit the same issue, so we route every reduction through `atomic_add`. Correctness sketch for
# the new variants: `count` = number of threads contributing 1; `any_nonzero` = 1 iff `count > 0`;
# `all_nonzero` = 1 iff no thread contributed a zero indicator (i.e. `count == 0` over the
# zero-indicator predicate). The bool returns are implicitly cast to i32 by Quadrants' AST
# transformer (`ast_transformer.py::qd_ops.cast(..., return_type)`).
@_func
def _block_reduce_count_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    if predicate != 0:
        _ops.atomic_add(counter[0], 1)
    sync()
    return counter[0]


@_func
def _block_reduce_any_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    if predicate != 0:
        _ops.atomic_add(counter[0], 1)
    sync()
    return counter[0] != 0


@_func
def _block_reduce_all_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    if predicate == 0:
        _ops.atomic_add(counter[0], 1)
    sync()
    return counter[0] == 0
