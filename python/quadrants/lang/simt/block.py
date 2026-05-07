# type: ignore

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import make_expr_group
from quadrants.lang.util import quadrants_scope


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
        return impl.call_internal("block_barrier_and_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_all_nonzero is not supported for arch {arch}")


def sync_any_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_or_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_any_nonzero is not supported for arch {arch}")


def sync_count_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_count_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_count_nonzero is not supported for arch {arch}")


def mem_fence():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        # NB: lowers via the same internal op as `block.sync()` today (see
        # `block.md` support-table footnote and PR #637 in flight). The
        # `block_mem_fence` runtime symbol is wired up in
        # `quadrants/runtime/llvm/runtime_module/runtime.cpp` and patched in
        # `quadrants/runtime/llvm/llvm_context.cpp`, but the Python -> IR path
        # still emits `block_barrier` to preserve current convergence
        # semantics until #637 lands.
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch == _qd_core.amdgpu:
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
