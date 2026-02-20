# type: ignore

from quadrants._lib import core as _ti_core
from quadrants.lang import impl
from quadrants.lang.expr import make_expr_group
from quadrants.lang.util import quadrants_scope


def arch_uses_spv(arch):
    return arch == _ti_core.vulkan or arch == _ti_core.metal


def sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda or arch == _ti_core.amdgpu:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.shared_array is not supported for arch {arch}")


def sync_all_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier_and_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_all_nonzero is not supported for arch {arch}")


def sync_any_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier_or_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_any_nonzero is not supported for arch {arch}")


def sync_count_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier_count_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_count_nonzero is not supported for arch {arch}")


def mem_sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.mem_sync is not supported for arch {arch}")


def thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch_uses_spv(arch):
        return impl.call_internal("localInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.thread_idx is not supported for arch {arch}")


def global_thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _ti_core.cuda or _ti_core.amdgpu:
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
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )
