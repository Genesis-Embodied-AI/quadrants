"""Per-offloaded-task compilation cache.

The LLVM codegen path (CPU / CUDA / AMDGPU) caches each offloaded task's compiled module in a per-Program in-memory
cache keyed on the task's IR (config + device caps + touched-SNode layout + re-id'd task body + autodiff mode). Editing
one offloaded task in a many-offload kernel makes the whole-kernel cache miss, but only the edited task recompiles: the
unchanged tasks are cloned out of the per-task cache. Asserted on counts (not wall time) exposed as
``kernel._primal.per_offload_cache_observations``.

``offline_cache=False`` so the on-disk whole-kernel cache never short-circuits codegen; the in-memory per-task cache is
what we exercise here. (A byte-identical kernel would instead hit the whole-kernel in-memory cache and never reach
per-task codegen, so it is not a useful probe of this layer.)
"""

import quadrants as qd

from tests import test_utils

# Distinct, deliberately-unusual constants so these tasks' cache keys do not collide with tasks other tests compiled.
# The tiers under test are content-keyed and program-scoped, so a test sharing kernel bodies with another would find
# its "cold" phase already warm.
_C = (61001.0, 61002.0, 61003.0, 61004.0)
_C_EDIT = 69999.0
_N = 8


# CUDA only: on CUDA each top-level loop lowers to one offloaded task and editing one leaves the others'
# task keys untouched, so the per-task cache is reused. On the CPU backend a loop can lower to an extra
# global-temp-using task whose key moves when an unrelated task changes, so editing one loop misses every task
# until the content-keyed global-temp offsets land (ref 1c). Reuse-on-edit is therefore only assertable per task
# on CUDA for standalone 1a; broaden this once 1c stabilises CPU offsets.
@test_utils.test(arch=[qd.cuda], offline_cache=False)
def test_per_offload_cache_one_construct_edit() -> None:
    # Four distinct top-level parallel loops => four independent groups of offloaded task(s) with distinct cache keys.
    @qd.kernel
    def kernel_a(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]
        for i in range(_N):
            x[i] += _C[2]
        for i in range(_N):
            x[i] += _C[3]

    # Exactly one task changed (third loop's constant): only that task recompiles, the other three hit the cache.
    @qd.kernel
    def kernel_edit_one(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]
        for i in range(_N):
            x[i] += _C_EDIT
        for i in range(_N):
            x[i] += _C[3]

    arr = qd.ndarray(qd.f32, shape=(_N,))

    # Cold compile: the per-task cache starts empty, so every task is recompiled and stored. The four structurally
    # identical loops each decompose into the same number of offloaded tasks -- one apiece on CUDA, more on some CPU
    # builds -- so the total is a multiple of four. Derive the per-loop task count rather than hard-coding one
    # backend's decomposition.
    kernel_a(arr)
    obs_a = kernel_a._primal.per_offload_cache_observations
    assert obs_a.constructs_total >= 4 and obs_a.constructs_total % 4 == 0, obs_a
    assert obs_a.constructs_recompiled == obs_a.constructs_total, obs_a
    assert obs_a.constructs_cache_hit == 0, obs_a
    per_loop = obs_a.constructs_total // 4

    # Editing one loop's constant leaves the other three loops byte-identical, so their tasks reuse the per-task
    # cache and only the edited loop's task(s) recompile.
    kernel_edit_one(arr)
    obs_edit = kernel_edit_one._primal.per_offload_cache_observations
    assert obs_edit.constructs_total == obs_a.constructs_total, obs_edit
    assert 1 <= obs_edit.constructs_recompiled <= per_loop, obs_edit
    assert obs_edit.constructs_cache_hit == obs_edit.constructs_total - obs_edit.constructs_recompiled, obs_edit
