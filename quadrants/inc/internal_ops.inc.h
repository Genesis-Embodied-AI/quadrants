PER_INTERNAL_OP(composite_extract_0)
PER_INTERNAL_OP(composite_extract_1)
PER_INTERNAL_OP(composite_extract_2)
PER_INTERNAL_OP(composite_extract_3)

PER_INTERNAL_OP(insert_triplet_f32)
PER_INTERNAL_OP(insert_triplet_f64)

PER_INTERNAL_OP(linear_thread_idx)
PER_INTERNAL_OP(block_thread_idx)

PER_INTERNAL_OP(test_stack)
PER_INTERNAL_OP(test_active_mask)
PER_INTERNAL_OP(test_shfl)
PER_INTERNAL_OP(test_list_manager)
PER_INTERNAL_OP(test_node_allocator)
PER_INTERNAL_OP(test_node_allocator_gc_cpu)
PER_INTERNAL_OP(do_nothing)
PER_INTERNAL_OP(refresh_counter)
PER_INTERNAL_OP(test_internal_func_args)

// SPIRV
PER_INTERNAL_OP(workgroupBarrier)
PER_INTERNAL_OP(workgroupMemoryBarrier)
PER_INTERNAL_OP(gridMemoryBarrier)
PER_INTERNAL_OP(localInvocationId)
PER_INTERNAL_OP(globalInvocationId)
PER_INTERNAL_OP(vkGlobalThreadIdx)
PER_INTERNAL_OP(subgroupBarrier)
PER_INTERNAL_OP(subgroupMemoryBarrier)
PER_INTERNAL_OP(subgroupElect)
PER_INTERNAL_OP(subgroupBroadcast)
PER_INTERNAL_OP(subgroupShuffle)
PER_INTERNAL_OP(subgroupShuffleDown)
PER_INTERNAL_OP(subgroupShuffleUp)
// Two ballot variants: u32 covers lanes [0, 32) (the most common case, used by `subgroup.ballot_first_n`); u64 covers
// the whole subgroup ([0, 32) on wave32 with the high 32 bits zero, [0, 64) on wave64).  See `subgroup.py` for the
// public API and the per-backend codegen (CUDA / AMDGPU / SPIR-V) for the lowering details.
PER_INTERNAL_OP(subgroupBallotU32)
PER_INTERNAL_OP(subgroupBallotU64)
// ``subgroupSize`` (the previous IR op) was removed: ``qd.simt.subgroup.group_size()`` now resolves at compile time via
// ``Program::subgroup_size()`` and returns a Python ``int`` (32 on CUDA, 64 on AMDGPU, device-probed on Vulkan /
// Metal), so the value is folded into the kernel IR as a literal on every backend instead of going through an
// internal-op- dispatched ``OpLoad`` / constant-fold on each codegen.  Net effect: one fewer op, identical generated
// code, and the value is usable as a ``qd.template()`` argument (which an IR op couldn't be).
PER_INTERNAL_OP(subgroupInvocationId)
// subgroupAdd / subgroupMul / subgroupMin / subgroupMax / subgroupAnd / subgroupOr / subgroupXor and subgroupInclusive*
// / subgroupExclusive* removed: use portable Python `subgroup.reduce_add_tiled(value, log2_size)` /
// `subgroup.inclusive_add_tiled` / `subgroup.exclusive_add_tiled` (and equivalents), implemented as `@qd.func`
// Hillis-Steele scans on top of `subgroupShuffleDown` / `subgroupShuffleUp` / `subgroupShuffle`, which work on all
// backends.
PER_INTERNAL_OP(spirv_clock_i64)

// CUDA
PER_INTERNAL_OP(cuda_clock_i64)
PER_INTERNAL_OP(block_barrier)
PER_INTERNAL_OP(block_barrier_and_i32)
PER_INTERNAL_OP(block_barrier_or_i32)
PER_INTERNAL_OP(block_barrier_count_i32)
PER_INTERNAL_OP(block_mem_fence)
PER_INTERNAL_OP(grid_mem_fence)
PER_INTERNAL_OP(cuda_all_sync_i32)
PER_INTERNAL_OP(cuda_any_sync_i32)
PER_INTERNAL_OP(cuda_uni_sync_i32)
PER_INTERNAL_OP(cuda_ballot_i32)
PER_INTERNAL_OP(cuda_shfl_sync_i32)
PER_INTERNAL_OP(cuda_shfl_sync_f32)
PER_INTERNAL_OP(cuda_shfl_up_sync_i32)
PER_INTERNAL_OP(cuda_shfl_up_sync_f32)
PER_INTERNAL_OP(cuda_shfl_down_sync_i32)
PER_INTERNAL_OP(cuda_shfl_down_sync_f32)
PER_INTERNAL_OP(cuda_shfl_xor_sync_i32)
PER_INTERNAL_OP(cuda_match_any_sync_i32)
PER_INTERNAL_OP(cuda_match_all_sync_i32)
PER_INTERNAL_OP(cuda_active_mask)
// Find-n-th-set-bit fast path for qd.math.fns, lowered to a single PTX `fns.b32` instruction via inline asm
// (`__nv_fns` is *not* in the slim libdevice.10.bc we ship). The portable / non-CUDA implementation lives in
// Python (`_fns_portable` in python/quadrants/math/mathimpl.py) and is a 32-iteration linear scan over bit
// positions, fully unrolled by each backend's lowering pipeline.
PER_INTERNAL_OP(cuda_fns_u32)
PER_INTERNAL_OP(warp_barrier)

// AMDGPU
PER_INTERNAL_OP(amdgpu_clock_i64)

// CPU
PER_INTERNAL_OP(cpu_clock_i64)
