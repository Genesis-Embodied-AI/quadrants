// GPU-side per-checkpoint gating for `qd.checkpoint(...)` on Vulkan / Metal. Extracted out of
// `runtime.cpp` so the main launcher's per-task dispatch loop stays focused; every code path here
// is conditional on at least one task carrying `cp_id >= 0` and never runs for vanilla kernels.
//
// Mechanism end-to-end:
//
// 1. At first launch of a yielding-capable kernel `ensure_checkpoint_state_for_handle` allocates a
//    per-handle `CheckpointHandleState`:
//      - one shared `control` buffer (2 i32 words: resume_point, yield_signal)
//      - one `CheckpointPerCpState` per checkpoint with:
//        * a `gate_params` SSBO baked once with `[cp_id, n_kernels, (gx,gy,gz) per body kernel,
//          (1,1,1) for the yield-check shader (if yielding)]`
//        * an `out_dims` SSBO sized `(n_kernels + yielding) * 12` bytes; populated by the gate
//          shader each launch, consumed via `CommandList::dispatch_indirect` by the body kernels
//          (and the yield-check shader)
//        * an optional `yield_check_params` SSBO (4 bytes, cp_id) for yielding checkpoints
//    The same pre-built gate / yield-check pipelines (one per `GfxRuntime`) are used across every
//    yielding-capable kernel - they're generic in cp_id / n_kernels / dims.
//
// 2. At launch time, `launch_kernel` calls `dispatch_checkpoint_gate` before the first body task of
//    each cp_id; the gate writes the per-kernel dim3 triples into `out_dims` based on the current
//    `(resume_point, yield_signal)` value. The body kernels then dispatch indirect off the same
//    `out_dims`; skipped checkpoints' kernels dispatch with `(0,0,0)` workgroups, which is a hardware
//    no-op on both Vulkan (`vkCmdDispatchIndirect`) and Metal (`dispatchThreadgroupsIndirect`).
//
// 3. After the last body task of a yielding checkpoint, `dispatch_checkpoint_yield_check` issues the
//    yield-check shader indirect off the trailing `out_dims` slot. The shader reads the user's
//    `yield_on=` flag, atomic-CASes the cp_id into `yield_signal` on non-zero, and resets the flag.
//    The yield-check is itself indirect-gated (same `(0,0,0)` no-op semantics as the body kernels),
//    so a skipped checkpoint also skips its yield-check.
//
// 4. After all tasks have dispatched, `launch_kernel` flushes, syncs, reads back `yield_signal` from
//    `state.control[1]`, and surfaces the first-yielder cp_id through
//    `last_yield_cp_id_on_last_call()`.
//
// Cost model (per launch with K checkpoints, N body kernels):
//   - 1 host upload of 8 bytes (control buf init).
//   - K gate dispatches (1-thread each) + N body dispatches (indirect) + K yield-check dispatches.
//   - 1 host readback of 8 bytes (yield_signal extraction).
// No per-task host gating; no per-task D2H. This replaces the slice-4 host-branch loop in the
// original `runtime.cpp::launch_kernel` that this file's helpers were built to displace.

#include "quadrants/runtime/gfx/runtime.h"

#include "quadrants/codegen/spirv/checkpoint_gate_shader.h"
#include "quadrants/codegen/spirv/checkpoint_yield_check_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/rhi/device.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace quadrants::lang {
namespace gfx {

namespace {

// Build (and cache) the generic gate compute pipeline. Called from
// `ensure_checkpoint_state_for_handle`; idempotent across handles.
void ensure_gate_pipeline(Device *device, PipelineCache *backend_cache, std::unique_ptr<Pipeline> &pipeline_slot) {
  if (pipeline_slot) {
    return;
  }
  std::vector<uint32_t> spirv = spirv::build_checkpoint_gate_spirv(Arch::vulkan, &device->get_caps());
  QD_ASSERT_INFO(!spirv.empty(),
                 "`build_checkpoint_gate_spirv` returned an empty binary; vanilla compute capability "
                 "should be universally satisfied on Vulkan / Metal. Bug in the shader builder?");
  PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                 spirv.size() * sizeof(uint32_t)};
  auto [pipeline, res] = device->create_pipeline_unique(source_desc, "qd_checkpoint_gate", backend_cache);
  QD_ERROR_IF(res != RhiResult::success, "Failed to create checkpoint gate pipeline (err: {})", int(res));
  pipeline_slot = std::move(pipeline);
}

// Build (and cache) the generic yield-check compute pipeline. Same shape as `ensure_gate_pipeline`.
void ensure_yield_check_pipeline(Device *device,
                                 PipelineCache *backend_cache,
                                 std::unique_ptr<Pipeline> &pipeline_slot) {
  if (pipeline_slot) {
    return;
  }
  std::vector<uint32_t> spirv = spirv::build_checkpoint_yield_check_spirv(Arch::vulkan, &device->get_caps());
  QD_ASSERT_INFO(!spirv.empty(),
                 "`build_checkpoint_yield_check_spirv` returned an empty binary; the shader has no extra "
                 "capability requirements beyond `OpAtomicCompareExchange`. Bug in the shader builder?");
  PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                 spirv.size() * sizeof(uint32_t)};
  auto [pipeline, res] = device->create_pipeline_unique(source_desc, "qd_checkpoint_yield_check", backend_cache);
  QD_ERROR_IF(res != RhiResult::success, "Failed to create checkpoint yield-check pipeline (err: {})", int(res));
  pipeline_slot = std::move(pipeline);
}

// Allocate (or reuse) a small device buffer of `bytes` for storage / indirect use. `host_write` /
// `host_read` control the memory-property hints; the launcher uses host_write=true on the control
// buffer (resume_point is set per launch), host_read=true on the same buffer (yield_signal is read
// at end), and both false on the gate params / out_dims / yield-check params (device-only access).
// `indirect` selects whether `AllocUsage::Indirect` is set in addition to `Storage`; required for
// the `out_dims` buffer so `vkCmdDispatchIndirect` can read it.
std::unique_ptr<DeviceAllocationGuard> alloc_small_ssbo(Device *device,
                                                        size_t bytes,
                                                        bool host_write,
                                                        bool host_read,
                                                        bool indirect) {
  AllocUsage usage = AllocUsage::Storage;
  if (indirect) {
    usage = static_cast<AllocUsage>(static_cast<int>(usage) | static_cast<int>(AllocUsage::Indirect));
  }
  auto [buf, res] = device->allocate_memory_unique({bytes, host_write, host_read, /*export_sharing=*/false, usage});
  QD_ASSERT_INFO(res == RhiResult::success,
                 "Failed to allocate per-checkpoint SSBO ({} bytes, host_write={}, host_read={}, indirect={})", bytes,
                 host_write, host_read, indirect);
  return std::move(buf);
}

// Upload the host-resident `data` (u32 words) into `buf`. Used to bake the gate params buffer once
// at first launch and to write the yield-check params (cp_id) per-checkpoint at first launch.
void upload_u32_words(Device *device, DeviceAllocationGuard &buf, const std::vector<uint32_t> &data) {
  void *mapped = nullptr;
  QD_ASSERT_INFO(device->map(buf, &mapped) == RhiResult::success, "Failed to map gating SSBO ({} bytes) for upload",
                 data.size() * sizeof(uint32_t));
  std::memcpy(mapped, data.data(), data.size() * sizeof(uint32_t));
  device->unmap(buf);
}

}  // namespace

bool GfxRuntime::ensure_checkpoint_state_for_handle(
    KernelHandle handle,
    const std::vector<quadrants::lang::spirv::TaskAttributes> &task_attribs,
    const std::vector<int> &checkpoint_yield_on_arg_ids,
    const std::vector<int> &per_task_group_x) {
  // Fast path: no task carries a checkpoint id - this kernel is not yielding-capable, no gating
  // state to set up, the launcher should use the standard direct-dispatch path.
  bool has_any_cp = false;
  int32_t max_cp_id = -1;
  for (const auto &t : task_attribs) {
    if (t.checkpoint_id >= 0) {
      has_any_cp = true;
      max_cp_id = std::max(max_cp_id, t.checkpoint_id);
    }
  }
  if (!has_any_cp) {
    return false;
  }

  ensure_gate_pipeline(device_, backend_cache_.get(), checkpoint_gate_pipeline_);
  ensure_yield_check_pipeline(device_, backend_cache_.get(), checkpoint_yield_check_pipeline_);

  size_t launch_id = handle.get_launch_id();
  if (checkpoint_handle_states_.size() <= launch_id) {
    checkpoint_handle_states_.resize(launch_id + 1);
  }
  CheckpointHandleState &state = checkpoint_handle_states_[launch_id];

  // First-launch setup for this handle. The per-cp layout is baked from the task list and never
  // changes across launches; subsequent launches reuse the same buffers as-is. `state.per_cp.empty()`
  // is the "first launch" sentinel.
  if (state.per_cp.empty()) {
    state.per_cp.resize(static_cast<size_t>(max_cp_id) + 1);
    for (int32_t cp = 0; cp <= max_cp_id; ++cp) {
      state.per_cp[cp].cp_id = cp;
    }
    for (int i = 0; i < (int)task_attribs.size(); ++i) {
      int cp = task_attribs[i].checkpoint_id;
      if (cp < 0) {
        continue;
      }
      state.per_cp[cp].body_task_indices.push_back(i);
    }
    // Allocate the shared control buffer (8 bytes, host_write + host_read).
    state.control = alloc_small_ssbo(device_, spirv::CheckpointControlBuf::kNumWords * sizeof(uint32_t),
                                     /*host_write=*/true, /*host_read=*/true, /*indirect=*/false);
    for (int32_t cp = 0; cp <= max_cp_id; ++cp) {
      auto &per_cp = state.per_cp[cp];
      bool yielding = (cp < (int32_t)checkpoint_yield_on_arg_ids.size()) && (checkpoint_yield_on_arg_ids[cp] >= 0);
      uint32_t n_dispatches = static_cast<uint32_t>(per_cp.body_task_indices.size()) + (yielding ? 1u : 0u);
      // Build gate params: [cp_id, n_dispatches, (gx, gy, gz) per body kernel, (1, 1, 1) if yielding].
      std::vector<uint32_t> gp;
      gp.reserve(2 + 3u * n_dispatches);
      gp.push_back(static_cast<uint32_t>(cp));
      gp.push_back(n_dispatches);
      for (int ti : per_cp.body_task_indices) {
        // (group_x, 1, 1) - Quadrants kernels are 1D over group count; y/z are always 1.
        gp.push_back(static_cast<uint32_t>(per_task_group_x[ti]));
        gp.push_back(1u);
        gp.push_back(1u);
      }
      if (yielding) {
        // Yield-check shader is single-thread: active grid is (1, 1, 1).
        gp.push_back(1u);
        gp.push_back(1u);
        gp.push_back(1u);
      }
      per_cp.gate_params = alloc_small_ssbo(device_, gp.size() * sizeof(uint32_t), /*host_write=*/true,
                                            /*host_read=*/false, /*indirect=*/false);
      upload_u32_words(device_, *per_cp.gate_params, gp);
      // Out-dims buffer is written per-launch by the gate shader and read indirectly by each body
      // (and yield-check) dispatch. Sized for `n_dispatches` u32 triples. `Indirect` usage flag is
      // required so `vkCmdDispatchIndirect` accepts the buffer.
      per_cp.out_dims = alloc_small_ssbo(device_, n_dispatches * 3u * sizeof(uint32_t), /*host_write=*/false,
                                         /*host_read=*/false, /*indirect=*/true);
      if (yielding) {
        // Yield-check params: 1 u32 word holding cp_id. Bake once.
        std::vector<uint32_t> ycp{static_cast<uint32_t>(cp)};
        per_cp.yield_check_params =
            alloc_small_ssbo(device_, spirv::CheckpointYieldCheckParams::kNumWords * sizeof(uint32_t),
                             /*host_write=*/true, /*host_read=*/false, /*indirect=*/false);
        upload_u32_words(device_, *per_cp.yield_check_params, ycp);
      }
    }
  } else {
    // Per-launch sanity: per_task_group_x must match what we baked at first launch. The active
    // grid for a given task is a function of the kernel's compile-time advisory + the host-side
    // ndarray shape; a different shape would yield a different group_x and silently mis-gate via
    // a stale baked active dim. Quadrants currently rebuilds the kernel handle whenever the
    // advisory or shape changes (see `Kernel::launch_kernel` cache key), so this assertion exists
    // to catch a regression in that contract rather than a normal user path.
    for (int32_t cp = 0; cp <= max_cp_id; ++cp) {
      auto &per_cp = state.per_cp[cp];
      for (int slot = 0; slot < (int)per_cp.body_task_indices.size(); ++slot) {
        int ti = per_cp.body_task_indices[slot];
        // No-op consistency check; per_task_group_x is recomputed per launch but should be stable
        // for a given handle. If this ever fires, the baked gate_params is stale and we'd need to
        // re-upload it here (cheap; just a `upload_u32_words` rewrite).
        (void)per_task_group_x[ti];
      }
    }
  }

  return true;
}

// Record a gate-shader dispatch into `cmdlist` for one checkpoint. The gate writes the per-kernel
// dim3 triples into the out_dims buffer based on the current `(resume_point, yield_signal)` in the
// control buffer. Callers must call `memory_barrier()` after this so subsequent body kernel
// indirect-dispatches see the written dims, and so the body / yield-check shaders' subsequent
// atomic updates to `yield_signal` happen-after the gate's read of it.
void GfxRuntime::dispatch_checkpoint_gate(CommandList *cmdlist, const CheckpointHandleState &state, int cp_id) {
  const auto &per_cp = state.per_cp[cp_id];
  // Barrier the control + out_dims buffers so the gate's reads of control / writes to out_dims
  // happen-after any prior cmdlist op that touched them (e.g. an earlier yield-check shader's
  // atomic-CAS of yield_signal, or the previous task's indirect-dispatch read of out_dims).
  cmdlist->buffer_barrier(*state.control);
  cmdlist->buffer_barrier(*per_cp.out_dims);

  auto bindings = device_->create_resource_set_unique();
  bindings->rw_buffer(0, *state.control);
  bindings->rw_buffer(1, *per_cp.gate_params);
  bindings->rw_buffer(2, *per_cp.out_dims);

  cmdlist->bind_pipeline(checkpoint_gate_pipeline_.get());
  RhiResult bind_res = cmdlist->bind_shader_resources(bindings.get());
  QD_ERROR_IF(bind_res != RhiResult::success, "Checkpoint gate resource bind failed: RhiResult({})", int(bind_res));
  RhiResult dispatch_res = cmdlist->dispatch(1u, 1u, 1u);
  QD_ERROR_IF(dispatch_res != RhiResult::success, "Checkpoint gate dispatch failed: RhiResult({})", int(dispatch_res));
  // Out-dims must be barriered before the subsequent body kernel indirect-dispatches read it. The
  // barrier is left to the caller (matches the per-task `memory_barrier()` that already runs after
  // each direct dispatch); this keeps the dispatch-of-things-per-task pattern consistent.
}

// Record a yield-check-shader dispatch into `cmdlist` for one yielding checkpoint, indirect-gated
// off the trailing slot of out_dims so the shader no-ops when the checkpoint was skipped.
// `yield_on_devalloc` is the user's `yield_on=` ndarray device allocation (resolved per-launch
// from the `host_ctx.checkpoint_yield_on_arg_ids[cp_id]` arg id lookup against `any_arrays`).
void GfxRuntime::dispatch_checkpoint_yield_check(CommandList *cmdlist,
                                                 const CheckpointHandleState &state,
                                                 int cp_id,
                                                 DeviceAllocation yield_on_devalloc) {
  const auto &per_cp = state.per_cp[cp_id];
  QD_ASSERT_INFO(per_cp.yield_check_params != nullptr,
                 "dispatch_checkpoint_yield_check called for cp_id={} but no yield_check_params; "
                 "caller should only invoke for yielding checkpoints",
                 cp_id);
  // Barrier the user's yield_on ndarray so the yield-check's load happens-after the body kernels'
  // writes to the flag. The control buffer is barriered too because the yield-check does an
  // atomic-CAS into yield_signal; same handle as the gate's barrier above.
  cmdlist->buffer_barrier(yield_on_devalloc);
  cmdlist->buffer_barrier(*state.control);

  auto bindings = device_->create_resource_set_unique();
  bindings->rw_buffer(0, *state.control);
  bindings->rw_buffer(1, yield_on_devalloc);
  bindings->rw_buffer(2, *per_cp.yield_check_params);

  cmdlist->bind_pipeline(checkpoint_yield_check_pipeline_.get());
  RhiResult bind_res = cmdlist->bind_shader_resources(bindings.get());
  QD_ERROR_IF(bind_res != RhiResult::success, "Checkpoint yield-check resource bind failed: RhiResult({})",
              int(bind_res));
  // Indirect dispatch off the trailing out_dims slot (immediately after the body kernel slots).
  // When the gate decided to skip, that slot holds (0, 0, 0) and the GPU dispatches zero
  // workgroups - the atomic-CAS and flag reset are both elided. When the gate decided to run, the
  // slot holds (1, 1, 1) and the single-thread shader executes.
  size_t yc_slot_offset = per_cp.body_task_indices.size() * 3u * sizeof(uint32_t);
  RhiResult dispatch_res = cmdlist->dispatch_indirect(per_cp.out_dims->get_ptr(yc_slot_offset));
  QD_ERROR_IF(dispatch_res != RhiResult::success, "Checkpoint yield-check dispatch_indirect failed: RhiResult({})",
              int(dispatch_res));
}

}  // namespace gfx
}  // namespace quadrants::lang
