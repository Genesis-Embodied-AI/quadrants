// On-device adstack `SizeExpr` sizer dispatch for SPIR-V backends. Extracted out of `runtime.cpp` so
// `GfxRuntime::launch_kernel` stays focused on the main-kernel record/submit flow; every code path here is
// conditional on at least one task having adstack allocas and never runs for forward-only kernels.
//
// Mechanism end-to-end:
// 1. Encode each task's `SerializedSizeExpr` trees into a flat device bytecode (see
//    `quadrants/program/adstack_size_expr_eval.cpp::encode_adstack_size_expr_device_bytecode_for_spirv`).
// 2. Upload the concatenated blob into a grow-on-demand shared bytecode buffer.
// 3. Per task: allocate a fresh metadata output buffer, bind (bytecode, metadata, args_buffer) to the sizer
//    pipeline, dispatch 1x1x1.
// 4. `submit_synced` the sizer cmdlist, then read back each per-task metadata buffer into the returned
//    `PerTaskAdStackRuntime` vector. The main-kernel cmdlist has not been opened yet, so there is no
//    descriptor-bind reentrancy against any in-flight launch.
//
// All pipelined state lives on `GfxRuntime` (the sizer pipeline, the bytecode scratch buffer and its size,
// the deferred-free `ctx_buffers_`), so this translation unit is purely a method implementation.

#include "quadrants/runtime/gfx/runtime.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace gfx {

std::vector<PerTaskAdStackRuntime> GfxRuntime::publish_adstack_metadata_spirv(
    LaunchContextBuilder &host_ctx,
    DeviceAllocationGuard *args_buffer,
    const std::unordered_map<int, DeviceAllocation> &ndarray_allocs,
    const std::vector<quadrants::lang::spirv::TaskAttributes> &task_attribs,
    const std::string &kernel_name) {
  std::vector<PerTaskAdStackRuntime> per_task_ad_stack(task_attribs.size());
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    per_task_ad_stack[ti].stride_float = task_attribs[ti].ad_stack.per_thread_stride_float_compile_time;
    per_task_ad_stack[ti].stride_int = task_attribs[ti].ad_stack.per_thread_stride_int_compile_time;
  }

  std::vector<size_t> adstack_task_indices;
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    if (!task_attribs[ti].ad_stack.allocas.empty())
      adstack_task_indices.push_back(ti);
  }

  if (adstack_task_indices.empty()) {
    return per_task_ad_stack;
  }

  QD_ASSERT_INFO(program_impl_ != nullptr && program_impl_->program != nullptr,
                 "GfxRuntime::launch_kernel: `ProgramImpl::program` back-reference not set; cannot "
                 "encode AdStack SizeExpr bytecode. Ensure GfxProgramImpl passes `program_impl = this` "
                 "into `GfxRuntime::Params`.");
  QD_ERROR_IF(!device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
                  !device_->get_caps().get(DeviceCapability::spirv_has_int64) ||
                  !device_->get_caps().get(DeviceCapability::spirv_has_int8) ||
                  !device_->get_caps().get(DeviceCapability::spirv_has_int16),
              "GfxRuntime::launch_kernel: the on-device adstack SizeExpr sizer requires the Physical Storage "
              "Buffer, Int64, Int8, and Int16 SPIR-V capabilities, but at least one is missing on this device. "
              "There is no correct host-eval fallback for `qd.ndarray`-backed reverse-mode state on a GPU-private "
              "backend; the shader must run on-device or the kernel's adstack sizing is garbage. Use a backend "
              "that advertises all four caps (e.g. Metal on Apple Silicon, Vulkan 1.2+ with "
              "`VK_KHR_buffer_device_address` and `VK_KHR_shader_float16_int8`), or run the workload on the LLVM "
              "runtime (CPU / CUDA / AMDGPU).");

  // Build the sizer pipeline on first use.
  if (!adstack_sizer_pipeline_) {
    std::vector<uint32_t> spirv = spirv::build_adstack_sizer_spirv(Arch::vulkan, &device_->get_caps());
    QD_ASSERT_INFO(!spirv.empty(),
                   "`build_adstack_sizer_spirv` returned an empty binary despite the PSB+Int64+Int8+Int16 "
                   "capability check passing; bug in the shader builder's capability gating.");
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                   spirv.size() * sizeof(uint32_t)};
    auto [pipeline, res] = device_->create_pipeline_unique(source_desc, "adstack_sizer", backend_cache_.get());
    QD_ERROR_IF(res != RhiResult::success, "Failed to create pipeline for the adstack SizeExpr sizer shader (err: {})",
                int(res));
    adstack_sizer_pipeline_ = std::move(pipeline);
  }

  // Encode per-task bytecodes and compute per-task metadata sizes. Each task's starting offset inside the
  // shared bytecode buffer must satisfy Vulkan's `minStorageBufferOffsetAlignment` because that offset flows
  // verbatim into `VkDescriptorBufferInfo::offset` through `bindings->rw_buffer(...)` below; raw
  // `sizeof(AdStackSizeExpr*)` arithmetic is only 4-byte aligned, which trips VUID-02999 on NVIDIA / Intel
  // desktop / MoltenVK (16 B) and Adreno (64 B). RHI does not expose the queried minimum, so pick 256 B -
  // the largest cap we expect in the wild (older NVIDIA) - as a safe fixed rounding.
  constexpr size_t kDescriptorOffsetAlignment = 256;
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  std::vector<std::vector<uint8_t>> per_task_bytecodes(adstack_task_indices.size());
  std::vector<size_t> per_task_bytecode_offsets(adstack_task_indices.size());
  std::vector<size_t> per_task_metadata_bytes(adstack_task_indices.size());
  size_t total_bytecode_bytes = 0;
  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    size_t ti = adstack_task_indices[k];
    per_task_bytecodes[k] = encode_adstack_size_expr_device_bytecode_for_spirv(task_attribs[ti].ad_stack,
                                                                               program_impl_->program, &host_ctx);
    per_task_bytecode_offsets[k] = align_up(total_bytecode_bytes, kDescriptorOffsetAlignment);
    total_bytecode_bytes = per_task_bytecode_offsets[k] + per_task_bytecodes[k].size();
    per_task_metadata_bytes[k] = (2u + 2u * task_attribs[ti].ad_stack.allocas.size()) * sizeof(uint32_t);
  }

  // Grow the shared bytecode scratch buffer if the concatenated blob outgrew it. Amortised doubling so
  // steady-state launches see no allocation traffic.
  if (!adstack_sizer_bytecode_buffer_ || adstack_sizer_bytecode_buffer_size_ < total_bytecode_bytes) {
    size_t new_size = std::max(total_bytecode_bytes, 2 * adstack_sizer_bytecode_buffer_size_);
    auto [buf, res] = device_->allocate_memory_unique(
        {new_size, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer bytecode buffer (size={})", new_size);
    if (adstack_sizer_bytecode_buffer_)
      ctx_buffers_.push_back(std::move(adstack_sizer_bytecode_buffer_));
    adstack_sizer_bytecode_buffer_ = std::move(buf);
    adstack_sizer_bytecode_buffer_size_ = new_size;
  }
  {
    void *mapped = nullptr;
    RhiResult map_res = device_->map_range(adstack_sizer_bytecode_buffer_->get_ptr(0), total_bytecode_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack sizer bytecode buffer for upload");
    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      std::memcpy(reinterpret_cast<char *>(mapped) + per_task_bytecode_offsets[k], per_task_bytecodes[k].data(),
                  per_task_bytecodes[k].size());
    }
    device_->unmap(*adstack_sizer_bytecode_buffer_);
  }

  // Per-task metadata output buffers. Defer-freed via `ctx_buffers_` after readback so any in-flight writes
  // from the just-synced sizer dispatch can finish draining through the normal cmdlist cleanup path.
  std::vector<DeviceAllocationUnique> per_task_metadata_allocs(adstack_task_indices.size());
  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    auto [buf, res] =
        device_->allocate_memory_unique({per_task_metadata_bytes[k], /*host_write=*/false,
                                         /*host_read=*/true, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer output buffer (size={})",
                   per_task_metadata_bytes[k]);
    per_task_metadata_allocs[k] = std::move(buf);
  }

  // Force visibility of prior device-side writes (accessor-kernel snode writes, user-side ndarray h2d blits,
  // adstack heap grow-path `buffer_fill`s) to the sizer's `PhysicalStorageBuffer` loads. An intra-cmdlist
  // `memory_barrier()` is not sufficient on MoltenVK: the Metal command encoder backs PSB loads through the
  // device-address path, which bypasses the descriptor-bound cache a prior accessor kernel's `submit_synced`
  // flushed via `vkQueueWaitIdle`. `flush()` drains any pending `current_cmdlist_` the outer launcher may have
  // left behind (the one the main-kernel dispatch below will reuse), and `vkDeviceWaitIdle` pairs with the
  // queue-level fence semantics the MoltenVK driver honours for cross-memory-path coherency. Symptom without
  // this: `FieldLoad(n_iter) -> 0` instead of the live field value, then an adstack overflow at the next
  // `qd.sync()`.
  flush();
  device_->wait_idle();
  auto [sizer_cmdlist, cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(cmdlist_res == RhiResult::success, "Failed to create adstack sizer cmdlist");

  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    auto bindings = device_->create_resource_set_unique();
    // All three bindings are declared as `StorageBuffer` in the sizer shader (`buffer_argument` lowers to
    // a storage SSBO in the SPIR-V IR, not a uniform). Vulkan distinguishes uniform and storage buffers
    // via distinct `VkDescriptorType` values - binding these slots via `buffer()` (uniform) produces a
    // descriptor set layout that doesn't match the pipeline's, and `bind_shader_resources` returns
    // `invalid_usage` with "Layout mismatch". Use `rw_buffer` across the board so the descriptor types
    // match; the shader is disciplined about not writing slots 0 and 2, so granting write-capable
    // descriptors there is harmless. The backing `args_buffer` is allocated with
    // `Uniform | Storage` at `runtime.cpp::launch_kernel`'s allocation site so the storage-buffer bind on
    // slot 2 satisfies VUID-VkDescriptorBufferInfo-buffer-02999.
    bindings->rw_buffer(0, adstack_sizer_bytecode_buffer_->get_ptr(per_task_bytecode_offsets[k]),
                        per_task_bytecodes[k].size());
    bindings->rw_buffer(1, *per_task_metadata_allocs[k]);
    // Buffer(2) holds the outer kernel's arg buffer: the sizer reads ndarray data pointers out of it to
    // resolve `ExternalTensorRead` nodes. A kernel can legitimately have adstack allocas *without* any
    // ndarray-backed inputs (e.g. adstacks sized from a field value only, not from an ndarray shape),
    // in which case `args_buffer` is null and no ExternalTensorRead nodes ever get interpreted - so any
    // valid allocation is safe to bind here. Fall back to the bytecode buffer rather than plumbing a
    // conditional null-binding path that every RHI backend would need to support.
    bindings->rw_buffer(2, args_buffer ? *args_buffer : *adstack_sizer_bytecode_buffer_);

    sizer_cmdlist->bind_pipeline(adstack_sizer_pipeline_.get());
    RhiResult bind_res = sizer_cmdlist->bind_shader_resources(bindings.get());
    QD_ERROR_IF(bind_res != RhiResult::success, "Sizer resource binding error: RhiResult({})", int(bind_res));
    // Mark every ndarray data buffer resident for this dispatch. The sizer reads each `ExternalTensorRead`
    // via a `buffer_reference` / PSB load against a u64 pointer stored in the kernel arg buffer, which
    // bypasses Metal's descriptor-based resource tracking - without an explicit `useResource:` hint the
    // Apple7 GPU family (M1) returns zero for those loads and the shader sizes every MOR-over-ETR to zero,
    // tripping an `Adstack overflow` at the next `qd.sync()`. The main kernel dispatch already calls
    // `track_physical_buffer` on the same set at `runtime.cpp`'s pre-dispatch block; this mirrors it for
    // the sizer. Backends that do not need residency hints (Vulkan/MoltenVK, LLVM-native) no-op the base
    // `track_physical_buffer` and pay nothing.
    if (device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
      for (const auto &[arg_id, alloc] : ndarray_allocs) {
        sizer_cmdlist->track_physical_buffer(alloc);
      }
    }
    RhiResult dispatch_res = sizer_cmdlist->dispatch(1, 1, 1);
    QD_ERROR_IF(dispatch_res != RhiResult::success, "Sizer dispatch error: RhiResult({})", int(dispatch_res));
    sizer_cmdlist->buffer_barrier(*per_task_metadata_allocs[k]);
  }
  device_->get_compute_stream()->submit_synced(sizer_cmdlist.get());

  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    size_t ti = adstack_task_indices[k];
    auto &rt = per_task_ad_stack[ti];
    const size_t n_u32 = per_task_metadata_bytes[k] / sizeof(uint32_t);
    rt.metadata.resize(n_u32);
    void *mapped = nullptr;
    RhiResult map_res = device_->map(*per_task_metadata_allocs[k], &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack sizer output buffer for readback");
    std::memcpy(rt.metadata.data(), mapped, per_task_metadata_bytes[k]);
    device_->unmap(*per_task_metadata_allocs[k]);
    rt.stride_float = rt.metadata[0];
    rt.stride_int = rt.metadata[1];
    // `QD_DEBUG_ADSTACK=1` opt-in diagnostic. Dumps the encoded bytecode's per-stack header (root_node_idx,
    // max_size_compile_time, heap_kind, entry_size_bytes) alongside the runtime-evaluated (offset,
    // max_size) that the sizer shader wrote back. A `root_node_idx < 0` stack means the host encoder
    // found no symbolic SizeExpr for the alloca (empty `size_expr.nodes`), so the sizer falls back to
    // `max_size_compile_time` - that's the most common overflow cause observed in practice and it's otherwise
    // invisible. Printed to stderr, one line per stack, unconditional when the env var is set.
    if (std::getenv("QD_DEBUG_ADSTACK")) {
      const auto &bc = per_task_bytecodes[k];
      const auto *hdr = reinterpret_cast<const AdStackSizeExprDeviceHeader *>(bc.data());
      const auto *stack_headers =
          reinterpret_cast<const AdStackSizeExprDeviceStackHeader *>(bc.data() + sizeof(AdStackSizeExprDeviceHeader));
      std::fprintf(stderr,
                   "[adstack_sizer] kernel='%s' task=%zu allocas=%zu bytecode_bytes=%zu "
                   "n_stacks=%u total_nodes=%u total_indices=%u stride_f=%u stride_i=%u\n",
                   kernel_name.c_str(), ti, task_attribs[ti].ad_stack.allocas.size(), bc.size(), hdr->n_stacks,
                   hdr->total_nodes, hdr->total_indices, rt.stride_float, rt.stride_int);
      for (uint32_t si = 0; si < hdr->n_stacks; ++si) {
        const auto &sh = stack_headers[si];
        uint32_t off = rt.metadata[2 + 2 * si];
        uint32_t mx = rt.metadata[2 + 2 * si + 1];
        std::fprintf(stderr,
                     "[adstack_sizer]   stack[%u]: heap=%s entry_bytes=%u root_idx=%d max_size_ct=%u -> offset=%u "
                     "max_size=%u%s\n",
                     si, sh.heap_kind == 0 ? "F" : "I", sh.entry_size_bytes, sh.root_node_idx, sh.max_size_compile_time,
                     off, mx, sh.root_node_idx < 0 ? " [fallback to compile-time bound - no symbolic tree]" : "");
      }
    }
    // Sanity cap: a per-thread adstack stride larger than this indicates the sizer shader returned garbage
    // (e.g. an `ExternalTensorRead` dereferenced an uninitialised arg-buffer slot). Without this guard the
    // downstream heap allocation below multiplies by `dispatched_threads` and asks the RHI for hundreds of
    // GB, which tears the machine down with OOM before any error surface has a chance to run. 16 Mi u32 words
    // per thread is already far beyond any realistic reverse-mode workload; pin it and hard-error so the bug
    // is attributed to the sizer output, not to the heap allocator at the call site that used the result.
    constexpr uint32_t kMaxSaneStridePerThread = 1u << 24;
    QD_ERROR_IF(rt.stride_float > kMaxSaneStridePerThread || rt.stride_int > kMaxSaneStridePerThread,
                "Adstack sizer shader returned an implausibly large per-thread stride (stride_float={}, "
                "stride_int={}, cap={}). This is almost always a bug in `encode_adstack_size_expr_device_"
                "bytecode_for_spirv` (wrong `kNodeOffArgBufferOffset` or missing `ExternalTensorRead` "
                "pre-substitution) or in the sizer shader's PSB read path, not a legitimate workload.",
                rt.stride_float, rt.stride_int, kMaxSaneStridePerThread);
    ctx_buffers_.push_back(std::move(per_task_metadata_allocs[k]));
  }

  return per_task_ad_stack;
}

}  // namespace gfx
}  // namespace quadrants::lang
