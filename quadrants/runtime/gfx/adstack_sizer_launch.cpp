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

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

#include "quadrants/codegen/spirv/adstack_sizer_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/program/program.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace gfx {

namespace {

// Walk a `SerializedSizeExpr` tree structurally and append every leaf the GPU sizer will read at
// launch time into the dependency sets: `snode_ids` for `FieldLoad` leaves, `arg_ids` for
// `ExternalTensorShape` and `ExternalTensorRead` leaves. The walk reads no live values (no SNode
// access, no buffer dereference, no nested kernel launch) - it is pure tree inspection - so it is
// safe to call from anywhere in the launcher's pre-publish phase. Used by the per-task metadata cache
// to know which generation counters to snapshot at record time and re-check at lookup time.
void collect_size_expr_dep_keys(const SerializedSizeExpr &expr,
                                std::unordered_set<int> &snode_ids,
                                std::unordered_set<int> &arg_ids) {
  for (const auto &node : expr.nodes) {
    switch (static_cast<SizeExpr::Kind>(node.kind)) {
      case SizeExpr::Kind::FieldLoad:
        if (node.snode_id >= 0) {
          snode_ids.insert(node.snode_id);
        }
        break;
      case SizeExpr::Kind::ExternalTensorShape:
      case SizeExpr::Kind::ExternalTensorRead:
        if (!node.arg_id_path.empty()) {
          arg_ids.insert(node.arg_id_path.front());
        }
        break;
      default:
        break;
    }
  }
}

// True iff every SizeExpr across every adstack alloca in every task is host-resolvable WITHOUT triggering a
// nested kernel launch. On SPIR-V backends both `ExternalTensorRead` and `FieldLoad` would require a device
// round-trip from the host: ExternalTensorRead dereferences ndarray data that lives in GPU-private memory,
// and FieldLoad would have to launch a `SNodeRwAccessorsBank::read_int` accessor kernel to read the SNode
// value off-device. Either of those launched mid-publish corrupts the SPIR-V launcher's per-task metadata
// upload state and shows up as wrong gradients on kernels that mix the two paths (this is the leaf set the
// SPIR-V on-device sizer was specifically built to handle on-device via PSB loads). The fast path therefore
// only fires when every SizeExpr leaf is one of `Const`, `ExternalTensorShape`, `BoundVariable`, or the
// arithmetic / `MaxOverRange` combinators.
bool all_size_exprs_host_resolvable(const std::vector<size_t> &adstack_task_indices,
                                    const std::vector<spirv::TaskAttributes> &task_attribs) {
  for (size_t ti : adstack_task_indices) {
    for (const auto &alloca : task_attribs[ti].ad_stack.allocas) {
      for (const auto &node : alloca.size_expr.nodes) {
        auto kind = static_cast<SizeExpr::Kind>(node.kind);
        if (kind == SizeExpr::Kind::ExternalTensorRead || kind == SizeExpr::Kind::FieldLoad) {
          return false;
        }
      }
    }
  }
  return true;
}

// Replicates the sizer shader's per-task metadata layout on the host: for each task, write a flat
// `[stride_float, stride_int, (offset_i, max_size_i)*]` buffer where `max_size_i` is the host-evaluated
// SizeExpr (falling back to `max_size_compile_time` for stacks with an empty `nodes` vector), and offsets
// run as a per-heap prefix sum (Float advances by `2 * max_size`, Int by `max_size`, matching the
// primal+adjoint interleaved Float-heap layout the main-kernel codegen already bakes in). The shader logic
// this mirrors lives in `quadrants/codegen/spirv/adstack_sizer_shader.cpp` (the `running_off_f` /
// `running_off_i` accumulator and the metadata writeback at the bottom of the per-stack loop).
//
// Caller must have verified `all_size_exprs_host_resolvable` first; this function asserts that no
// SizeExpr contains `ExternalTensorRead`.
void eval_per_task_metadata_on_host(const std::vector<size_t> &adstack_task_indices,
                                    const std::vector<spirv::TaskAttributes> &task_attribs,
                                    Program *prog,
                                    LaunchContextBuilder &host_ctx,
                                    std::vector<PerTaskAdStackRuntime> &per_task_ad_stack,
                                    const MaxReducerResultMap &max_reducer_results) {
  using HeapKind = spirv::TaskAttributes::AdStackAllocaAttribs::HeapKind;
  // Span the per-task `evaluate_adstack_size_expr` calls below with one shared read cache.
  SizeExprLaunchScope launch_scope;
  for (size_t ti : adstack_task_indices) {
    const auto &allocas = task_attribs[ti].ad_stack.allocas;
    const uint32_t registry_id = task_attribs[ti].ad_stack.registry_id;
    auto &rt = per_task_ad_stack[ti];
    const size_t n_stacks = allocas.size();
    rt.metadata.assign(3 + 2 * n_stacks, 0);  // trailing slot is the overflow-flag the on-device sizer writes
    uint32_t running_off_f = 0;
    uint32_t running_off_i = 0;
    for (size_t i = 0; i < n_stacks; ++i) {
      const auto &a = allocas[i];
      uint32_t max_size;
      if (a.size_expr.nodes.empty()) {
        // Cache-load fallback: no symbolic tree captured, the compile-time bound is authoritative.
        // Match the shader's `max(max_size_compile_time, 1)` lower clamp.
        max_size = std::max<uint32_t>(a.max_size_compile_time, 1u);
      } else {
        // Substitute any captured `MaxOverRange` whose result the max-reducer dispatched into a `Const` before the host
        // evaluator walks the tree. The substituted tree is a stack-local that cannot be used as a stable cache key, so
        // the substitution branch routes through `evaluate_adstack_size_expr_no_cache`; the empty-results fast path
        // keeps the live `a.size_expr` reference and the cache stays warm. The non-cache branch's per-launch eval cost
        // is small (a single tree walk dominated by `ExternalTensorRead` PSB dereferences); the dispatch the
        // substitution feeds off was the dominant cost in the first place.
        int64_t evaluated;
        if (max_reducer_results.empty()) {
          evaluated = evaluate_adstack_size_expr(a.size_expr, prog, &host_ctx);
        } else {
          const SerializedSizeExpr substituted = substitute_precomputed_max_over_range(
              a.size_expr, registry_id, static_cast<int32_t>(i), max_reducer_results);
          evaluated = evaluate_adstack_size_expr_no_cache(substituted, prog, &host_ctx);
        }
        // `evaluate_adstack_size_expr` returns -1 only when `expr.nodes` is empty (handled above) or hits
        // an internal hard error; clamp to the same `max(_, 1)` lower bound the shader applies.
        if (evaluated < 1) {
          evaluated = 1;
        }
        max_size = static_cast<uint32_t>(evaluated);
      }
      uint32_t offset;
      if (a.heap_kind == HeapKind::Float) {
        offset = running_off_f;
        // Primal + adjoint interleaved.
        running_off_f += 2u * max_size;
      } else {
        offset = running_off_i;
        running_off_i += max_size;
      }
      rt.metadata[2 + 2 * i] = offset;
      rt.metadata[2 + 2 * i + 1] = max_size;
    }
    rt.metadata[0] = running_off_f;
    rt.metadata[1] = running_off_i;
    rt.stride_float = running_off_f;
    rt.stride_int = running_off_i;
  }
}

}  // namespace

std::vector<PerTaskAdStackRuntime> GfxRuntime::publish_adstack_metadata_spirv(
    LaunchContextBuilder &host_ctx,
    DeviceAllocationGuard *args_buffer,
    const std::unordered_map<int, DeviceAllocation> &ndarray_allocs,
    const std::vector<quadrants::lang::spirv::TaskAttributes> &task_attribs,
    const std::string &kernel_name,
    const MaxReducerResultMap &max_reducer_results) {
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

  // Reverse-mode autodiff with adstacks requires Vulkan 1.3 (or Metal at MTLArgumentBuffersTier::Tier2) on this device.
  // Older drivers cannot run the sizer paths correctly; the per-helper cap gates downstream
  // (`dispatch_adstack_bound_reducers`, `dispatch_max_reducers`) rely on this single check and skip their own.
  QD_ERROR_IF(!device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer),
              "Reverse-mode autodiff with adstacks needs Vulkan 1.3 (or Metal Argument Buffers Tier 2); this "
              "device does not advertise `spirv_has_physical_storage_buffer`.");
  QD_ERROR_IF(!device_->get_caps().get(DeviceCapability::spirv_has_int64),
              "Reverse-mode autodiff with adstacks needs Vulkan 1.3 (or Metal Argument Buffers Tier 2); this "
              "device does not advertise `spirv_has_int64`.");

  // Register each adstack-bearing task with the Program-side identity registry so the host raise site
  // can name the offending kernel + task in its diagnostic message. Idempotent: re-registration of the
  // same `&task_attribs[ti].ad_stack` returns the same id and just refreshes the metadata. The
  // `task_attribs` vector lives inside the cached compiled kernel handle, so the address is stable
  // across launches. The id is written into `task_attribs[ti].ad_stack.registry_id` so the
  // synchronous sizer rerun in `Program::diagnose_adstack_overflow_message` can find the live
  // pointer; `set_adstack_sizing_info_pointer` flips the pointer-live sentinel that gates the deref.
  // The `task_attribs` parameter is passed `const` here so we cast away constness for the in-place
  // id stash; the cached compiled kernel owns the storage and outlives the launch.
  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    size_t ti = adstack_task_indices[k];
    auto &mutable_attribs =
        const_cast<quadrants::lang::spirv::TaskAttributes::AdStackSizingAttribs &>(task_attribs[ti].ad_stack);
    // Skip re-registration once the registry has the entry. `task_attribs` lives inside the cached compiled-kernel
    // handle and its allocas / size_exprs / max_size_compile_time are codegen-time immutable; a fresh compile would
    // use a new `mutable_attribs` address with `registry_id == 0`, so this is the natural one-shot guard.
    // Re-registering on every launch was costing ~2% of wallclock on the Metal rigid-step bench (vector copies for
    // allocated_max_sizes + size_exprs, plus a deep copy of the kernel_name string into the entry).
    if (mutable_attribs.registry_id != 0) {
      continue;
    }
    std::vector<int> allocated_max_sizes;
    std::vector<SerializedSizeExpr> size_exprs;
    allocated_max_sizes.reserve(mutable_attribs.allocas.size());
    size_exprs.reserve(mutable_attribs.allocas.size());
    for (const auto &a : mutable_attribs.allocas) {
      allocated_max_sizes.push_back(static_cast<int>(a.max_size_compile_time));
      size_exprs.push_back(a.size_expr);
    }
    uint32_t id = program_impl_->program->adstack_cache().register_adstack_sizing_info(
        static_cast<const void *>(&mutable_attribs), kernel_name, static_cast<int>(ti), std::move(allocated_max_sizes),
        std::move(size_exprs));
    mutable_attribs.registry_id = id;
  }

  // Populate the per-task `BufferType::AdStackTaskRegistryId` buffer with the registry id assigned in
  // the loop above. The codegen task-end overflow check reads slot `task_id_in_kernel_` and
  // `OpAtomicCompareExchange`'s it into `AdStackOverflow[1]` when the latter is still 0 (the FIRST
  // overflowing task across the dispatch records its identity). Slots for forward-only tasks default
  // to 0; the codegen short-circuits the cmpxchg when the loaded id is 0 anyway, but the explicit fill
  // keeps the buffer deterministic across launches and survives the offline-cache reload path.
  // Allocation policy mirrors `adstack_bound_row_capacity_buffer_` in
  // `adstack_bound_reducer_launch.cpp`: `host_write=true` SSBO, grow on amortised doubling, displaced
  // buffer parked in `ctx_buffers_` for in-flight cmdlist safety.
  {
    const size_t needed_bytes = std::max<size_t>(task_attribs.size(), 1) * sizeof(uint32_t);
    if (!adstack_task_registry_id_buffer_ || adstack_task_registry_id_buffer_size_ < needed_bytes) {
      size_t new_size = std::max(needed_bytes, 2 * adstack_task_registry_id_buffer_size_);
      auto [buf, res] = device_->allocate_memory_unique({new_size,
                                                         /*host_write=*/true,
                                                         /*host_read=*/false,
                                                         /*export_sharing=*/false, AllocUsage::Storage});
      QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack task registry id buffer (size={})",
                     new_size);
      if (adstack_task_registry_id_buffer_) {
        ctx_buffers_.push_back(std::move(adstack_task_registry_id_buffer_));
      }
      adstack_task_registry_id_buffer_ = std::move(buf);
      adstack_task_registry_id_buffer_size_ = new_size;
    }
    void *mapped = nullptr;
    RhiResult map_res = device_->map_range(adstack_task_registry_id_buffer_->get_ptr(0), needed_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack task registry id buffer");
    uint32_t *slots = reinterpret_cast<uint32_t *>(mapped);
    for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
      slots[ti] = task_attribs[ti].ad_stack.registry_id;
    }
    device_->unmap(*adstack_task_registry_id_buffer_);
  }

  // Fast path: when no SizeExpr in any adstack-bearing task contains an `ExternalTensorRead` leaf, every
  // capacity bound is host-resolvable through `evaluate_adstack_size_expr`, and the entire GPU sizer pipeline
  // (sizer-bytecode upload, per-task metadata-buffer alloc, `flush()` + `device_->wait_idle()` to force PSB
  // visibility, sizer cmdlist record + `submit_synced`, blocking metadata readback) drops out. On Metal /
  // MoltenVK the dropped pair of `wait_idle()` calls each cost a full GPU-host stall per launch; with one
  // launch per substep across forward + backward of a 100-substep test, that stall cost compounds linearly
  // with the test's launch count and is the dominant per-launch overhead under adstack mode. Kernels whose
  // size_expr trees include `ExternalTensorRead` (an ndarray scalar load whose data lives in GPU-private
  // memory) still need the on-device sizer below.
  if (all_size_exprs_host_resolvable(adstack_task_indices, task_attribs)) {
    eval_per_task_metadata_on_host(adstack_task_indices, task_attribs, program_impl_->program, host_ctx,
                                   per_task_ad_stack, max_reducer_results);
    return per_task_ad_stack;
  }

  // Per-task metadata cache fast path. Each adstack-bearing task is keyed by the stable address of its
  // `AdStackSizingAttribs` struct (lifetime of the compiled kernel); cached payload is the metadata
  // bytes the sizer wrote back plus per-source generation snapshots tagged at record time. Hit means
  // the entire sizer pipeline (`flush + wait_idle`, per-task metadata buffer alloc, cmdlist record,
  // `submit_synced`, readback) drops out, which on Metal is the dominant per-launch cost. Partial hits
  // fall through to a full pipeline run rather than threading a "skip task k" path through the cmdlist
  // record loop; mixed hit / miss only happens immediately after an invalidation so the simpler code
  // path is the right tradeoff. Soundness comes from `Program::try_per_task_ad_stack_cache_hit`
  // re-checking every counter the recorded entry tagged: per-snode `snode_write_gen_` covers
  // `FieldLoad` reads, per-DeviceAllocation `ndarray_data_gen_` covers `ExternalTensorRead` reads, and
  // the cached `arg_id -> devalloc` map catches a different tensor at the same arg slot
  // (`ExternalTensorShape` invariance is per-tensor, not across tensors).
  {
    bool all_hit = true;
    for (size_t k = 0; k < adstack_task_indices.size() && all_hit; ++k) {
      size_t ti = adstack_task_indices[k];
      AdStackCache::PerTaskAdStackCacheEntry entry;
      if (program_impl_->program->adstack_cache().try_per_task_ad_stack_cache_hit(
              static_cast<const void *>(&task_attribs[ti].ad_stack), &host_ctx, entry)) {
        auto &rt = per_task_ad_stack[ti];
        rt.metadata = std::move(entry.metadata);
        rt.stride_float = entry.stride_float;
        rt.stride_int = entry.stride_int;
      } else {
        all_hit = false;
      }
    }
    if (all_hit) {
      return per_task_ad_stack;
    }
    // Reset per-task strides clobbered by partial hits above; the GPU sizer pipeline below repopulates
    // them from scratch.
    for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
      size_t ti = adstack_task_indices[k];
      per_task_ad_stack[ti].metadata.clear();
      per_task_ad_stack[ti].stride_float = task_attribs[ti].ad_stack.per_thread_stride_float_compile_time;
      per_task_ad_stack[ti].stride_int = task_attribs[ti].ad_stack.per_thread_stride_int_compile_time;
    }
  }

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

  // Lazily allocate the two scratch SSBOs that host the sizer's per-invocation interpreter state. Bound on
  // every sizer dispatch at slots 3 (i64) and 4 (i32). Sizes are fixed by the shader-side layout constants;
  // see `kAdStackSizerScratchI64Elems` / `kAdStackSizerScratchI32Elems` in `adstack_sizer_shader.h`.
  if (!adstack_sizer_scratch_i64_buffer_) {
    auto [buf, res] = device_->allocate_memory_unique(
        {static_cast<size_t>(spirv::kAdStackSizerScratchI64Elems) * sizeof(int64_t),
         /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer i64 scratch buffer ({} bytes)",
                   static_cast<size_t>(spirv::kAdStackSizerScratchI64Elems) * sizeof(int64_t));
    adstack_sizer_scratch_i64_buffer_ = std::move(buf);
  }
  if (!adstack_sizer_scratch_i32_buffer_) {
    auto [buf, res] = device_->allocate_memory_unique(
        {static_cast<size_t>(spirv::kAdStackSizerScratchI32Elems) * sizeof(int32_t),
         /*host_write=*/false, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack sizer i32 scratch buffer ({} bytes)",
                   static_cast<size_t>(spirv::kAdStackSizerScratchI32Elems) * sizeof(int32_t));
    adstack_sizer_scratch_i32_buffer_ = std::move(buf);
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
  // Span the per-task bytecode encoding below with one shared read cache.
  SizeExprLaunchScope launch_scope;
  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    size_t ti = adstack_task_indices[k];
    per_task_bytecodes[k] = encode_adstack_size_expr_device_bytecode_for_spirv(
        task_attribs[ti].ad_stack, program_impl_->program, &host_ctx, max_reducer_results);
    per_task_bytecode_offsets[k] = align_up(total_bytecode_bytes, kDescriptorOffsetAlignment);
    total_bytecode_bytes = per_task_bytecode_offsets[k] + per_task_bytecodes[k].size();
    per_task_metadata_bytes[k] = (3u + 2u * task_attribs[ti].ad_stack.allocas.size()) * sizeof(uint32_t);
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
    // Per-invocation interpreter scratch (see allocation site above). The contents are entirely overwritten on every
    // dispatch (the shader zero-inits its `scope_arr` slice and writes-before-reads the rest as it walks the bytecode),
    // so no inter-launch state needs to survive.
    bindings->rw_buffer(3, *adstack_sizer_scratch_i64_buffer_);
    bindings->rw_buffer(4, *adstack_sizer_scratch_i32_buffer_);

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
    // Cap-hit tripwire. The on-device sizer writes 1 into the trailing overflow-flag slot when it observes a
    // `MaxOverRange` whose iteration count exceeds the `1<<24` cap; the hard error here surfaces the failure at
    // `qd.sync()` with a clean attribution. Recognized `MaxOverRange` shapes are dispatched in parallel by the
    // max-reducer and substituted to `Const` before the sizer interpreter sees them, so this path is reachable only for
    // out-of-grammar shapes; broadening the recognizer grammar moves more shapes onto the loud path automatically.
    const size_t overflow_word_idx = 2u + 2u * task_attribs[ti].ad_stack.allocas.size();
    QD_ERROR_IF(overflow_word_idx < rt.metadata.size() && rt.metadata[overflow_word_idx] != 0,
                "Adstack on-device sizer hit a `MaxOverRange` whose iteration count exceeds the {} cap. The recognized "
                "grammar's max-reducer dispatch did not capture this shape so the substitution path could not pre-fold "
                "the `MaxOverRange` to a `Const`. Restructure the source kernel to fit the recognizer grammar (single "
                "bound variable per body, body limited to `Const` / `ExternalTensorRead(arg, [BoundVariable])` / `Add` "
                "/ `Sub` / `Mul` / `Max`), or shrink the enclosing reverse-mode loop's iteration count below the cap.",
                int64_t{1} << 24);
    ctx_buffers_.push_back(std::move(per_task_metadata_allocs[k]));
  }

  // Record cache entries. Per task we walk every alloca's `size_expr` to build the dependency set
  // (snode_ids referenced by `FieldLoad`, arg_ids referenced by `ExternalTensorShape` /
  // `ExternalTensorRead`), snapshot the corresponding generation counters and the bound
  // DeviceAllocation for each arg_id, and stash everything alongside the metadata bytes. The
  // structural walk reads no live values, so it never re-enters `launch_kernel`.
  for (size_t k = 0; k < adstack_task_indices.size(); ++k) {
    size_t ti = adstack_task_indices[k];
    auto &rt = per_task_ad_stack[ti];
    std::unordered_set<int> snode_ids;
    std::unordered_set<int> arg_ids;
    for (const auto &alloca : task_attribs[ti].ad_stack.allocas) {
      collect_size_expr_dep_keys(alloca.size_expr, snode_ids, arg_ids);
    }
    std::vector<std::pair<int, uint64_t>> snode_gens;
    snode_gens.reserve(snode_ids.size());
    for (int snode_id : snode_ids) {
      snode_gens.emplace_back(snode_id, program_impl_->program->adstack_cache().snode_write_gen(snode_id));
    }
    std::vector<std::tuple<int, void *, uint64_t>> arg_gens;
    arg_gens.reserve(arg_ids.size());
    for (int arg_id : arg_ids) {
      ArgArrayPtrKey data_key{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      auto ap_it = host_ctx.array_ptrs.find(data_key);
      void *devalloc = (ap_it == host_ctx.array_ptrs.end()) ? nullptr : ap_it->second;
      arg_gens.emplace_back(arg_id, devalloc, program_impl_->program->adstack_cache().ndarray_data_gen(devalloc));
    }
    program_impl_->program->adstack_cache().record_per_task_ad_stack(
        static_cast<const void *>(&task_attribs[ti].ad_stack), rt.metadata, rt.stride_float, rt.stride_int,
        std::move(snode_gens), std::move(arg_gens));
  }

  return per_task_ad_stack;
}

}  // namespace gfx
}  // namespace quadrants::lang
