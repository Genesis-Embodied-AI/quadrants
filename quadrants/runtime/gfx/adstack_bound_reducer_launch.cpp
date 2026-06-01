// Static-IR-bound sparse-adstack-heap reducer dispatch for SPIR-V backends. Extracted out of `runtime.cpp` for the same
// reason `adstack_sizer_launch.cpp` is - keep `GfxRuntime::launch_kernel` focused on the main-kernel record/submit
// flow. Every code path here is conditional on at least one task in the kernel having a captured
// `TaskAttributes::AdStackSizingAttribs::bound_expr` whose `field_source_kind` is `NdArray`; on kernels without such a
// task, or on devices missing the required SPIR-V capabilities, the helper returns an empty map and the heap-bind path
// in `launch_kernel` falls through to the dispatched-threads worst-case sizing - safe but no savings.
//
// Mechanism end-to-end:
// 1. Filter `task_attribs` to the tasks whose `bound_expr` matches the supported shape (NdArray-backed,
//    f32 or i32 element type). Build a parallel vector of `AdStackBoundReducerParams` blobs keyed by the task's
//    `task_id_in_kernel`.
// 2. Lazy-initialise the reducer pipeline (`adstack_bound_reducer_pipeline_`) on the first call.
// 3. Lazy-grow the parameter blob storage buffer to fit `n_matches` blobs at descriptor-alignment offsets.
// 4. Lazy-grow the `AdStackRowCounter` buffer to fit `num_tasks_in_kernel` u32 slots, then clear it (the
//    reducer's atomic-adds accumulate into slot[task_id], so a leftover count from a prior launch would contaminate
//    this launch's reduce).
// 5. Build a fresh cmdlist, bind+dispatch the reducer per matched task at its corresponding params offset,
//    submit_synced.
// 6. Map the counter buffer, read each matched task's slot into the result map, unmap.
// 7. Clear the counter buffer AGAIN before returning: the main task's own LCA-block atomic-add writes the
//    same slots during its dispatch (Phase A+B+C lazy row claim), and a leftover reducer count there would skew the row
//    id range the main pass produces.
//
// Caller responsibility: invoke `dispatch_adstack_bound_reducers` BEFORE the main task bind/dispatch loop and consult
// the returned map at the AdStackHeapFloat bind site to size each matched task's heap allocation to `count[task_id] *
// stride_float * sizeof(f32)`. Tasks not in the map (no `bound_expr`, SNode-backed, or capability-missing fallback)
// keep the existing `dispatched_threads * stride_float` worst-case sizing.

#include "quadrants/runtime/gfx/runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include "quadrants/codegen/spirv/adstack_bound_reducer_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/stmt_op_types.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace gfx {

namespace {

// Map a captured `BinaryOpType` (stored as int in `StaticBoundExpr::cmp_op`) onto the `AdStackBoundReducerOpCode` value
// the shader's OpSwitch dispatches on. Returns an out-of-range value when the captured op is not one of the six
// recognized comparisons; the caller is expected to have already filtered such bound_exprs out at the IR-pattern-match
// stage, so reaching the default branch is an internal-consistency error.
spirv::AdStackBoundReducerOpCode encode_cmp_op(int captured_cmp_op) {
  switch (static_cast<BinaryOpType>(captured_cmp_op)) {
    case BinaryOpType::cmp_lt:
      return spirv::kAdStackBoundReducerOpLt;
    case BinaryOpType::cmp_le:
      return spirv::kAdStackBoundReducerOpLe;
    case BinaryOpType::cmp_gt:
      return spirv::kAdStackBoundReducerOpGt;
    case BinaryOpType::cmp_ge:
      return spirv::kAdStackBoundReducerOpGe;
    case BinaryOpType::cmp_eq:
      return spirv::kAdStackBoundReducerOpEq;
    case BinaryOpType::cmp_ne:
      return spirv::kAdStackBoundReducerOpNe;
    default:
      QD_ERROR(
          "static_bound_expr captured unsupported BinaryOpType={} (internal-consistency: the IR "
          "pattern matcher should have rejected this at codegen time)",
          captured_cmp_op);
      return spirv::kAdStackBoundReducerOpEq;  // unreachable after QD_ERROR
  }
}

// Resolve the byte offset within the kernel arg buffer where the ndarray's `data_ptr` (u64) lives. Mirrors the
// `kNodeOffArgBufferOffset` precomputation the SizeExpr device-bytecode encoder does for its own `ExternalTensorRead`
// nodes (see `adstack_size_expr_eval.cpp`) - the layout knowledge is centralised in
// `LaunchContextBuilder::args_type->get_element_offset`, so any update to the args-struct layout flows through both
// call sites uniformly. Returned offset is in BYTES; the shader divides by 4 (because the params blob slot stores a u32
// word offset into the arg buffer's u32[] view).
size_t resolve_ndarray_data_ptr_byte_offset(LaunchContextBuilder &host_ctx, const std::vector<int> &arg_id_path) {
  QD_ASSERT_INFO(host_ctx.args_type != nullptr,
                 "adstack bound reducer: LaunchContextBuilder::args_type is null; cannot resolve ndarray "
                 "data pointer offset for the captured StaticBoundExpr");
  std::vector<int> indices = arg_id_path;
  indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  return host_ctx.args_type->get_element_offset(indices);
}

}  // namespace

std::unordered_map<int, uint32_t> GfxRuntime::dispatch_adstack_bound_reducers(
    LaunchContextBuilder &host_ctx,
    DeviceAllocationGuard *args_buffer,
    const std::vector<spirv::TaskAttributes> &task_attribs) {
  std::unordered_map<int, uint32_t> result;

  // Hoisted ABOVE the capability gates so cap-missing devices still receive inert UINT32_MAX defaults: every
  // reverse-mode kernel with at least one f32 adstack reaches the codegen-emitted defense-in-depth bounds check at the
  // float Lowest Common Ancestor (LCA) block, which loads `AdStackBoundRowCapacity[task_id]`. If the buffer stays
  // unallocated on cap-missing devices the runtime bind path routes `kDeviceNullAllocation` there, robustBufferAccess
  // returns 0, and the divergence-overflow OpAtomicUMax fires unconditionally (`claimed_row >= 0u` is always true for
  // u32) - hard-erroring every adstack-bearing kernel at sync. The capacity-buffer alloc + UINT32_MAX fill is host-side
  // only (SSBO host-write through map_range) and does NOT require PSB or Int64 - those caps gate the reducer compute
  // shader, not the host-side buffer fill. Run the fill first so cap-missing devices still produce inert defaults that
  // the codegen clamp leaves alone, then early-return on cap-miss for the dispatch.
  const size_t needed_capacity_bytes = std::max<size_t>(task_attribs.size(), 1) * sizeof(uint32_t);
  if (!adstack_bound_row_capacity_buffer_ || adstack_bound_row_capacity_buffer_size_ < needed_capacity_bytes) {
    size_t new_size = std::max(needed_capacity_bytes, 2 * adstack_bound_row_capacity_buffer_size_);
    auto [buf, res] = device_->allocate_memory_unique({new_size,
                                                       /*host_write=*/true,
                                                       /*host_read=*/false,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack bound row capacity buffer (size={})",
                   new_size);
    if (adstack_bound_row_capacity_buffer_) {
      ctx_buffers_.push_back(std::move(adstack_bound_row_capacity_buffer_));
    }
    adstack_bound_row_capacity_buffer_ = std::move(buf);
    adstack_bound_row_capacity_buffer_size_ = new_size;
  }
  {
    void *mapped = nullptr;
    RhiResult map_res =
        device_->map_range(adstack_bound_row_capacity_buffer_->get_ptr(0), needed_capacity_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack bound row capacity buffer for default fill");
    uint32_t *slots = reinterpret_cast<uint32_t *>(mapped);
    for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
      slots[ti] = std::numeric_limits<uint32_t>::max();
    }
    device_->unmap(*adstack_bound_row_capacity_buffer_);
  }

  // Capability gate: the reducer shader builds an empty SPIR-V binary on devices without PSB+Int64, so the lazy-init
  // below would fail and there is no correct host-eval fallback for an ndarray data pointer that lives in GPU-private
  // memory. Skip the dispatch and return an empty map; the caller falls back to dispatched-threads worst-case heap
  // sizing for every task with the inert UINT32_MAX defaults the hoisted capacity-fill above produced. Every backend
  // Quadrants targets that has adstack support advertises both caps, so this is a defensive guard rather than a routine
  // path.
  if (!device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer)) {
    return result;
  }
  if (!device_->get_caps().get(DeviceCapability::spirv_has_int64)) {
    return result;
  }

  // Filter to the tasks whose bound_expr is consumable by the reducer (NdArray-backed via the kernel arg buffer + PSB
  // load, or SNode-backed via a direct word load from the matching root buffer at compile-time-precomputed byte offset
  // / cell stride). Both source kinds use the same generic shader; the dispatch-time params blob's
  // `field_source_is_snode` flag picks the path per task.
  const bool has_f64 = device_->get_caps().get(DeviceCapability::spirv_has_float64);
  std::vector<int> matched_task_indices;
  matched_task_indices.reserve(task_attribs.size());
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    const auto &be = task_attribs[ti].ad_stack.bound_expr;
    if (!be.has_value()) {
      continue;
    }
    using FSK = spirv::TaskAttributes::StaticBoundExpr::FieldSourceKind;
    if (be->field_source_kind != FSK::NdArray && be->field_source_kind != FSK::SNode) {
      continue;
    }
    // f64-captured gates need the f64 reducer arm in the shader; on devices without `spirv_has_float64` the shader was
    // built without an OpType for f64 and the f64-bitcast / OpFOrd* for f64 would not be valid, so route those tasks
    // through the worst-case heap-sizing fallback (drop them from the matched set).
    if (be->field_dtype_is_float && be->field_dtype_is_double && !has_f64) {
      continue;
    }
    matched_task_indices.push_back(static_cast<int>(ti));
  }

  if (matched_task_indices.empty()) {
    return result;
  }

  // Resolve buffers per source kind. The reducer dispatch always binds slots 0/1/2/3; binding slot 0 (args_buffer) and
  // slot 3 (root_buffer) is required to satisfy the descriptor set layout, but only the slot matching the captured
  // `field_source_kind` is read by the shader. For tasks whose source kind has no real backing buffer in this kernel,
  // fall back to the params buffer as a safe non-null placeholder (the shader's load against the placeholder is never
  // executed because of the `field_source_is_snode` branch).
  bool any_ndarray_source = false;
  bool any_snode_source = false;
  for (int ti : matched_task_indices) {
    using FSK = spirv::TaskAttributes::StaticBoundExpr::FieldSourceKind;
    const auto &be = *task_attribs[ti].ad_stack.bound_expr;
    if (be.field_source_kind == FSK::NdArray) {
      any_ndarray_source = true;
    } else if (be.field_source_kind == FSK::SNode) {
      any_snode_source = true;
    }
  }
  QD_ASSERT_INFO(!any_ndarray_source || args_buffer != nullptr,
                 "adstack bound reducer: a matched task has NdArray-backed bound_expr but the kernel arg "
                 "buffer is null; the launcher should have allocated it before reaching here");

  // Lazy-init pipeline. Mirrors `adstack_sizer_launch.cpp`'s pattern: build the SPIR-V binary once via the shader-build
  // helper, hand to the device's pipeline factory, cache for the runtime's lifetime.
  if (!adstack_bound_reducer_pipeline_) {
    std::vector<uint32_t> spirv = spirv::build_adstack_bound_reducer_spirv(Arch::vulkan, &device_->get_caps());
    QD_ASSERT_INFO(!spirv.empty(),
                   "build_adstack_bound_reducer_spirv returned an empty binary despite the PSB+Int64 cap "
                   "check passing; bug in the shader builder's capability gating");
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                   spirv.size() * sizeof(uint32_t)};
    auto [pipeline, res] = device_->create_pipeline_unique(source_desc, "adstack_bound_reducer", backend_cache_.get());
    QD_ERROR_IF(res != RhiResult::success, "Failed to create pipeline for the adstack bound reducer (err: {})",
                int(res));
    adstack_bound_reducer_pipeline_ = std::move(pipeline);
  }

  // Pack one params blob per matched task at descriptor-alignment offsets. Vulkan's minStorageBufferOffsetAlignment
  // caps at 256 B for the most conservative drivers in the wild (older NVIDIA), so we round up to that; this trades a
  // little extra buffer space for a fixed alignment that every backend can bind without VUID-02999 violations. Pack the
  // blobs into a single contiguous host-visible buffer and bind each task's per-task slice via `get_ptr(offset) +
  // size`.
  constexpr size_t kDescriptorOffsetAlignment = 256;
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  const size_t params_size_bytes = spirv::AdStackBoundReducerParams::kNumWords * sizeof(uint32_t);
  std::vector<size_t> per_task_params_offsets(matched_task_indices.size());
  size_t total_params_bytes = 0;
  for (size_t k = 0; k < matched_task_indices.size(); ++k) {
    per_task_params_offsets[k] = align_up(total_params_bytes, kDescriptorOffsetAlignment);
    total_params_bytes = per_task_params_offsets[k] + params_size_bytes;
  }

  if (!adstack_bound_reducer_params_buffer_ || adstack_bound_reducer_params_buffer_size_ < total_params_bytes) {
    size_t new_size = std::max(total_params_bytes, 2 * adstack_bound_reducer_params_buffer_size_);
    auto [buf, res] = device_->allocate_memory_unique(
        {new_size, /*host_write=*/true, /*host_read=*/false, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack bound reducer params buffer (size={})",
                   new_size);
    if (adstack_bound_reducer_params_buffer_) {
      ctx_buffers_.push_back(std::move(adstack_bound_reducer_params_buffer_));
    }
    adstack_bound_reducer_params_buffer_ = std::move(buf);
    adstack_bound_reducer_params_buffer_size_ = new_size;
  }

  // Resolve per-task length. The reducer walks `selector[0..length)` and counts gate-passing cells; the main-kernel
  // LCA-block atomic-rmw fires once per gated iteration across the full logical loop span (the kernel grid-strides via
  // `loop_var += total_invocs` so dispatched-thread count does not cap the claim count). For ndarray-backed gates we
  // therefore walk the gating ndarray's full flat element product - mirrors the LLVM launchers' shape-product walk and
  // removes the prior cap at `advisory_total_num_threads` which under-counted on workloads larger than 65536
  // (struct_for) or 131072 (range_for). For SNode-backed gates `be.snode_iter_count` already carries the full iteration
  // count, so the call site reads it directly without going through this lambda.
  auto resolve_length_ndarray = [&](const spirv::TaskAttributes::StaticBoundExpr &be) -> uint32_t {
    int64_t flat_len = 1;
    for (int axis = 0; axis < be.ndarray_ndim; ++axis) {
      std::vector<int> indices = be.ndarray_arg_id;
      indices.push_back(TypeFactory::SHAPE_POS_IN_NDARRAY);
      indices.push_back(axis);
      flat_len *= int64_t(host_ctx.get_struct_arg<int32_t>(indices));
    }
    return static_cast<uint32_t>(std::max<int64_t>(0, flat_len));
  };

  // Build params blobs and write them into the params buffer. Resolve the captured ndarray data-ptr byte offset via
  // `LaunchContextBuilder::args_type::get_element_offset` (same path the SizeExpr encoder uses), then convert byte
  // offset to u32 word offset for the shader's index arithmetic.
  {
    void *mapped = nullptr;
    RhiResult map_res =
        device_->map_range(adstack_bound_reducer_params_buffer_->get_ptr(0), total_params_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack bound reducer params buffer");
    for (size_t k = 0; k < matched_task_indices.size(); ++k) {
      const int ti = matched_task_indices[k];
      const auto &attribs = task_attribs[ti];
      const auto &be = *attribs.ad_stack.bound_expr;
      using FSK = spirv::TaskAttributes::StaticBoundExpr::FieldSourceKind;
      const bool is_snode = be.field_source_kind == FSK::SNode;
      uint32_t arg_word_offset = 0;
      if (!is_snode) {
        const size_t data_ptr_byte_off = resolve_ndarray_data_ptr_byte_offset(host_ctx, be.ndarray_arg_id);
        QD_ASSERT_INFO(data_ptr_byte_off % sizeof(uint32_t) == 0,
                       "adstack bound reducer: ndarray data pointer offset {} is not 4-byte aligned in the "
                       "kernel arg buffer; layout mismatch with the SizeExpr encoder",
                       data_ptr_byte_off);
        arg_word_offset = static_cast<uint32_t>(data_ptr_byte_off / sizeof(uint32_t));
      } else {
        QD_ASSERT_INFO(
            be.snode_byte_base_offset % sizeof(uint32_t) == 0 && be.snode_byte_cell_stride % sizeof(uint32_t) == 0,
            "adstack bound reducer: SNode-backed bound_expr offsets must be 4-byte aligned "
            "(base={}, stride={})",
            be.snode_byte_base_offset, be.snode_byte_cell_stride);
      }
      spirv::AdStackBoundReducerParams params{};
      params.task_id_in_kernel = static_cast<uint32_t>(ti);
      params.length = is_snode ? be.snode_iter_count : resolve_length_ndarray(be);
      params.arg_word_offset = arg_word_offset;
      params.op_code = static_cast<uint32_t>(encode_cmp_op(be.cmp_op));
      params.field_dtype_is_float = be.field_dtype_is_float ? 1u : 0u;
      params.field_dtype_is_double = be.field_dtype_is_double ? 1u : 0u;
      params.polarity = be.polarity ? 1u : 0u;
      // Threshold encoding mirrors the LLVM reducer's `LlvmAdStackBoundReducerDeviceParams.threshold_bits[_high]` pair
      // (see runtime_eval_static_bound_count in runtime/llvm/runtime_module/runtime.cpp). f64 splits the 64-bit literal
      // across the low / high u32 pair so the shader can reassemble it without hardcoding a 64-bit OpConstant; f32 /
      // i32 keep the high half at zero.
      if (be.field_dtype_is_float && be.field_dtype_is_double) {
        uint64_t bits64 = 0;
        std::memcpy(&bits64, &be.literal_f64, sizeof(bits64));
        params.threshold_bits = static_cast<uint32_t>(bits64 & 0xFFFFFFFFu);
        params.threshold_bits_high = static_cast<uint32_t>(bits64 >> 32);
      } else if (be.field_dtype_is_float) {
        uint32_t bits32 = 0;
        std::memcpy(&bits32, &be.literal_f32, sizeof(bits32));
        params.threshold_bits = bits32;
        params.threshold_bits_high = 0u;
      } else {
        params.threshold_bits = static_cast<uint32_t>(be.literal_i32);
        params.threshold_bits_high = 0u;
      }
      params.field_source_is_snode = is_snode ? 1u : 0u;
      params.snode_byte_base_offset = be.snode_byte_base_offset;
      params.snode_byte_cell_stride = be.snode_byte_cell_stride;
      std::memcpy(reinterpret_cast<char *>(mapped) + per_task_params_offsets[k], &params, params_size_bytes);
    }
    device_->unmap(*adstack_bound_reducer_params_buffer_);
  }

  // Ensure the per-task counter slots fit `num_tasks_in_kernel` u32 entries (same precondition the main-kernel codegen
  // relies on for its LCA-block atomic-add) and clear them before the reducer dispatches. The buffer may have been
  // grown by an earlier kernel launch with more tasks; we only grow on demand.
  const size_t needed_counter_bytes = task_attribs.size() * sizeof(uint32_t);
  if (!adstack_row_counter_buffer_ || adstack_row_counter_buffer_size_ < needed_counter_bytes) {
    size_t new_size = std::max(needed_counter_bytes, 2 * adstack_row_counter_buffer_size_);
    auto [buf, res] = device_->allocate_memory_unique({new_size,
                                                       /*host_write=*/false,
                                                       /*host_read=*/true,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack row counter buffer (size={})", new_size);
    if (adstack_row_counter_buffer_) {
      ctx_buffers_.push_back(std::move(adstack_row_counter_buffer_));
    }
    adstack_row_counter_buffer_ = std::move(buf);
    adstack_row_counter_buffer_size_ = new_size;
  }

  // Force visibility of prior writes the same way `adstack_sizer_launch.cpp` does (see its block comment around
  // `flush(); device_->wait_idle();`): MoltenVK's PSB load path bypasses the descriptor-bound cache that a prior
  // accessor kernel's submit_synced flushed via vkQueueWaitIdle, so without this sequence the reducer reads stale
  // ndarray contents on Apple Silicon and undercounts.
  flush();
  device_->wait_idle();

  // Zero the counter slots through a fresh cmdlist (RHI does not expose a host-side fill on a host_read-only
  // allocation, and we want the clear ordered before the reducer dispatch). buffer_fill is the same primitive the
  // main-launch path uses to clear the counter on `i==0`.
  auto [clear_cmdlist, clear_cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(clear_cmdlist_res == RhiResult::success, "Failed to create adstack reducer clear cmdlist");
  clear_cmdlist->buffer_fill(adstack_row_counter_buffer_->get_ptr(0), needed_counter_bytes, /*data=*/0);
  clear_cmdlist->buffer_barrier(*adstack_row_counter_buffer_);
  device_->get_compute_stream()->submit_synced(clear_cmdlist.get());

  // Dispatch the reducer per matched task. Each dispatch binds the same args + counter buffers but a different per-task
  // slice of the params buffer; the shader reads `task_id_in_kernel` out of its slice and atomic-adds 1 into
  // `counter[task_id]` for each matched thread.
  auto [reducer_cmdlist, reducer_cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(reducer_cmdlist_res == RhiResult::success, "Failed to create adstack reducer cmdlist");
  for (size_t k = 0; k < matched_task_indices.size(); ++k) {
    const int ti = matched_task_indices[k];
    const auto &attribs = task_attribs[ti];
    const auto &be = *attribs.ad_stack.bound_expr;
    using FSK = spirv::TaskAttributes::StaticBoundExpr::FieldSourceKind;
    const bool is_snode = be.field_source_kind == FSK::SNode;
    auto bindings = device_->create_resource_set_unique();
    // Slot 0 (args_buffer): required for ndarray-backed; on SNode-only tasks supply a dedicated lazy-allocated
    // placeholder buffer so the descriptor layout is satisfied. We cannot reuse the params buffer here because some RHI
    // backends (Metal / MoltenVK) reject the same DeviceAllocation appearing on two slots of one descriptor set, and
    // the params buffer is already bound at slot 2.
    if (args_buffer != nullptr) {
      bindings->rw_buffer(0, *args_buffer);
    } else {
      if (!adstack_bound_reducer_args_placeholder_buffer_) {
        auto [buf, res] = device_->allocate_memory_unique({sizeof(uint32_t),
                                                           /*host_write=*/false,
                                                           /*host_read=*/false,
                                                           /*export_sharing=*/false, AllocUsage::Storage});
        QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack bound reducer slot-0 placeholder buffer");
        adstack_bound_reducer_args_placeholder_buffer_ = std::move(buf);
      }
      bindings->rw_buffer(0, *adstack_bound_reducer_args_placeholder_buffer_);
    }
    bindings->rw_buffer(1, *adstack_row_counter_buffer_);
    bindings->rw_buffer(2, adstack_bound_reducer_params_buffer_->get_ptr(per_task_params_offsets[k]),
                        params_size_bytes);
    // Slot 3 (root_buffer): required for SNode-backed; supply the params buffer as a placeholder for ndarray-only tasks
    // so the descriptor layout is satisfied without the shader actually reading it.
    if (is_snode) {
      DeviceAllocation *root_alloc = get_root_buffer(be.snode_root_id);
      QD_ASSERT_INFO(root_alloc != nullptr,
                     "adstack bound reducer: SNode-backed bound_expr references root_id={} but the runtime has no "
                     "matching root buffer; check that the kernel's snode tree was registered",
                     be.snode_root_id);
      bindings->rw_buffer(3, *root_alloc);
    } else {
      // ndarray-only path: bind a non-null storage buffer the shader's branch never reads. Some RHI backends (Metal /
      // MoltenVK) reject the same DeviceAllocation appearing on two slots of one descriptor set, so we cannot reuse the
      // params or counter buffer here. Lazy-allocate a one-word scratch buffer dedicated to this placeholder slot the
      // first time we need it; it lives for the runtime's lifetime and never gets read.
      if (!adstack_bound_reducer_root_placeholder_buffer_) {
        auto [buf, res] = device_->allocate_memory_unique({sizeof(uint32_t),
                                                           /*host_write=*/false,
                                                           /*host_read=*/false,
                                                           /*export_sharing=*/false, AllocUsage::Storage});
        QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack bound reducer slot-3 placeholder buffer");
        adstack_bound_reducer_root_placeholder_buffer_ = std::move(buf);
      }
      bindings->rw_buffer(3, *adstack_bound_reducer_root_placeholder_buffer_);
    }

    reducer_cmdlist->bind_pipeline(adstack_bound_reducer_pipeline_.get());
    RhiResult bind_res = reducer_cmdlist->bind_shader_resources(bindings.get());
    QD_ERROR_IF(bind_res != RhiResult::success, "adstack bound reducer resource binding error: RhiResult({})",
                int(bind_res));

    const uint32_t length = is_snode ? be.snode_iter_count : resolve_length_ndarray(be);
    const uint32_t group_x =
        (length + spirv::kAdStackBoundReducerWorkgroupSize - 1) / spirv::kAdStackBoundReducerWorkgroupSize;
    if (group_x == 0) {
      // Empty dispatch: the matched task has zero threads; record a zero count and skip the dispatch entirely (RHI
      // rejects 0x1x1 dispatches on most backends).
      result[ti] = 0;
      continue;
    }
    RhiResult dispatch_res = reducer_cmdlist->dispatch(group_x, 1, 1);
    QD_ERROR_IF(dispatch_res != RhiResult::success, "adstack bound reducer dispatch error: RhiResult({})",
                int(dispatch_res));
    reducer_cmdlist->buffer_barrier(*adstack_row_counter_buffer_);
  }
  device_->get_compute_stream()->submit_synced(reducer_cmdlist.get());

  // Read back the matched tasks' counter slots into the result map. Tasks that hit the empty-dispatch shortcut above
  // already have entries; the readback overrides them with the (still zero) post-dispatch value, which is consistent.
  {
    void *mapped = nullptr;
    RhiResult map_res = device_->map(*adstack_row_counter_buffer_, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack row counter buffer for readback");
    const uint32_t *slots = reinterpret_cast<const uint32_t *>(mapped);
    for (int ti : matched_task_indices) {
      result[ti] = slots[ti];
    }
    device_->unmap(*adstack_row_counter_buffer_);
  }

  // Clear the counter slots before returning so the main kernel's per-task LCA-block atomic-add (Phase A+B+C) starts
  // from zero. Without this the main pass would observe its slot pre-loaded with the reducer's count and assign row ids
  // in `[count, 2*count)`, indexing past the heap allocation we just sized to `count` rows.
  auto [post_clear_cmdlist, post_clear_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(post_clear_res == RhiResult::success, "Failed to create adstack reducer post-clear cmdlist");
  post_clear_cmdlist->buffer_fill(adstack_row_counter_buffer_->get_ptr(0), needed_counter_bytes, /*data=*/0);
  post_clear_cmdlist->buffer_barrier(*adstack_row_counter_buffer_);
  device_->get_compute_stream()->submit_synced(post_clear_cmdlist.get());

  // Overwrite the matched tasks' capacity slots with their resolved reducer counts. The default fill earlier in this
  // function set every slot to UINT32_MAX; matched tasks now get their exact count so the bounds check at the float
  // LCA-block claim site fires only on a reducer / main divergence. Non-matched tasks keep the UINT32_MAX default and
  // the bounds check stays inert for them.
  {
    void *mapped = nullptr;
    RhiResult map_res =
        device_->map_range(adstack_bound_row_capacity_buffer_->get_ptr(0), needed_capacity_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success,
                   "Failed to map adstack bound row capacity buffer to publish per-task counts");
    uint32_t *slots = reinterpret_cast<uint32_t *>(mapped);
    for (const auto &kv : result) {
      slots[kv.first] = kv.second;
    }
    device_->unmap(*adstack_bound_row_capacity_buffer_);
  }

  return result;
}

}  // namespace gfx
}  // namespace quadrants::lang
