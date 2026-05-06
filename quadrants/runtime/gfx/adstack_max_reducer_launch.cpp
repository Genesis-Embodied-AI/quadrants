// Max-reducer dispatch for SPIR-V backends. Extracted out of `runtime.cpp` for the same reason
// `adstack_bound_reducer_launch.cpp` is - keeps `GfxRuntime::launch_kernel` focused on the main-kernel record/submit
// flow. Conditional on at least one task in the kernel having a non-empty
// `TaskAttributes::AdStackSizingAttribs::max_reducer_specs`. Returns an empty map on devices missing PSB+Int64 caps or
// on kernels with no captured specs; the caller falls through to the per-thread sizer eval, whose `1<<24` cap then
// surfaces as a hard error via the device sizer's overflow-flag slot if the iteration count exceeds the cap.
//
// Per-spec mechanism:
// 1. Pack the cache key `(registry_id, stack_id, mor_node_idx)` and query `AdStackCache::try_max_reducer_cache_hit`. On
// hit, record the cached value in the result map and skip the dispatch.
// 2. On miss, host-evaluate the captured `begin` and `end` subtrees via `evaluate_adstack_size_expr_at_node` (The
// recognizer grammar guarantees both subtrees are closed-form). Skip with -1 length on resolution failure.
// 3. Encode the body subtree into the shared bytecode buffer via `encode_max_reducer_body_bytecode`. The encoder
// extracts reachable nodes in post-order, renumbers to dense `[0, body_node_count)` indices, copies referenced indices
// entries, and resolves each `kExternalTensorRead` leaf's `arg_buffer_offset` via the closure passed here.
// 4. Build the `AdStackMaxReducerParams` blob into the shared params buffer at descriptor-aligned offset.
// 5. Build a single cmdlist with one dispatch per missed spec (each binds the same args/output buffers but a per-spec
// slice of params + bytecode), submit_synced.
// 6. Map the output buffer, read each missed spec's i64 slot into the result map, and call
// `AdStackCache::record_max_reducer_eval` with the body's read observations + the dispatched value so the next launch
// can short-circuit on a generation match.
//
// Caller responsibility: invoke `dispatch_max_reducers` BEFORE `publish_adstack_metadata_spirv` and pass the returned
// map down so the per-task sizer / device sizer encoder can substitute results into per-stack `SerializedSizeExpr`
// trees via `substitute_precomputed_max_over_range`.

#include "quadrants/runtime/gfx/runtime.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "quadrants/codegen/spirv/adstack_max_reducer_shader.h"
#include "quadrants/common/logging.h"
#include "quadrants/ir/adstack_size_expr_device.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/program/launch_context_builder.h"
#include "quadrants/rhi/device.h"

namespace quadrants::lang {
namespace gfx {

namespace {

// Resolve the byte offset within the kernel arg buffer where an ndarray argument's `data_ptr` (u64) lives. Mirrors
// `adstack_bound_reducer_launch.cpp::resolve_ndarray_data_ptr_byte_offset`; centralised in a single helper per launcher
// TU to keep the layout knowledge pinned to one call site per backend.
size_t resolve_ndarray_data_ptr_byte_offset(LaunchContextBuilder &host_ctx, const std::vector<int> &arg_id_path) {
  QD_ASSERT_INFO(host_ctx.args_type != nullptr,
                 "adstack max reducer: LaunchContextBuilder::args_type is null; cannot resolve ndarray data "
                 "pointer offset for the captured spec");
  std::vector<int> indices = arg_id_path;
  indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  return host_ctx.args_type->get_element_offset(indices);
}

// Per-spec dispatch unit, populated from each captured `StaticAdStackMaxReducerSpec` after host-resolving every axis's
// begin / end and encoding the body bytecode. Specs whose body encoding fails or whose total cross-product length is
// zero are dropped before reaching the dispatch loop. `per_axis_*` are ordered outermost-first to match the
// recognizer's chain-capture order; `length` is the cross-product (product of `per_axis_length`).
struct PendingMaxReducerDispatch {
  uint64_t cache_key;
  uint32_t registry_id;
  int32_t stack_id;
  int32_t mor_node_idx;
  uint32_t length;
  uint32_t num_axes;
  std::vector<uint32_t> per_axis_length;
  std::vector<int64_t> per_axis_begin;
  std::vector<uint8_t> body_bytecode;
  uint32_t body_node_count;
  uint32_t indices_count;
  std::vector<AdStackCache::SizeExprReadObservation> reads;
};

// Pack `(registry_id, stack_id, mor_node_idx)` into the same 64-bit key encoding `AdStackCache::pack_max_reducer_key`
// uses internally (low 32 bits = registry_id, mid 16 = stack_id, high 16 = mor_node_idx). Mirrored here rather than
// exposed as a public helper because the caller's need is limited to this TU.
uint64_t pack_max_reducer_key(uint32_t registry_id, int32_t stack_id, int32_t mor_node_idx) {
  return (static_cast<uint64_t>(registry_id) & 0xFFFFFFFFull) | ((static_cast<uint64_t>(stack_id) & 0xFFFFull) << 32) |
         ((static_cast<uint64_t>(mor_node_idx) & 0xFFFFull) << 48);
}

}  // namespace

MaxReducerResultMap GfxRuntime::dispatch_max_reducers(LaunchContextBuilder &host_ctx,
                                                      DeviceAllocationGuard *args_buffer,
                                                      const std::vector<spirv::TaskAttributes> &task_attribs) {
  MaxReducerResultMap result;

  // The shader builder requires `spirv_has_physical_storage_buffer` (PSB body-leaf reads through the kernel arg
  // buffer's data pointers) and `spirv_has_int64` (i64 arithmetic inside the body interpreter, plus i64 begin
  // reassembly). On a device missing either cap, `build_adstack_max_reducer_spirv` returns an empty binary and the lazy
  // pipeline init below would assert. Skip the dispatch entirely and return an empty result map; the caller's
  // substitution helper then leaves every captured `MaxOverRange` in place, so the per-task sizer falls back to its
  // existing capped host-eval path. This call is sequenced before `publish_adstack_metadata_spirv`'s own cap gate so
  // the latter is not load-bearing for this entry-point; we recheck here independently.
  if (!device_->get_caps().get(DeviceCapability::spirv_has_physical_storage_buffer) ||
      !device_->get_caps().get(DeviceCapability::spirv_has_int64)) {
    return result;
  }

  Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
  AdStackCache *cache = (prog != nullptr) ? &prog->adstack_cache() : nullptr;

  // First pass: query the cache per spec, dropping hits straight into `result` and queueing misses for dispatch. Each
  // miss host-resolves `begin` / `end` and encodes the body bytecode up front so the second pass only writes packed
  // params / bytecode bytes and bookkeeps offsets.
  std::vector<PendingMaxReducerDispatch> pending;
  pending.reserve(task_attribs.size());
  for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
    const auto &attribs = task_attribs[ti];
    if (attribs.ad_stack.max_reducer_specs.empty()) {
      continue;
    }
    // Lazily register the task with the Program-side identity registry if `publish_adstack_metadata_spirv` has not run
    // yet for this task. The registration is idempotent so the later publish call is a no-op for already-registered
    // tasks. The cache-key encoding uses `registry_id` to disambiguate same-shape `MaxOverRange` nodes across kernels,
    // so the id has to exist before the first cache lookup.
    auto &mutable_attribs =
        const_cast<quadrants::lang::spirv::TaskAttributes::AdStackSizingAttribs &>(attribs.ad_stack);
    if (mutable_attribs.registry_id == 0 && cache != nullptr) {
      std::vector<int> allocated_max_sizes;
      std::vector<SerializedSizeExpr> size_exprs;
      allocated_max_sizes.reserve(mutable_attribs.allocas.size());
      size_exprs.reserve(mutable_attribs.allocas.size());
      for (const auto &a : mutable_attribs.allocas) {
        allocated_max_sizes.push_back(static_cast<int>(a.max_size_compile_time));
        size_exprs.push_back(a.size_expr);
      }
      mutable_attribs.registry_id = cache->register_adstack_sizing_info(
          static_cast<const void *>(&mutable_attribs), /*kernel_name=*/std::string{}, static_cast<int>(ti),
          std::move(allocated_max_sizes), std::move(size_exprs));
    }
    const uint32_t registry_id = mutable_attribs.registry_id;
    if (registry_id == 0) {
      continue;  // no Program back-ref available; skip the spec
    }
    for (const auto &spec : attribs.ad_stack.max_reducer_specs) {
      const uint64_t key = pack_max_reducer_key(registry_id, spec.stack_id, spec.mor_node_idx);
      if (cache != nullptr) {
        int64_t cached;
        if (cache->try_max_reducer_cache_hit(registry_id, spec.stack_id, spec.mor_node_idx, &host_ctx, cached)) {
          result[key] = cached;
          continue;
        }
      }
      // Multi-axis chain support: host-evaluate `[begin, end)` per captured axis, accumulate the cross-product length,
      // and pass the per-axis arrays into the dispatch. Any axis that fails to host-resolve, or whose product overflows
      // u32, drops the spec (fallback: the per-thread sizer's capped path absorbs it).
      const SerializedSizeExpr &expr = attribs.ad_stack.allocas[spec.stack_id].size_expr;
      const std::size_t num_axes = spec.axis_var_ids.size();
      if (num_axes == 0 || num_axes > static_cast<std::size_t>(kAdStackMaxReducerMaxAxes)) {
        continue;
      }
      std::vector<uint32_t> per_axis_length_v(num_axes, 0);
      std::vector<int64_t> per_axis_begin_v(num_axes, 0);
      bool axes_ok = true;
      uint64_t total_length = 1;
      for (std::size_t a = 0; a < num_axes; ++a) {
        const int64_t bv = evaluate_adstack_size_expr_at_node(expr, spec.axis_begin_node_idxs[a], prog, &host_ctx);
        const int64_t ev = evaluate_adstack_size_expr_at_node(expr, spec.axis_end_node_idxs[a], prog, &host_ctx);
        if (bv < 0 || ev < 0 || ev <= bv) {
          axes_ok = false;
          break;
        }
        const int64_t len = ev - bv;
        if (len > std::numeric_limits<uint32_t>::max()) {
          axes_ok = false;
          break;
        }
        per_axis_begin_v[a] = bv;
        per_axis_length_v[a] = static_cast<uint32_t>(len);
        total_length *= static_cast<uint64_t>(len);
        if (total_length > std::numeric_limits<uint32_t>::max()) {
          axes_ok = false;
          break;
        }
      }
      if (!axes_ok) {
        continue;
      }

      // Encode the body bytecode via `encode_max_reducer_body_bytecode`. The encoder resolves each
      // `kExternalTensorRead` leaf's `arg_buffer_offset` via the closure below, mirroring the SizeExpr device bytecode
      // encoder's offset precomputation, and dense-remaps the chain bound-var ids into device-scope slots `[0,
      // num_axes)`.
      auto arg_buffer_offset_resolver = [&](const std::vector<int32_t> &arg_id_path) -> int32_t {
        std::vector<int> path(arg_id_path.begin(), arg_id_path.end());
        const size_t byte_off = resolve_ndarray_data_ptr_byte_offset(host_ctx, path);
        if (byte_off > std::numeric_limits<int32_t>::max()) {
          return -1;
        }
        return static_cast<int32_t>(byte_off);
      };
      EncodedMaxReducerBody encoded = encode_max_reducer_body_bytecode(expr, spec.body_node_idx, spec.axis_var_ids,
                                                                       arg_buffer_offset_resolver, &host_ctx, prog);
      if (encoded.body_node_count == 0 || encoded.body_node_count > spirv::kAdStackMaxReducerMaxBodyNodes) {
        // Encoder rejected the body (out-of-grammar or too many nodes); skip the spec.
        continue;
      }

      PendingMaxReducerDispatch p{};
      p.cache_key = key;
      p.registry_id = registry_id;
      p.stack_id = spec.stack_id;
      p.mor_node_idx = spec.mor_node_idx;
      p.length = static_cast<uint32_t>(total_length);
      p.num_axes = static_cast<uint32_t>(num_axes);
      p.per_axis_length = std::move(per_axis_length_v);
      p.per_axis_begin = std::move(per_axis_begin_v);
      p.body_bytecode = std::move(encoded.bytes);
      p.body_node_count = encoded.body_node_count;
      p.indices_count = encoded.indices_count;
      p.reads = std::move(encoded.body_reads);
      pending.push_back(std::move(p));
    }
  }

  if (pending.empty()) {
    return result;
  }

  // Lazy-init pipeline. Mirror `adstack_bound_reducer_launch.cpp`'s pattern: build the SPIR-V binary once via the
  // shader-build helper, hand to the device's pipeline factory, cache for the runtime's lifetime.
  if (!adstack_max_reducer_pipeline_) {
    std::vector<uint32_t> spirv = spirv::build_adstack_max_reducer_spirv(Arch::vulkan, &device_->get_caps());
    QD_ASSERT_INFO(!spirv.empty(),
                   "build_adstack_max_reducer_spirv returned an empty binary despite the PSB+Int64 cap "
                   "check passing; bug in the shader builder's capability gating");
    PipelineSourceDesc source_desc{PipelineSourceType::spirv_binary, (void *)spirv.data(),
                                   spirv.size() * sizeof(uint32_t)};
    auto [pipeline, res] = device_->create_pipeline_unique(source_desc, "adstack_max_reducer", backend_cache_.get());
    QD_ERROR_IF(res != RhiResult::success, "Failed to create pipeline for the adstack max reducer (err: {})", int(res));
    adstack_max_reducer_pipeline_ = std::move(pipeline);
  }

  // Pack one params blob per pending spec at descriptor-alignment offsets (256B on the most conservative drivers in the
  // wild; matches the bound reducer's choice). Pack body bytecode at 4-byte alignment within a separate buffer (no
  // descriptor offset alignment needed there because the entire bytecode buffer is bound at offset 0 and each spec's
  // offset is encoded inside its params blob as a u32 word offset).
  constexpr size_t kDescriptorOffsetAlignment = 256;
  auto align_up = [](size_t v, size_t a) { return (v + a - 1) & ~(a - 1); };
  const size_t params_size_bytes = spirv::AdStackMaxReducerParams::kNumWords * sizeof(uint32_t);
  std::vector<size_t> per_spec_params_offsets(pending.size());
  std::vector<uint32_t> per_spec_bytecode_word_offsets(pending.size());
  size_t total_params_bytes = 0;
  size_t total_bytecode_bytes = 0;
  for (size_t k = 0; k < pending.size(); ++k) {
    per_spec_params_offsets[k] = align_up(total_params_bytes, kDescriptorOffsetAlignment);
    total_params_bytes = per_spec_params_offsets[k] + params_size_bytes;
    QD_ASSERT_INFO(pending[k].body_bytecode.size() % sizeof(uint32_t) == 0,
                   "max-reducer body bytecode is not 4-byte aligned (size={})", pending[k].body_bytecode.size());
    per_spec_bytecode_word_offsets[k] = static_cast<uint32_t>(total_bytecode_bytes / sizeof(uint32_t));
    total_bytecode_bytes += pending[k].body_bytecode.size();
  }
  // Output buffer: two u32 slots per pending spec (`[value, overflow_flag]`), cleared to {0, 0} before dispatch.
  // Threads `OpAtomicUMax` into the value slot for body values that fit in `UINT32_MAX`, and `OpAtomicOr 1` into the
  // overflow flag for body values above that. The two-slot layout sidesteps spirv-cross's MSL backend gap on i64
  // atomics (`MSL currently does not support 64-bit atomics`), unlocking Metal / Vulkan-via-MoltenVK paths that would
  // otherwise fail at pipeline-create. The host (`map` + read below) detects overflow per spec and falls back to direct
  // host-eval for that spec via `evaluate_adstack_size_expr_at_node`.
  const size_t output_bytes = pending.size() * 2 * sizeof(uint32_t);

  auto grow_buffer = [&](std::unique_ptr<DeviceAllocationGuard> &buf, size_t &capacity, size_t needed, bool host_write,
                         bool host_read, const char *label) {
    if (buf && capacity >= needed) {
      return;
    }
    size_t new_size = std::max(needed, 2 * capacity);
    auto [new_buf, res] = device_->allocate_memory_unique(
        {new_size, host_write, host_read, /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate {} (size={})", label, new_size);
    if (buf) {
      ctx_buffers_.push_back(std::move(buf));
    }
    buf = std::move(new_buf);
    capacity = new_size;
  };

  grow_buffer(adstack_max_reducer_params_buffer_, adstack_max_reducer_params_buffer_size_, total_params_bytes,
              /*host_write=*/true, /*host_read=*/false, "adstack max reducer params buffer");
  grow_buffer(adstack_max_reducer_bytecode_buffer_, adstack_max_reducer_bytecode_buffer_size_, total_bytecode_bytes,
              /*host_write=*/true, /*host_read=*/false, "adstack max reducer bytecode buffer");
  grow_buffer(adstack_max_reducer_output_buffer_, adstack_max_reducer_output_buffer_size_, output_bytes,
              /*host_write=*/false, /*host_read=*/true, "adstack max reducer output buffer");

  // Write params + bytecode into their respective host-mapped buffers in one pass. Output buffer gets the INT64_MIN
  // sentinel fill via the same map; the dispatch's atomic-SMax then beats this baseline on every matching thread.
  {
    void *mapped = nullptr;
    RhiResult map_res = device_->map_range(adstack_max_reducer_params_buffer_->get_ptr(0), total_params_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack max reducer params buffer");
    for (size_t k = 0; k < pending.size(); ++k) {
      spirv::AdStackMaxReducerParams params{};
      params.output_slot = static_cast<uint32_t>(k);
      params.length = pending[k].length;
      params.num_axes = pending[k].num_axes;
      params.body_bytecode_offset_words = per_spec_bytecode_word_offsets[k];
      params.body_node_count = pending[k].body_node_count;
      // Indices table starts immediately after the body's nodes within the spec's bytecode region. The encoder wrote
      // `[nodes][indices]` contiguously, so the indices base word offset = body_bytecode_offset_words + body_node_count
      // * (sizeof(AdStackSizeExprDeviceNode) / 4).
      const uint32_t node_words = sizeof(AdStackSizeExprDeviceNode) / 4u;
      params.body_indices_offset_words = per_spec_bytecode_word_offsets[k] + pending[k].body_node_count * node_words;
      // Per-axis fields (slots beyond `num_axes` left zero by the value-init above so the shader's bounds checks never
      // read uninitialised data on lower-rank specs). The encoder dense-remaps each chain bound-var id to slot `a` in
      // `[0, num_axes)`; encode that slot directly here so the shader's `kBoundVariable` / `kExternalTensorRead` cases
      // land at the right scope index.
      for (uint32_t a = 0; a < pending[k].num_axes; ++a) {
        params.per_axis_length[a] = pending[k].per_axis_length[a];
        const uint64_t begin_u64 = static_cast<uint64_t>(pending[k].per_axis_begin[a]);
        params.per_axis_begin_lo[a] = static_cast<uint32_t>(begin_u64 & 0xFFFFFFFFull);
        params.per_axis_begin_hi[a] = static_cast<uint32_t>((begin_u64 >> 32) & 0xFFFFFFFFull);
        params.per_axis_var_id[a] = static_cast<int32_t>(a);
      }
      std::memcpy(reinterpret_cast<char *>(mapped) + per_spec_params_offsets[k], &params, params_size_bytes);
    }
    device_->unmap(*adstack_max_reducer_params_buffer_);
  }
  if (total_bytecode_bytes > 0) {
    void *mapped = nullptr;
    RhiResult map_res =
        device_->map_range(adstack_max_reducer_bytecode_buffer_->get_ptr(0), total_bytecode_bytes, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack max reducer bytecode buffer");
    char *base = reinterpret_cast<char *>(mapped);
    size_t cursor = 0;
    for (size_t k = 0; k < pending.size(); ++k) {
      std::memcpy(base + cursor, pending[k].body_bytecode.data(), pending[k].body_bytecode.size());
      cursor += pending[k].body_bytecode.size();
    }
    device_->unmap(*adstack_max_reducer_bytecode_buffer_);
  }
  // Slot-0 placeholder for kernels with no kernel arg buffer. Same RHI rule as the bound reducer: descriptor-set
  // layouts require a non-null binding even if the shader's branch never reads it.
  if (args_buffer == nullptr && !adstack_max_reducer_args_placeholder_buffer_) {
    auto [buf, res] = device_->allocate_memory_unique({sizeof(uint32_t), /*host_write=*/false, /*host_read=*/false,
                                                       /*export_sharing=*/false, AllocUsage::Storage});
    QD_ASSERT_INFO(res == RhiResult::success, "Failed to allocate adstack max reducer slot-0 placeholder buffer");
    adstack_max_reducer_args_placeholder_buffer_ = std::move(buf);
  }

  // Force visibility of prior writes (matches `adstack_bound_reducer_launch.cpp`'s rationale): MoltenVK's PSB load path
  // bypasses the descriptor-bound cache that a prior accessor kernel's submit_synced flushed via vkQueueWaitIdle, so
  // without this sequence the reducer reads stale ndarray contents on Apple Silicon.
  flush();
  device_->wait_idle();

  // Zero the per-spec output slots through a fresh GPU cmdlist instead of a host `map_range` + `memcpy`. The host-side
  // clear is not guaranteed to be fenced ahead of the next compute pipeline read on every backend / driver combination
  // - Apple M1 leaves the host write sitting in a write-combined cache and the dispatch reads stale bytes (typically
  // the previous launch's max), so the atomic-UMax never beats them. A GPU-side `buffer_fill` is sequenced by the
  // compute queue's own dependency tracking and avoids the host-vs-GPU coherence question entirely. Mirrors
  // `adstack_bound_reducer_launch.cpp`'s `buffer_fill`-via-separate-cmdlist pattern.
  auto [clear_cmdlist, clear_cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(clear_cmdlist_res == RhiResult::success, "Failed to create adstack max reducer clear cmdlist");
  clear_cmdlist->buffer_fill(adstack_max_reducer_output_buffer_->get_ptr(0), output_bytes, /*data=*/0);
  clear_cmdlist->buffer_barrier(*adstack_max_reducer_output_buffer_);
  device_->get_compute_stream()->submit_synced(clear_cmdlist.get());

  // Build a single cmdlist with one dispatch per pending spec.
  auto [cmdlist, cmdlist_res] = device_->get_compute_stream()->new_command_list_unique();
  QD_ASSERT_INFO(cmdlist_res == RhiResult::success, "Failed to create adstack max reducer cmdlist");
  for (size_t k = 0; k < pending.size(); ++k) {
    auto bindings = device_->create_resource_set_unique();
    if (args_buffer != nullptr) {
      bindings->rw_buffer(0, *args_buffer);
    } else {
      bindings->rw_buffer(0, *adstack_max_reducer_args_placeholder_buffer_);
    }
    bindings->rw_buffer(1, *adstack_max_reducer_output_buffer_);
    bindings->rw_buffer(2, adstack_max_reducer_params_buffer_->get_ptr(per_spec_params_offsets[k]), params_size_bytes);
    bindings->rw_buffer(3, *adstack_max_reducer_bytecode_buffer_);

    cmdlist->bind_pipeline(adstack_max_reducer_pipeline_.get());
    RhiResult bind_res = cmdlist->bind_shader_resources(bindings.get());
    QD_ERROR_IF(bind_res != RhiResult::success, "adstack max reducer resource binding error: RhiResult({})",
                int(bind_res));

    // Each thread walks `kElementsPerThread` elements via a strided loop inside the shader, so the host can cap
    // workgroup count well below the Vulkan / Metal `maxComputeWorkGroupCount[0]` minimum (65535) without dropping any
    // elements. With workgroup_size = 128 and elements_per_thread = 64 the dispatch covers up to 65535 * 128 * 64 =
    // ~536M elements per spec, two orders of magnitude above any reverse-mode workload the recognizer captures in
    // practice. Keep the constant in sync with `kElementsPerThread` in `adstack_max_reducer_shader.cpp`.
    constexpr uint32_t kElementsPerThread = 64u;
    constexpr uint32_t kMaxWorkgroupCountX = 65535u;
    const uint32_t threads_per_workgroup = spirv::kAdStackMaxReducerWorkgroupSize;
    const uint32_t elements_per_workgroup = threads_per_workgroup * kElementsPerThread;
    uint32_t group_x = (pending[k].length + elements_per_workgroup - 1) / elements_per_workgroup;
    if (group_x > kMaxWorkgroupCountX) {
      group_x = kMaxWorkgroupCountX;
    }
    if (group_x == 0) {
      // Empty range; record 0 directly. Caller's substitute helper treats 0 as the dispatched value (per-thread sizer's
      // `kConst 0` evaluator floors at 1 downstream). RHI rejects 0x1x1 dispatches on most backends.
      result[pending[k].cache_key] = 0;
      continue;
    }
    RhiResult dispatch_res = cmdlist->dispatch(group_x, 1, 1);
    QD_ERROR_IF(dispatch_res != RhiResult::success, "adstack max reducer dispatch error: RhiResult({})",
                int(dispatch_res));
    cmdlist->buffer_barrier(*adstack_max_reducer_output_buffer_);
  }
  device_->get_compute_stream()->submit_synced(cmdlist.get());

  // Read back per-spec output slots: `slots[2*k]` = u32 max, `slots[2*k + 1]` = overflow flag. Overflow specs fall back
  // to direct host-eval over the `MaxOverRange` node; the host evaluator's `1<<24` cap surfaces the hard error path on
  // out-of-grammar shapes that exceed both u32 (body value) and the cap (iteration count).
  {
    void *mapped = nullptr;
    RhiResult map_res = device_->map(*adstack_max_reducer_output_buffer_, &mapped);
    QD_ASSERT_INFO(map_res == RhiResult::success, "Failed to map adstack max reducer output buffer for readback");
    const uint32_t *slots = reinterpret_cast<const uint32_t *>(mapped);
    for (size_t k = 0; k < pending.size(); ++k) {
      const uint32_t value_u32 = slots[2 * k];
      const uint32_t overflow_flag = slots[2 * k + 1];
      int64_t v;
      if (overflow_flag != 0) {
        // Host eval over the captured `MaxOverRange` node. Re-resolve `expr` from the per-task attribs by matching
        // `(stack_id, mor_node_idx)` against the recognized specs; on the gfx side we look up
        // `task_attribs[ti]->ad_stack.size_exprs[stack_id]` indirectly via the cache key triplet.
        const SerializedSizeExpr *expr = nullptr;
        for (size_t ti = 0; ti < task_attribs.size(); ++ti) {
          if (task_attribs[ti].ad_stack.registry_id != pending[k].registry_id) {
            continue;
          }
          const auto &allocas = task_attribs[ti].ad_stack.allocas;
          if (pending[k].stack_id < 0 || static_cast<size_t>(pending[k].stack_id) >= allocas.size()) {
            continue;
          }
          expr = &allocas[pending[k].stack_id].size_expr;
          break;
        }
        if (expr != nullptr) {
          v = evaluate_adstack_size_expr_at_node(*expr, pending[k].mor_node_idx, prog, &host_ctx);
          if (v < 0) {
            v = 0;
          }
        } else {
          v = 0;  // unrecoverable; leave as zero so the per-task sizer floors at 1
        }
      } else {
        v = static_cast<int64_t>(value_u32);
      }
      result[pending[k].cache_key] = v;
      if (cache != nullptr) {
        populate_max_reducer_body_observations(pending[k].reads, &host_ctx, cache);
        cache->record_max_reducer_eval(pending[k].registry_id, pending[k].stack_id, pending[k].mor_node_idx, v,
                                       std::move(pending[k].reads));
      }
    }
    device_->unmap(*adstack_max_reducer_output_buffer_);
  }

  return result;
}

}  // namespace gfx
}  // namespace quadrants::lang
