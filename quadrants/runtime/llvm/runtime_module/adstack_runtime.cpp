// Adstack runtime helpers for the LLVM bitcode runtime. This file is `#include`d once from runtime.cpp so the
// bitcode build stays a single translation unit (see adstack_runtime.h for why a real two-TU split via `llvm-link`
// is broken by LLVM's named-struct type uniquing). It sees runtime.cpp's preamble - type aliases, the `STRUCT_FIELD`
// macro, the `LLVMRuntime` / `RuntimeContext` struct definitions - through that include site, mirroring how
// `node_*.h` / `internal_functions.h` / `locked_task.h` are factored out today.
//
// See adstack_runtime.h for the public API documentation; the file-scope comments in each function below cover
// the implementation details.

// STRUCT_FIELD getters for the adstack-prefixed `LLVMRuntime` fields. The struct fields themselves live in
// llvm_runtime.h (they are part of the struct memory layout), but the macro-generated `extern "C"` getters / setters
// can live anywhere in the TU - co-locating them here keeps the adstack surface in one place.
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride);
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer_float);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size_float);
STRUCT_FIELD(LLVMRuntime, adstack_heap_buffer_int);
STRUCT_FIELD(LLVMRuntime, adstack_heap_size_int);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride_float);
STRUCT_FIELD(LLVMRuntime, adstack_per_thread_stride_int);
STRUCT_FIELD(LLVMRuntime, adstack_offsets);
STRUCT_FIELD(LLVMRuntime, adstack_max_sizes);
STRUCT_FIELD(LLVMRuntime, adstack_row_counters);
STRUCT_FIELD(LLVMRuntime, adstack_bound_row_capacities);
STRUCT_FIELD(LLVMRuntime, adstack_max_reducer_outputs);
STRUCT_FIELD(LLVMRuntime, adstack_overflow_flag_dev_ptr);
STRUCT_FIELD(LLVMRuntime, adstack_overflow_task_id_dev_ptr);

// Writes the addresses of `runtime->adstack_heap_buffer` and `runtime->adstack_heap_size` into the result buffer so the
// host-side executor can cache them. With those cached device pointers the host grows the heap by issuing two simple
// `memcpy_host_to_device` writes - no per-grow kernel launch for the setters, which sidesteps any questions about
// AMDGPU kernel calling convention on the auto-generated STRUCT_FIELD setters vs the hand-written `runtime_*` wrappers.
// Writes the addresses of the legacy combined-heap fields into the result buffer so the host caches them and then
// issues per-launch grows via `memcpy_host_to_device` to the cached pointers. Returns two addresses: combined-heap-ptr,
// combined-heap-size. The split-heap path uses a separate getter below.
extern "C" void runtime_get_adstack_heap_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_heap_buffer);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_heap_size);
}

// Per-kind heap field getters for the split-heap path. Returns four addresses in fixed slot order: float-buffer-ptr,
// float-size, int-buffer-ptr, int-size.
extern "C" void runtime_get_adstack_split_heap_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_heap_buffer_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_heap_size_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 2, (u64)(void *)&runtime->adstack_heap_buffer_int);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 3, (u64)(void *)&runtime->adstack_heap_size_int);
}

// Mirrors `runtime_get_adstack_heap_field_ptrs` for the per-launch metadata fields. The host caches the four returned
// addresses once per program and then publishes new values (combined stride + offsets array pointer + max_sizes array
// pointer + float stride + int stride) before every kernel launch via the same `memcpy_host_to_device` / direct-store
// path used for the heap buffers. Slots 0/1/2 keep the legacy ordering (combined-stride, offsets, max_sizes) so any
// host code that has not migrated still works; slots 3/4 are the new per-kind strides.
extern "C" void runtime_get_adstack_metadata_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_per_thread_stride);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_offsets);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 2, (u64)(void *)&runtime->adstack_max_sizes);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 3, (u64)(void *)&runtime->adstack_per_thread_stride_float);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 4, (u64)(void *)&runtime->adstack_per_thread_stride_int);
}

// Writes the addresses of the per-task lazy-claim counter and bound-row-capacity arrays into the result buffer so the
// host caches them once. The arrays themselves are device-resident; the host publishes the array pointers via
// `memcpy_host_to_device` to the cached field addresses whenever the per-task slot count grows beyond the prior
// allocation.
extern "C" void runtime_get_adstack_lazy_claim_field_ptrs(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_row_counters);
  runtime->set_result(quadrants_result_buffer_ret_value_id + 1, (u64)(void *)&runtime->adstack_bound_row_capacities);
}

// Companion to `runtime_get_adstack_lazy_claim_field_ptrs` for the max-reducer outputs. The output buffer is
// per-launch-allocated host-side and the field-address is cached once so the per-launch publish only writes the new
// array pointer (when the buffer grows) and the read-back per-spec slot reads through the runtime's stable address.
// Single field, single result slot.
extern "C" void runtime_get_adstack_max_reducer_field_ptr(LLVMRuntime *runtime) {
  runtime->set_result(quadrants_result_buffer_ret_value_id, (u64)(void *)&runtime->adstack_max_reducer_outputs);
}

// Device-resident adstack SizeExpr interpreter. Runs on whatever backend the LLVM runtime JIT-compiles this
// bitcode to: a plain C function call on CPU, a single-thread kernel launch on CUDA / AMDGPU. The bytecode buffer
// layout is defined by `quadrants/ir/adstack_size_expr_device.h` and produced host-side by
// `encode_adstack_size_expr_device_bytecode` immediately before this call.
//
// For every alloca slot the interpreter walks its tree (recursive descent over node indices that point strictly
// backwards) and writes:
//   - `runtime->adstack_max_sizes[i]` = `clamp(tree_value, 1, max_size_compile_time)` if the tree is non-empty,
//     else `max_size_compile_time`. The compile-time cap is the structural upper bound the pre-pass proved, so
//     the clamp only ever tightens against a buggy tree evaluation; the `max(_, 1)` preserves the "always room
//     for one push" invariant the runtime's `stack_push` relies on.
//   - `runtime->adstack_offsets[i]` = cumulative byte offset inside the per-thread slice.
//   - `runtime->adstack_per_thread_stride` = final running sum (after last alloca).
// The host reads back `adstack_per_thread_stride` via the cached field pointer to size the heap with
// `ensure_adstack_heap`; the offsets / max_sizes arrays stay device-resident and feed the main kernel directly.
//
// Ndarray element access (`ExternalTensorRead`) reads `ctx->arg_buffer` at the `arg_buffer_offset` encoded into
// the node to fetch the data pointer, then indexes by the linear offset computed from the node's indices. There
// is no `array_ptrs` map on device; the host-side encoder has already resolved `arg_id -> arg_buffer_offset`
// through the kernel's `args_type` struct layout.
//
// Recursion bounded by tree depth (typically <10 for observed reverse-mode kernels, <30 worst case). The
// bound-variable scope is kept in a fixed-size array indexed by `var_id`; the host encoder dense-remaps each
// tree's `var_id`s into `[0, kDeviceBoundVarCap)` before emitting bytecode and hard-errors above the cap, so
// `values[var_id]` is always in bounds here.

namespace {

constexpr int kDeviceBoundVarCap = quadrants::lang::kAdStackSizeExprDeviceMaxBoundVars;

struct DeviceEvalScope {
  // Bound-var lookup by `var_id`. Unbound slots are sentinelled by the caller before the interpreter enters the
  // subtree; walking the code paths that read `values[vid]` without a matching `MaxOverRange` bind would be a
  // pre-pass bug. The interpreter does not validate - on GPU backends we cannot afford a host-style assert from
  // device code, so a buggy tree is caught through wrong max_size values and an overflow at `stack_push` rather
  // than a fatal trap here.
  i64 values[kDeviceBoundVarCap];
};

__attribute__((always_inline)) inline i64 device_load_element(const char *data_ptr, i64 linear, i32 prim_dt) {
  // Enum values mirror `PrimitiveTypeID` in `quadrants/inc/data_type.inc.h` (f16=0, f32=1, f64=2, i8=3, i16=4,
  // i32=5, i64=6, u1=7, u8=8, u16=9, u32=10, u64=11). The pre-pass only emits integer reads (the adstack-size
  // grammar rejects float-typed reads at build_value_expr), so we only decode the integer types here.
  switch (prim_dt) {
    case 3:  // i8
      return (i64) reinterpret_cast<const i8 *>(data_ptr)[linear];
    case 4:  // i16
      return (i64) reinterpret_cast<const i16 *>(data_ptr)[linear];
    case 5:  // i32
      return (i64) reinterpret_cast<const i32 *>(data_ptr)[linear];
    case 6:  // i64
      return reinterpret_cast<const i64 *>(data_ptr)[linear];
    case 8:  // u8
      return (i64) reinterpret_cast<const u8 *>(data_ptr)[linear];
    case 9:  // u16
      return (i64) reinterpret_cast<const u16 *>(data_ptr)[linear];
    case 10:  // u32
      return (i64) reinterpret_cast<const u32 *>(data_ptr)[linear];
    case 11:  // u64
      return (i64) reinterpret_cast<const u64 *>(data_ptr)[linear];
    default:
      return 0;  // unreachable: encoder rejects other types
  }
}

i64 device_eval_node(LLVMRuntime *runtime,
                     const quadrants::lang::AdStackSizeExprDeviceNode *nodes,
                     const i32 *indices,
                     i32 node_idx,
                     DeviceEvalScope *scope,
                     const char *arg_buffer) {
  const auto &node = nodes[node_idx];
  using K = quadrants::lang::AdStackSizeExprDeviceKind;
  switch (static_cast<K>(node.kind)) {
    case K::kConst:
      return node.const_value;
    case K::kAdd:
      return device_eval_node(runtime, nodes, indices, node.operand_a, scope, arg_buffer) +
             device_eval_node(runtime, nodes, indices, node.operand_b, scope, arg_buffer);
    case K::kSub: {
      // Match the host evaluator: clamp negative trip counts to zero so an underflowed `end - begin` doesn't poison a
      // surrounding `Mul` / `MaxOverRange` product.
      i64 lhs = device_eval_node(runtime, nodes, indices, node.operand_a, scope, arg_buffer);
      i64 rhs = device_eval_node(runtime, nodes, indices, node.operand_b, scope, arg_buffer);
      i64 diff = lhs - rhs;
      return diff > 0 ? diff : 0;
    }
    case K::kMul:
      return device_eval_node(runtime, nodes, indices, node.operand_a, scope, arg_buffer) *
             device_eval_node(runtime, nodes, indices, node.operand_b, scope, arg_buffer);
    case K::kMax: {
      i64 lhs = device_eval_node(runtime, nodes, indices, node.operand_a, scope, arg_buffer);
      i64 rhs = device_eval_node(runtime, nodes, indices, node.operand_b, scope, arg_buffer);
      return lhs > rhs ? lhs : rhs;
    }
    case K::kMaxOverRange: {
      i64 begin = device_eval_node(runtime, nodes, indices, node.operand_a, scope, arg_buffer);
      i64 end = device_eval_node(runtime, nodes, indices, node.operand_b, scope, arg_buffer);
      // Iteration guard. Recognized `MaxOverRange` shapes are dispatched in parallel by the max-reducer and substituted
      // to a `Const` before the sizer interpreter walks the tree, so the only way to land in this branch with a delta
      // above the cap is an out-of-grammar shape. Skip the walk and return 0 to keep the single-thread on-device
      // dispatch within the driver's TDR window; the host's `evaluate_node` re-runs the same tree synchronously during
      // the diagnose path and raises via its `QD_ERROR_IF` then.
      constexpr i64 kMaxOverRangeIterations = i64{1} << 24;
      if (end > begin && end - begin > kMaxOverRangeIterations) {
        return 0;
      }
      i64 result = 0;
      const i32 var = node.var_id;
      for (i64 i = begin; i < end; ++i) {
        if (var >= 0 && var < kDeviceBoundVarCap) {
          scope->values[var] = i;
        }
        i64 v = device_eval_node(runtime, nodes, indices, node.body_node_idx, scope, arg_buffer);
        if (v > result)
          result = v;
      }
      return result;
    }
    case K::kBoundVariable: {
      const i32 var = node.var_id;
      if (var >= 0 && var < kDeviceBoundVarCap)
        return scope->values[var];
      return 0;
    }
    case K::kExternalTensorRead: {
      // `data_ptr_slot = *(void **)(arg_buffer + arg_buffer_offset)`: read the ndarray's data pointer out of the kernel
      // arg buffer at the offset the host encoder precomputed via `args_type->get_element_offset`. This replaces the
      // host evaluator's `ctx->array_ptrs` map lookup with a straight field read that the device can perform without
      // reaching for a std::unordered_map.
      auto data_ptr_raw = *reinterpret_cast<const char *const *>(arg_buffer + node.arg_buffer_offset);
      // Indices encoded as `[idx_a_raw, elem_stride_a]` pairs per axis, matching `kFieldLoad`'s layout. The host
      // encoder in `adstack_size_expr_eval.cpp` pre-computes the C-order element strides from the launch context's
      // ndarray shape; a 1-D read collapses to `elem_stride = 1` and recovers the original stride-1 sum. The multi-axis
      // case is what this fix unblocks: without the per-axis multiply a 2-D `a[i, j]` read would land on
      // `a_flat[i + j]` instead of `a_flat[i * shape[1] + j]`, under-bounding the sizer and tripping `Adstack overflow`
      // at `qd.sync()`.
      i64 linear = 0;
      for (i32 k = 0; k < node.indices_count; ++k) {
        const i32 raw = indices[node.indices_offset + 2 * k];
        const i32 elem_stride = indices[node.indices_offset + 2 * k + 1];
        i64 v = 0;
        if (raw >= 0) {
          v = raw;
        } else {
          const i32 var = -(raw + 1);
          if (var >= 0 && var < kDeviceBoundVarCap)
            v = scope->values[var];
        }
        linear += v * static_cast<i64>(elem_stride);
      }
      return device_load_element(data_ptr_raw, linear, node.prim_dt);
    }
    case K::kFieldLoad: {
      // Bound-var-indexed `kFieldLoad` body leaf: the encoder stores `arg_buffer_offset = snode_root_id` and
      // `const_value = place_byte_offset_in_root` (i.e. the byte offset of the place leaf within its containing snode
      // tree). The base pointer is `runtime->roots[snode_root_id]`, which lives on every LLVM backend (CPU host pointer
      // / CUDA / AMDGPU device pointer set up at materialization time). The closed-FieldLoad path host-folds at encode
      // time and never reaches this arm; a `snode_root_id < 0` here means the bytecode came from a SPIR-V encoder
      // (which stores `root_psb + place_byte_offset` directly in `const_value` and leaves `arg_buffer_offset = -1`),
      // not the LLVM path; we cannot resolve it here so return 0 (safe over-approximation - a sentinel max forces the
      // host to fall back to capped sizer eval downstream).
      const i32 snode_root_id = node.arg_buffer_offset;
      if (snode_root_id < 0 || runtime == nullptr) {
        return 0;
      }
      const auto root_ptr = reinterpret_cast<const char *>(runtime->roots[snode_root_id]);
      const i64 place_byte_off = node.const_value;
      i64 elem_idx = 0;
      for (i32 k = 0; k < node.indices_count; ++k) {
        const i32 raw = indices[node.indices_offset + 2 * k];
        const i32 elem_stride = indices[node.indices_offset + 2 * k + 1];
        i64 v = 0;
        if (raw >= 0) {
          v = raw;
        } else {
          const i32 var = -(raw + 1);
          if (var >= 0 && var < kDeviceBoundVarCap)
            v = scope->values[var];
        }
        elem_idx += v * static_cast<i64>(elem_stride);
      }
      return device_load_element(root_ptr + place_byte_off, elem_idx, node.prim_dt);
    }
  }
  return 0;
}

}  // namespace

// Per-arch reducer counterpart to the SPIR-V `adstack_bound_reducer_shader.cpp` compute kernel: a single-thread serial
// function that walks the captured gating ndarray over `[0, length)`, evaluates the comparison + polarity at each
// thread index, and writes the gate-passing count into `runtime->adstack_bound_row_capacities[task_index]`. The
// codegen-emitted clamp at the float LCA-block claim site reads that slot back, so on backends that have a working
// reducer the bounds clamp activates per task and a future commit can size the float heap from the count instead of the
// dispatched-threads worst case.
//
// Single-thread execution is intentional: dispatching this as a parallel kernel would need a separate JIT-compiled
// compute kernel with atomic-add semantics per arch (the SPIR-V path emits a parallel reducer; LLVM's runtime functions
// go through `runtime_jit->call` which runs serially - on CUDA / AMDGPU it is a 1x1x1 grid kernel launch, on CPU a
// regular function call). For typical iteration bounds (a few hundred thousand on the largest reverse-mode kernels), a
// single device thread completes the count in well under a millisecond per task; that cost is dominated by the actual
// main kernel anyway.
//
// Both ndarray-backed and SNode-backed sources are dispatched through this function: the params blob's
// `field_source_is_snode` flag selects between reading the gating field through the kernel arg buffer (ndarray) or
// through `runtime->roots[snode_root_id]` (SNode), and the comparison + count loop is shared.
extern "C" void runtime_eval_static_bound_count(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr params_blob) {
  using quadrants::lang::kLlvmReducerCmpEq;
  using quadrants::lang::kLlvmReducerCmpGe;
  using quadrants::lang::kLlvmReducerCmpGt;
  using quadrants::lang::kLlvmReducerCmpLe;
  using quadrants::lang::kLlvmReducerCmpLt;
  using quadrants::lang::kLlvmReducerCmpNe;
  using quadrants::lang::LlvmAdStackBoundReducerDeviceParams;

  const auto *params = reinterpret_cast<const LlvmAdStackBoundReducerDeviceParams *>(params_blob);

  // Resolve the gating field's per-cell pointer + stride based on `field_source_is_snode`. The two source shapes share
  // the comparison + count loop below; only the per-`gid` element load differs.
  //   - ndarray (`field_source_is_snode == 0`): walk `data_ptr[i]` where `data_ptr` is reconstructed from the
  //     kernel arg buffer at `arg_word_offset` (u64 stored across two adjacent u32 words). The element stride is
  //     `sizeof(float)` / `sizeof(i32)` since ndarray data is densely packed by index.
  //   - SNode (`field_source_is_snode == 1`): walk `runtime->roots[snode_root_id] + snode_byte_base_offset +
  //     gid * snode_byte_cell_stride`. The base byte offset and cell stride were pre-resolved at codegen time by
  //     walking the SNode descriptor chain. Mirrors the SPIR-V reducer's `field_source_is_snode` branch.
  const char *field_base = nullptr;
  u32 element_stride_bytes = 0u;
  if (params->field_source_is_snode != 0u) {
    field_base = reinterpret_cast<const char *>(runtime->roots[params->snode_root_id]) + params->snode_byte_base_offset;
    element_stride_bytes = params->snode_byte_cell_stride;
  } else {
    const u32 *arg_buffer_u32 = reinterpret_cast<const u32 *>(ctx->arg_buffer);
    const u64 lo = static_cast<u64>(arg_buffer_u32[params->arg_word_offset]);
    const u64 hi = static_cast<u64>(arg_buffer_u32[params->arg_word_offset + 1]);
    field_base = reinterpret_cast<const char *>(lo | (hi << 32));
    // f32 / i32 share the 4-byte ndarray stride; f64 needs 8 bytes per cell.
    element_stride_bytes = (params->field_dtype_is_float != 0u && params->field_dtype_is_double != 0u)
                               ? 8u
                               : static_cast<u32>(sizeof(u32));
  }

  u32 count = 0;
  if (params->field_dtype_is_float != 0u && params->field_dtype_is_double != 0u) {
    // f64 path: reassemble the 64-bit threshold from the two u32 halves the host packed into the params blob, bitcast
    // to double, then walk the source ndarray as `double *`. f64 thresholds keep the user's full f64 precision;
    // narrowing to f32 here would risk a wrong count on gates whose threshold sits within an f32 representable gap.
    double threshold;
    u64 bits64 = static_cast<u64>(params->threshold_bits) | (static_cast<u64>(params->threshold_bits_high) << 32);
    __builtin_memcpy(&threshold, &bits64, sizeof(double));
    for (u32 i = 0; i < params->length; ++i) {
      const double v = *reinterpret_cast<const double *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  } else if (params->field_dtype_is_float != 0u) {
    float threshold;
    {
      // Bitcast the threshold's u32 storage back to f32. memcpy keeps the LLVM IR semantics-clean (no aliasing) and
      // compiles to a single load on every supported arch.
      u32 bits = params->threshold_bits;
      __builtin_memcpy(&threshold, &bits, sizeof(float));
    }
    for (u32 i = 0; i < params->length; ++i) {
      const float v = *reinterpret_cast<const float *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  } else {
    const i32 threshold = static_cast<i32>(params->threshold_bits);
    for (u32 i = 0; i < params->length; ++i) {
      const i32 v = *reinterpret_cast<const i32 *>(field_base + (u64)i * element_stride_bytes);
      bool match;
      switch (params->cmp_op) {
        case kLlvmReducerCmpLt:
          match = v < threshold;
          break;
        case kLlvmReducerCmpLe:
          match = v <= threshold;
          break;
        case kLlvmReducerCmpGt:
          match = v > threshold;
          break;
        case kLlvmReducerCmpGe:
          match = v >= threshold;
          break;
        case kLlvmReducerCmpEq:
          match = v == threshold;
          break;
        case kLlvmReducerCmpNe:
          match = v != threshold;
          break;
        default:
          match = false;
          break;
      }
      if ((params->polarity != 0u) ? match : !match) {
        ++count;
      }
    }
  }

  runtime->adstack_bound_row_capacities[params->task_index] = count;
}

// per-launch parallel-max evaluator over the body of a captured `StaticAdStackMaxReducerSpec`'s `MaxOverRange` node.
// Two entry points share the same per-iteration body so CPU keeps a single in-process call and GPU saturates the
// dispatch:
//   - `_serial` (CPU): one thread walks the full cross-product, writes the result directly to `output_slot`. The
//     host dispatcher caller does not seed the slot - this entry point publishes the final value (or the INT64_MIN
//     sentinel for an empty / out-of-bound axis count).
//   - `_parallel` (CUDA / AMDGPU): grid-strided walk; each thread accumulates a per-thread running max and atomically
//     reduces into `output_slot` with `atomic_max_i64`. The host dispatcher seeds the slot to INT64_MIN before
//     launching so an empty cross-product leaves the sentinel for the host launcher to detect and floor at zero.
//
// The body bytecode reuses the existing `AdStackSizeExprDeviceNode` POD format already shared between the host encoder
// and the LLVM device sizer interpreter (`device_eval_node`). The recognizer grammar restricts body kinds to `kConst /
// kBoundVariable / kExternalTensorRead / kAdd / kSub / kMul / kMax`, so the recursive walk never recurses through
// `kMaxOverRange` or `kFieldLoad` and the iteration stays linear in the cross-product of every captured axis.
//
// Multi-axis: walks the cross-product of `params->per_axis_length[0..num_axes)` outermost-first. Per-iteration the
// runtime pre-populates `scope.values[per_axis_var_id[a]] = per_axis_begin[a] + axis_idx_a` for every axis, then
// evaluates `device_eval_node(body_root_idx, &scope, ...)` and updates the running max.
namespace {
// Iterative post-order evaluator for the max-reducer body. The recursive `device_eval_node` poisons NVPTX
// codegen: the backend treats the recursive call as undef-returning, which folds the surrounding conditional to
// a constant and DCE's the entire post-loop atomic-reduce block. The recognizer accepts max-reducer body kinds
// `kConst / kBoundVariable / kExternalTensorRead / kFieldLoad / kAdd / kSub / kMul / kMax` (no `kMaxOverRange`),
// so a fixed-size stack walk of the post-order body bytecode covers every accepted shape with no recursion.
// Stack depth is bounded by tree depth; recognized bodies are tiny (a few nodes), so 32 slots is plenty.
constexpr int kMaxReducerEvalStackDepth = 32;

__attribute__((always_inline)) inline i64 device_eval_max_reduce_body_iterative(
    LLVMRuntime *runtime,
    const quadrants::lang::AdStackSizeExprDeviceNode *nodes,
    const i32 *indices,
    i32 body_node_count,
    DeviceEvalScope *scope,
    const char *arg_buffer) {
  using K = quadrants::lang::AdStackSizeExprDeviceKind;
  i64 stack[kMaxReducerEvalStackDepth];
  i32 sp = 0;
  for (i32 idx = 0; idx < body_node_count; ++idx) {
    const auto &node = nodes[idx];
    switch (static_cast<K>(node.kind)) {
      case K::kConst:
        stack[sp++] = node.const_value;
        break;
      case K::kBoundVariable: {
        const i32 var = node.var_id;
        i64 v = 0;
        if (var >= 0 && var < kDeviceBoundVarCap) {
          v = scope->values[var];
        }
        stack[sp++] = v;
        break;
      }
      case K::kExternalTensorRead: {
        auto data_ptr_raw = *reinterpret_cast<const char *const *>(arg_buffer + node.arg_buffer_offset);
        i64 linear = 0;
        for (i32 k = 0; k < node.indices_count; ++k) {
          const i32 raw = indices[node.indices_offset + 2 * k];
          const i32 elem_stride = indices[node.indices_offset + 2 * k + 1];
          i64 v = 0;
          if (raw >= 0) {
            v = raw;
          } else {
            const i32 var = -(raw + 1);
            if (var >= 0 && var < kDeviceBoundVarCap)
              v = scope->values[var];
          }
          linear += v * static_cast<i64>(elem_stride);
        }
        stack[sp++] = device_load_element(data_ptr_raw, linear, node.prim_dt);
        break;
      }
      case K::kFieldLoad: {
        // Bound-var-indexed `kFieldLoad` body leaf: encoder stores `arg_buffer_offset = snode_root_id` and
        // `const_value = place_byte_offset_in_root` for the LLVM path. Closed FieldLoads are host-folded to `kConst`
        // at encode time; SPIR-V-encoded bytecode (snode_root_id < 0) cannot be resolved here, so push 0 as a safe
        // over-approximation - the host diagnose path re-runs synchronously and raises if needed.
        const i32 snode_root_id = node.arg_buffer_offset;
        if (snode_root_id < 0 || runtime == nullptr) {
          stack[sp++] = 0;
          break;
        }
        const auto root_ptr = reinterpret_cast<const char *>(runtime->roots[snode_root_id]);
        const i64 place_byte_off = node.const_value;
        i64 elem_idx = 0;
        for (i32 k = 0; k < node.indices_count; ++k) {
          const i32 raw = indices[node.indices_offset + 2 * k];
          const i32 elem_stride = indices[node.indices_offset + 2 * k + 1];
          i64 v = 0;
          if (raw >= 0) {
            v = raw;
          } else {
            const i32 var = -(raw + 1);
            if (var >= 0 && var < kDeviceBoundVarCap)
              v = scope->values[var];
          }
          elem_idx += v * static_cast<i64>(elem_stride);
        }
        stack[sp++] = device_load_element(root_ptr + place_byte_off, elem_idx, node.prim_dt);
        break;
      }
      case K::kAdd: {
        i64 rhs = stack[--sp];
        i64 lhs = stack[--sp];
        stack[sp++] = lhs + rhs;
        break;
      }
      case K::kSub: {
        i64 rhs = stack[--sp];
        i64 lhs = stack[--sp];
        i64 diff = lhs - rhs;
        stack[sp++] = diff > 0 ? diff : 0;
        break;
      }
      case K::kMul: {
        i64 rhs = stack[--sp];
        i64 lhs = stack[--sp];
        stack[sp++] = lhs * rhs;
        break;
      }
      case K::kMax: {
        i64 rhs = stack[--sp];
        i64 lhs = stack[--sp];
        stack[sp++] = lhs > rhs ? lhs : rhs;
        break;
      }
      default:
        // Out-of-grammar kinds (e.g. `kMaxOverRange`) shouldn't appear in a recognizer-accepted body. Push 0 so the
        // kernel completes deterministically; the host's diagnose path re-runs the tree and raises.
        stack[sp++] = 0;
        break;
    }
  }
  return sp > 0 ? stack[sp - 1] : 0;
}

__attribute__((always_inline)) inline i64 runtime_eval_adstack_max_reduce_one_step(
    LLVMRuntime *runtime,
    const quadrants::lang::AdStackSizeExprDeviceNode *nodes,
    const i32 *indices,
    i32 body_node_count,
    const quadrants::lang::LlvmAdStackMaxReducerDeviceParams *params,
    u64 i,
    u32 num_axes,
    DeviceEvalScope *scope,
    const char *arg_buffer) {
  u64 rem = i;
  for (u32 a = num_axes; a-- > 0;) {
    const u32 len_a = params->per_axis_length[a];
    const u64 idx_a = rem % (u64)len_a;
    rem = rem / (u64)len_a;
    const i32 var_id = params->per_axis_var_id[a];
    if (var_id >= 0 && var_id < kDeviceBoundVarCap) {
      scope->values[var_id] = params->per_axis_begin[a] + (i64)idx_a;
    }
  }
  return device_eval_max_reduce_body_iterative(runtime, nodes, indices, body_node_count, scope, arg_buffer);
}
}  // namespace

#if ARCH_cuda || ARCH_amdgpu
// Forward decl: defined later in runtime.cpp; included here in single-TU layout means we forward-decl rather than
// reorder the function definitions.
void block_barrier();
extern "C" void runtime_eval_adstack_max_reduce(LLVMRuntime *runtime,
                                                RuntimeContext *ctx,
                                                Ptr params_blob,
                                                Ptr body_bytecode) {
  using quadrants::lang::AdStackSizeExprDeviceNode;
  using quadrants::lang::kAdStackMaxReducerMaxAxes;
  using quadrants::lang::LlvmAdStackMaxReducerDeviceParams;

  const auto *params = reinterpret_cast<const LlvmAdStackMaxReducerDeviceParams *>(params_blob);
  const auto *nodes = reinterpret_cast<const AdStackSizeExprDeviceNode *>(body_bytecode);
  const auto *indices = reinterpret_cast<const i32 *>(reinterpret_cast<const char *>(nodes) +
                                                      sizeof(AdStackSizeExprDeviceNode) * params->body_node_count);

  const u32 num_axes = params->num_axes;
  if (num_axes == 0 || num_axes > (u32)kAdStackMaxReducerMaxAxes) {
    return;  // Host pre-seeded the slot with INT64_MIN; nothing to reduce.
  }
  u64 total_length = 1;
  for (u32 a = 0; a < num_axes; ++a) {
    if (params->per_axis_length[a] == 0u) {
      return;  // Empty cross-product. Sentinel stays.
    }
    total_length *= (u64)params->per_axis_length[a];
  }

  const char *arg_buffer = ctx->arg_buffer;
  DeviceEvalScope scope;
  for (i32 k = 0; k < kDeviceBoundVarCap; ++k) {
    scope.values[k] = 0;
  }

  // Per-thread grid-strided walk over [0, total_length). Each thread tracks a private running max in registers;
  // the post-loop CAS reduces all per-thread maxima into the host-seeded INT64_MIN slot. Body evaluation goes
  // through `device_eval_max_reduce_body_iterative` (post-order stack walk, no recursion) - the recursive
  // `device_eval_node` triggers an NVPTX backend pathology where the recursive callee is treated as
  // undef-returning, which dead-code-eliminates the entire post-loop atomic-reduce block. The iterative walk
  // covers every body kind the recognizer accepts (`kConst / kBoundVariable / kExternalTensorRead / kAdd / kSub /
  // kMul / kMax`) on a fixed 32-slot stack.
  const u64 tid = (u64)(block_idx() * block_dim() + thread_idx());
  const u64 stride = (u64)(grid_dim() * block_dim());
  i64 running_max = (i64)0x8000000000000000ll;
  for (u64 i = tid; i < total_length; i += stride) {
    i64 v = runtime_eval_adstack_max_reduce_one_step(runtime, nodes, indices, params->body_node_count, params, i,
                                                     num_axes, &scope, arg_buffer);
    if (v > running_max) {
      running_max = v;
    }
  }
  // Atomic-max reduction across all per-thread maxima into the host-seeded INT64_MIN slot via a
  // `__atomic_compare_exchange_n` CAS loop. Both NVPTX and AMDGPU lower the builtin to a generic-addrspace `atom.cas`
  // (NVPTX) / `flat_atomic_cmpswap` (AMDGPU); the slot pointer comes from a runtime memory load
  // (`runtime->adstack_max_reducer_outputs[output_slot]`), and the unified builtin keeps the bitcode parseable on every
  // host triple, including aarch64 where PTX register-class constraints (`=l`) are rejected by clang's frontend.
  if (running_max != (i64)0x8000000000000000ll) {
    i64 *slot = &runtime->adstack_max_reducer_outputs[params->output_slot];
    i64 old_val = *slot;
    while (running_max > old_val) {
      i64 expected = old_val;
      if (__atomic_compare_exchange_n(slot, &expected, running_max, /*weak=*/false, __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
        break;
      }
      old_val = expected;
    }
  }
}
#endif

extern "C" void runtime_eval_adstack_size_expr(LLVMRuntime *runtime, RuntimeContext *ctx, Ptr bytecode) {
  // Bytecode layout:
  // [AdStackSizeExprDeviceHeader][stack_headers[n_stacks]][nodes[total_nodes]][indices[total_indices]]. All three
  // arrays live contiguously so the interpreter can index them by offset from the single `bytecode` pointer - the host
  // memcpys the whole blob in one go, and this function runs before any main-kernel dispatch that would stomp
  // `arg_buffer`.
  using quadrants::lang::AdStackSizeExprDeviceHeader;
  using quadrants::lang::AdStackSizeExprDeviceNode;
  using quadrants::lang::AdStackSizeExprDeviceStackHeader;

  const auto *header = reinterpret_cast<const AdStackSizeExprDeviceHeader *>(bytecode);
  const auto *stack_headers = reinterpret_cast<const AdStackSizeExprDeviceStackHeader *>(
      reinterpret_cast<const char *>(bytecode) + sizeof(AdStackSizeExprDeviceHeader));
  const auto *nodes = reinterpret_cast<const AdStackSizeExprDeviceNode *>(
      reinterpret_cast<const char *>(stack_headers) + sizeof(AdStackSizeExprDeviceStackHeader) * header->n_stacks);
  const auto *indices = reinterpret_cast<const i32 *>(reinterpret_cast<const char *>(nodes) +
                                                      sizeof(AdStackSizeExprDeviceNode) * header->total_nodes);

  const char *arg_buffer = ctx->arg_buffer;
  u64 *out_max_sizes = runtime->adstack_max_sizes;
  u64 *out_offsets = runtime->adstack_offsets;

  // Alignment rule copied from `publish_adstack_metadata` in `llvm_runtime_executor.cpp`: each stack's slice ends
  // aligned to 8 bytes so `stack_top_primal`'s `stack + sizeof(u64) + idx * 2 * element_size` math stays aligned
  // for every element type the IR may emit.
  auto align_up_8 = [](u64 n) -> u64 { return (n + 7u) & ~(u64)7u; };

  DeviceEvalScope scope;
  for (i32 k = 0; k < kDeviceBoundVarCap; ++k)
    scope.values[k] = 0;

  // Per-kind running offsets for the unconditional split-heap codegen path. Float allocas address via `row_id_var *
  // stride_float + float_offset_within_float_slice`; int / u1 allocas address via `linear_tid * stride_int +
  // int_offset_within_int_slice`. `out_offsets[i]` therefore must be the byte offset within the per-kind slice, not
  // within a combined slice (the codegen and the host-eval branch in `publish_adstack_metadata` both pick the per-kind
  // base + stride at the use site, so a combined offset would alias float and int slots for any kernel with mixed-kind
  // adstacks). The combined running offset is also tracked for the legacy `runtime->adstack_per_thread_stride` field
  // that offline-cache-loaded kernels predating the split read; on freshly-compiled kernels nothing dereferences it.
  u64 running_offset_combined = 0;
  u64 running_offset_float = 0;
  u64 running_offset_int = 0;
  for (u32 i = 0; i < header->n_stacks; ++i) {
    const auto &sh = stack_headers[i];
    u64 max_size;
    if (sh.root_node_idx < 0) {
      // No symbolic bound captured (offline-cache-hit with `size_exprs` dropped) - use the compile-time bound.
      max_size = sh.max_size_compile_time > 0 ? sh.max_size_compile_time : 1;
    } else {
      i64 v = device_eval_node(runtime, nodes, indices, sh.root_node_idx, &scope, arg_buffer);
      // Floor at 1 to match the host evaluator (`evaluate_adstack_size_expr`); a tree that evaluates to 0 or negative
      // leaves one slot reserved so the heap base address is still valid and any spurious push surfaces as an overflow
      // rather than a zero-slice alias. Do NOT clamp upward against `max_size_compile_time`: the compile-time seed is a
      // conservative placeholder for offline-cache fallback, NOT a proven upper bound. Clamping `v` against it would
      // silently truncate correct per-launch values and trigger overflow at the next sync; the SizeExpr evaluator is
      // the authoritative source for the per-launch capacity, and any push past `v` is the real overflow.
      if (v < 1)
        v = 1;
      max_size = static_cast<u64>(v);
    }
    out_max_sizes[i] = max_size;
    const u64 step = align_up_8(sizeof(i64) + (u64)sh.entry_size_bytes * max_size);
    if (sh.heap_kind == 0u) {
      out_offsets[i] = running_offset_float;
      running_offset_float += step;
    } else {
      out_offsets[i] = running_offset_int;
      running_offset_int += step;
    }
    running_offset_combined += step;
  }

  // Mirror the host-eval branch's contract (`llvm_runtime_executor.cpp::publish_adstack_metadata`): the legacy
  // `adstack_per_thread_stride` field publishes `stride_int_bytes` on both paths so any offline-cache-loaded kernel
  // that still reads it observes a consistent value. Earlier drafts published the combined `stride_float + stride_int`
  // here, which diverged from the host-eval branch on any kernel with at least one ExternalTensorRead-leaf SizeExpr
  // (the `use_host_eval=false` gate).
  (void)running_offset_combined;
  runtime->adstack_per_thread_stride = running_offset_int;
  runtime->adstack_per_thread_stride_float = running_offset_float;
  runtime->adstack_per_thread_stride_int = running_offset_int;
}

// Publish the device-mapped address of the pinned host slot the host allocated for the adstack overflow flag.
// Called once at materialise_runtime time after the host allocates the slot via `cuMemAllocHost_v2` / `hipHostMalloc`
// / plain malloc and obtains the device-mapped address (CUDA `cuMemHostGetDevicePointer` / HIP equivalent / identity
// on CPU). Subsequent kernel-side `stack_push` reads this pointer to write the overflow signal; the host polls the
// host-side address directly without involving any JIT helper.
extern "C" void runtime_set_adstack_overflow_flag_dev_ptr(LLVMRuntime *runtime, void *dev_ptr) {
  runtime->adstack_overflow_flag_dev_ptr = (i64 *)dev_ptr;
}

// Companion to `runtime_set_adstack_overflow_flag_dev_ptr`. Called once at materialise_runtime alongside the
// flag setter. The task-id slot lives on the same pinned host page so a single allocation backs both. Codegen
// emits a `cmpxchg(0, baked_id)` against this pointer at the lazy-claim overflow path; only the first
// overflowing thread's id sticks. Host reads the slot during the raise to look up the offending kernel /
// task in `Program::adstack_sizing_info_registry_`.
extern "C" void runtime_set_adstack_overflow_task_id_dev_ptr(LLVMRuntime *runtime, void *dev_ptr) {
  runtime->adstack_overflow_task_id_dev_ptr = (i64 *)dev_ptr;
}

// Zero-init helper called from `materialize_runtime` in runtime.cpp. `LLVMRuntime` is allocated from a raw memory pool
// rather than constructed via `new`, so the C++ default-member-initializers on the adstack fields never run. The host
// launcher writes real values into them via `publish_adstack_metadata` before dispatching any adstack-bearing kernel,
// but we still zero them here so an assert-driven read (or a stale cached kernel that runs before any publish) sees
// well-defined zeros instead of garbage.
void adstack_runtime_zero_init(LLVMRuntime *runtime) {
  runtime->adstack_heap_buffer = nullptr;
  runtime->adstack_heap_size = 0;
  runtime->adstack_per_thread_stride = 0;
  runtime->adstack_heap_buffer_float = nullptr;
  runtime->adstack_heap_size_float = 0;
  runtime->adstack_heap_buffer_int = nullptr;
  runtime->adstack_heap_size_int = 0;
  runtime->adstack_per_thread_stride_float = 0;
  runtime->adstack_per_thread_stride_int = 0;
  runtime->adstack_offsets = nullptr;
  runtime->adstack_max_sizes = nullptr;
  runtime->adstack_row_counters = nullptr;
  runtime->adstack_row_counters_capacity = 0;
  runtime->adstack_bound_row_capacities = nullptr;
  runtime->adstack_bound_row_capacities_capacity = 0;
  runtime->adstack_overflow_flag_dev_ptr = nullptr;
  runtime->adstack_overflow_task_id_dev_ptr = nullptr;
}

extern "C" {  // local stack operations

// The stack index `n` is clamped on read so that overflow (push past capacity) does not let subsequent pops and
// top-accesses underflow it and index far out of bounds. The corresponding stack_push writes through
// `runtime->adstack_overflow_flag_dev_ptr` (the device-mapped address of a pinned host slot) and skips the
// increment instead of trapping, so the host-side launcher surfaces the failure as a Python exception rather
// than killing the process via __builtin_trap. When n == 0 (pop-after-overflow underflow path) we return a
// pointer to slot 0 - an uninitialized-but-in-bounds slot. The caller will read garbage from it, but the host
// polls the pinned slot at every Quadrants Python entry and raises before any such value reaches user code.
Ptr stack_top_primal(Ptr stack, std::size_t element_size) {
  auto n = *(u64 *)stack;
  std::size_t idx = n > 0 ? n - 1 : 0;
  return stack + sizeof(u64) + idx * 2 * element_size;
}

Ptr stack_top_adjoint(Ptr stack, std::size_t element_size) {
  return stack_top_primal(stack, element_size) + element_size;
}

void stack_init(Ptr stack) {
  *(u64 *)stack = 0;
}

void stack_pop(Ptr stack) {
  auto &n = *(u64 *)stack;
  if (n > 0) {
    n--;
  }
}

void stack_push(LLVMRuntime *runtime,
                Ptr stack,
                size_t max_num_elements,
                std::size_t element_size,
                i64 task_registry_id) {
  u64 &n = *(u64 *)stack;
  if (n + 1 > max_num_elements) {
    // Overflow: the loop has more iterations than the adstack capacity. Skip the push and flip the dedicated
    // overflow flag in pinned host memory. The host polls the pinned slot at every Quadrants Python entry
    // and raises a `QuadrantsAssertionError` with a diagnosis routed through a synchronous sizer that
    // distinguishes a Quadrants bug (pre-pass undercount of the bound) from a user-side mutation that bypassed
    // tracking (DLPack zero-copy is the typical case; the sizer's freshly-computed required size will exceed
    // the cached allocated size in that case).
    //
    // Relaxed atomic ordering: multiple threads can hit this branch concurrently (CPU thread pool, GPU warp
    // divergence) and they all store the same sentinel value, so no inter-thread ordering is required. On
    // CPU this compiles to a naturally-aligned store; on CUDA/AMDGPU device kernels with the pointer aimed
    // at pinned host memory (`cuMemAllocHost_v2` / `hipHostMalloc` with the device-mapped address obtained
    // via `cuMemHostGetDevicePointer` / HIP equivalent) the store is a system-wide atomic on UVA host memory.
    // Available on Compute Capability 6.0+ / GFX9+, the same hardware envelope the existing pinned-host
    // H2D-async pattern in `llvm_adstack_lazy_claim.cpp` already requires.
    //
    // Nullptr-guard: `adstack_overflow_flag_dev_ptr` is nullptr until `materialize_runtime` initialises it.
    // A kernel running before that (a stale cached kernel, a C++-only test) silently no-ops the overflow
    // signal. The runtime cannot raise from device code; this is the safest behavior.
    i64 *flag_ptr = runtime->adstack_overflow_flag_dev_ptr;
    if (flag_ptr != nullptr) {
      __atomic_store_n(flag_ptr, (i64)1, __ATOMIC_RELAXED);
    }
    // Record task identity in the companion pinned-host slot via cmpxchg(0, registry_id) so the host raise
    // site can name the offending kernel + task in the diagnostic message. `task_registry_id == 0` means
    // "not registered" (e.g. a deserialised offline-cache task that has not yet been re-registered); skip
    // the cmpxchg so the slot stays zero and the host falls through to the generic dual-cause message.
    i64 *task_id_ptr = runtime->adstack_overflow_task_id_dev_ptr;
    if (task_id_ptr != nullptr && task_registry_id != 0) {
      i64 expected = 0;
      __atomic_compare_exchange_n(task_id_ptr, &expected, task_registry_id, /*weak=*/false, __ATOMIC_RELAXED,
                                  __ATOMIC_RELAXED);
    }
    return;
  }
  n += 1;
  std::memset(stack_top_primal(stack, element_size), 0, element_size * 2);
}

}  // extern "C" local stack operations
