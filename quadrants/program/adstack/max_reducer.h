#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/program/adstack/cache.h"
#include "quadrants/transforms/static_adstack_analysis.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;

//
// Type alias for the max-reducer result map. Keyed by `(registry_id, stack_id, mor_node_idx)` packed via the same
// `pack_max_reducer_key` encoding `AdStackCache::try_max_reducer_cache_hit` uses, so a single map shared between the
// dispatch path and the substitution helper avoids re-packing at every lookup.
using MaxReducerResultMap = std::unordered_map<uint64_t, int64_t>;
// Read-only shared ownership of a `MaxReducerResultMap`. The launch cache stores entries via this alias so the
// per-launch fast path can repoint `LlvmRuntimeExecutor::current_max_reducer_results_` to the cached map without
// copying it, and so consumers in `publish_adstack_metadata` can snapshot a stable view (`shared_ptr` copy = refcount
// bump) that survives a recursive snode-reader-kernel reentry into `dispatch_max_reducers_for_tasks`. `const` because
// cache entries are write-once at record time and read-only thereafter; the per-launch transient is rebuilt or
// repointed, never mutated in place.
using MaxReducerResultMapPtr = std::shared_ptr<const MaxReducerResultMap>;

// extract a captured `MaxOverRange`'s body subtree from `expr` and emit it as a flat `[AdStackSizeExprDeviceNode x
// body_node_count][int32 x indices_count]` bytecode blob plus a parallel `[uint8_t]` byte buffer ready to upload to a
// device storage buffer. Reachable nodes are walked in post-order from `body_node_idx` and renumbered to dense `[0,
// body_node_count)` indices; referenced indices entries from `expr.indices_table` (the `idx_raw, elem_stride` pairs
// `kExternalTensorRead` reads) are copied into the same flat buffer at `body_node_count *
// sizeof(AdStackSizeExprDeviceNode)`. Returns the raw bytes plus `body_node_count` and `indices_count` so the caller
// can populate the matching `AdStackMaxReducerParams` / `LlvmAdStackMaxReducerDeviceParams` fields. The recognizer
// grammar guarantees the body subtree contains no `kMaxOverRange` / `kFieldLoad`, so the body interpreter only needs
// the small grammar set the SPIR-V max-reducer shader and the LLVM `runtime_eval_adstack_max_reduce` runtime function
// both implement.
//
// `arg_buffer_offset_resolver` resolves `(arg_id_path) -> byte_offset_in_arg_buffer` for `kExternalTensorRead` leaves.
// On the gfx caller path this is a closure over `LaunchContextBuilder::args_type::get_element_offset` (same path the
// SizeExpr device-bytecode encoder uses). On the LLVM caller path the resolver mirrors the per-task adstack sizer's
// arg-buffer-offset precomputation. Returns `-1` on resolution failure (caller should hard-error or skip the spec).
struct EncodedMaxReducerBody {
  std::vector<uint8_t> bytes;
  uint32_t body_node_count{0};
  uint32_t indices_count{0};
  // Reads observed during encoding: one entry per body leaf (`kExternalTensorRead`) and per begin/end leaf the caller
  // resolved separately. Used by `AdStackCache::record_max_reducer_eval` so the next launch can short-circuit on a
  // generation match. Caller fills in the begin/end observations and appends body observations from this list.
  std::vector<AdStackCache::SizeExprReadObservation> body_reads;
};
// Forward decl: defined in `quadrants/program/adstack/device_bytecode.h`. Including the full header here would create a
// cycle (`device_bytecode.h` already includes this header for `MaxReducerResultMap`). The encoder only references the
// struct's `fetch` field via the pointer parameter so a forward declaration is enough at this site.
struct FieldLoadDeviceEmitter;

EncodedMaxReducerBody encode_max_reducer_body_bytecode(
    const SerializedSizeExpr &expr,
    int32_t body_node_idx,
    const std::vector<int32_t> &bound_var_ids,
    const std::function<int32_t(const std::vector<int32_t> &arg_id_path)> &arg_buffer_offset_resolver,
    LaunchContextBuilder *ctx,
    Program *prog,
    const FieldLoadDeviceEmitter *fl_emitter = nullptr);

// Snapshot the live ndarray data pointer + generation counter into each `ExternalReadObs` record. The encoder emits the
// observation skeleton (kind / arg_id_path / prim_dt) but cannot fill in the runtime-resolved `data_ptr` /
// `observed_gen` / `observed_value` because it has no `LaunchContextBuilder`. This helper closes that gap right before
// the max-reducer dispatch site calls `AdStackCache::record_max_reducer_eval`, so the next launch's
// `try_max_reducer_cache_hit` replay can fast-skip on a matching `ndarray_data_gen`. `observed_value` is recorded as
// `INT64_MIN` so the replay's gen-mismatch dereference path returns a value strictly greater than the recorded sentinel
// and forces the cache to invalidate; the cached max itself is stored in `MaxReducerCacheEntry::result`, not in any
// per-leaf observation.
void populate_max_reducer_body_observations(std::vector<AdStackCache::SizeExprReadObservation> &reads,
                                            LaunchContextBuilder *ctx,
                                            AdStackCache *cache);

// walk every per-stack `SerializedSizeExpr` in `size_exprs` post-order and return the list of `MaxOverRange` nodes the
// runtime can reduce in parallel via a dedicated max-reducer dispatch. Each returned spec references its alloca by
// `stack_id` (index into `size_exprs`) and its `MaxOverRange` by `mor_node_idx` (index into
// `size_exprs[stack_id].nodes`). Specs are returned in dependency order: deeper `MaxOverRange` nodes first so the
// runtime can substitute their results before evaluating outer nodes that depend on them. Grammar:
// * `body` subtree references only `Const`, `ExternalTensorRead(arg, [BoundVariable(this_var_id)])`, and `Add` / `Sub`
// / `Mul` / `Max` of those. Single index axis. Integer dtype on every leaf.
// * `begin` and `end` subtrees reference only `Const`, `ExternalTensorShape`, `Add` / `Sub` / `Mul` / `Max`, or another
// `MaxOverRange` already captured deeper in the same tree (becomes a `Const` after substitution). Anything outside the
// grammar is skipped silently; that `MaxOverRange` continues to fall through to the existing capped path (host
// hard-error when `QD_DEBUG_ADSTACK=1`, silent truncation otherwise).
std::vector<StaticAdStackMaxReducerSpec> recognize_adstack_max_reducer_specs(
    const std::vector<SerializedSizeExpr> &size_exprs);

// walk `expr.nodes`, replace every captured `MaxOverRange` node whose `(registry_id, stack_id, mor_node_idx)` is in
// `results` with a `Const` carrying the dispatched value. Other nodes (and their `operand_a` / `operand_b` /
// `body_node_idx` references) are copied through verbatim. The returned `SerializedSizeExpr` has `nodes.size() ==
// expr.nodes.size()` (in-place substitution); operand indices in non-substituted nodes remain valid because the count
// is unchanged.
//
// Empty-input fast path: when no captured spec matches this `(registry_id, stack_id)` (computed by checking `results`
// against every `MaxOverRange` node in `expr`), return `expr` unchanged (the caller's reference into the per-stack tree
// stays valid). Use `SerializedSizeExpr` by value as the return so the caller can transparently swap the reference
// depending on whether substitution fired.
SerializedSizeExpr substitute_precomputed_max_over_range(const SerializedSizeExpr &expr,
                                                         uint32_t registry_id,
                                                         int32_t stack_id,
                                                         const MaxReducerResultMap &results);

}  // namespace quadrants::lang
