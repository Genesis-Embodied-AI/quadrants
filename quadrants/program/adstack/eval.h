#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/program/adstack/cache.h"

namespace quadrants::lang {

class LaunchContextBuilder;
class Program;

// Evaluates a compile-time captured `SerializedSizeExpr` against the current field state of `prog` and the
// per-launch argument values in `ctx`, returning the concrete adstack capacity for this launch. Scalar i32/i64
// field loads are serviced by `SNodeRwAccessorsBank` (one reader-kernel dispatch each); ndarray-argument shapes
// are read from `ctx->get_struct_arg<int64>`; constants and arithmetic are folded in plain C++; `MaxOverRange`
// enumerates its range and takes the max of the body expression across the bound variable. Returns -1 when the
// expression is empty (no symbolic bound captured), signalling to the caller to use the compile-time fallback.
int64_t evaluate_adstack_size_expr(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);
// Variant of `evaluate_adstack_size_expr` that bypasses `size_expr_cache_`. Used by the host-eval branch of the
// per-task sizer when feeding a stack-local substituted tree (the `size_expr_cache_` is keyed by `SerializedSizeExpr
// *`, so a transient stack address would alias unrelated cache entries across launches and return wrong cached values).
// Callers that need cache-warmed evaluation should use `evaluate_adstack_size_expr` with the original tree's stable
// pointer.
int64_t evaluate_adstack_size_expr_no_cache(const SerializedSizeExpr &expr, Program *prog, LaunchContextBuilder *ctx);

// Sub-tree variant of `evaluate_adstack_size_expr`: evaluates the subtree rooted at `node_idx` instead of the full
// tree's root. Used by the max-reducer launcher to host-resolve a captured spec's `begin` / `end` subtrees against the
// live ctx (The recognizer grammar guarantees both subtrees are closed-form, so the recursive evaluator never re-enters
// a `MaxOverRange`). Returns -1 when `node_idx` is out of range; -1 from a deeper host-eval failure propagates the same
// way as in the full-tree variant.
int64_t evaluate_adstack_size_expr_at_node(const SerializedSizeExpr &expr,
                                           int32_t node_idx,
                                           Program *prog,
                                           LaunchContextBuilder *ctx);

// Diagnose-time variant that evaluates the same `SerializedSizeExpr` against the captured
// `AdStackCache::DiagnoseLaunchSnapshot` rather than a live `LaunchContextBuilder`. Used by
// `AdStackCache::diagnose_adstack_overflow` to resolve `ExternalTensorRead` / `ExternalTensorShape` leaves at
// error time against the live (potentially mutated) ndarray contents, without needing the launch ctx that is
// gone by sync time on async backends. The cross-backend `Device::map(*allocation, &host_ptr)` path is the
// design pivot - see `AdStackCache::DiagnoseLaunchSnapshot`'s comment for the rationale (vs. re-dispatching
// the on-device sizer). Returns -1 if any leaf cannot be resolved (e.g. an arg_id missing from the snapshot,
// or an allocation whose `Device::map` fails); callers fall back to the static dual-cause body in that case.
int64_t evaluate_adstack_size_expr_for_diagnose(const SerializedSizeExpr &expr, Program *prog);

// RAII guard opening a thread-local read-cache scope. Every nested `evaluate_adstack_size_expr` running inside the
// scope shares one cache, so repeated `(snode_id, indices)` reads share a single reader-kernel dispatch. Place around
// any block that calls `evaluate_adstack_size_expr` more than once back-to-back.
class SizeExprLaunchScope {
 public:
  SizeExprLaunchScope();
  ~SizeExprLaunchScope();
  SizeExprLaunchScope(const SizeExprLaunchScope &) = delete;
  SizeExprLaunchScope &operator=(const SizeExprLaunchScope &) = delete;

 private:
  bool owns_;
};

// Internal helper exposed for cross-TU use by `quadrants/program/adstack/device_bytecode.cpp` (host-fold path of
// `encode_subtree`) and `quadrants/program/adstack/diagnose.cpp` (FieldLoad delegation in the diagnose evaluator). The
// recursive walker visits every node kind; `bound_vars` carries the live `MaxOverRange` bindings; `reads`, when
// non-null, accumulates `SizeExprReadObservation` entries for cache invalidation.
int64_t evaluate_node(const SerializedSizeExpr &expr,
                      int32_t node_idx,
                      std::unordered_map<int32_t, int64_t> &bound_vars,
                      Program *prog,
                      LaunchContextBuilder *ctx,
                      std::vector<AdStackCache::SizeExprReadObservation> *reads);

// Internal helper exposed for the diagnose evaluator and the max-reducer body encoder. Resolves a `FieldLoad` leaf via
// `SNodeRwAccessorsBank::read_int` plus the launch-scoped read cache, optionally appending a `FieldLoadObs` record to
// `reads`.
int64_t evaluate_field_load(const SerializedSizeExprNode &node,
                            std::unordered_map<int32_t, int64_t> &bound_vars,
                            Program *prog,
                            std::vector<AdStackCache::SizeExprReadObservation> *reads);

// Internal helper exposed for the max-reducer body encoder's `ExternalTensorShape` host-fold path. Reads the matching
// shape slot from the kernel arg buffer via `LaunchContextBuilder::get_struct_arg_host` and (when `reads` is non-null)
// appends an `ExternalShapeObs` observation.
int64_t evaluate_external_tensor_shape(const SerializedSizeExprNode &node,
                                       LaunchContextBuilder *ctx,
                                       std::vector<AdStackCache::SizeExprReadObservation> *reads);

// Internal helper exposed for the cache replay path (`replay_one_observation`). Reads SNode `snode_id` at `indices`
// through the launch-scoped read cache so multiple size-expr trees evaluated within the same outer launch share a
// single reader-kernel dispatch per `(snode_id, indices)` pair.
int64_t read_field_with_launch_cache(int snode_id, const std::vector<int> &indices, Program *prog);

}  // namespace quadrants::lang
