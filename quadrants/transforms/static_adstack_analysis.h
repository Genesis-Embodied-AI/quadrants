// Static-IR-bound sparse-adstack-heap analysis. Walks an OffloadedStmt's body and produces three pieces of metadata the
// SPIR-V and LLVM codegens both consume to size the per-task float adstack heap to the count of threads that actually
// reach a push site (rather than the dispatched-threads worst case):
//
// 1. The Lowest Common Ancestor (LCA) block of every f32-typed `AdStackPushStmt` / `AdStackLoadTopStmt` /
//    `AdStackLoadTopAdjStmt` in the task body. The codegen emits a one-shot atomic row-claim at this block; threads
//    that never reach the LCA never claim a heap row and never touch the float heap. Push/load-top contributions are
//    folded together because both paths reach the heap (push writes, load-top reads), but pop sites are NOT folded
//    -pops only mutate `count_var` and impose no dominance requirement.
//
// 2. The set of autodiff-bootstrap const-init pushes - the `push(stack, ConstStmt)` shape the autodiff transform
//    emits at the offload body root (immediately following the matching `AdStackAllocaStmt`) so the matching reverse
//    pop has a value to consume on every thread regardless of any later gating. Folding these into the LCA would drag
//    the LCA up to the offload body root and revert the per-thread (worst-case) sizing - they belong to every thread,
//    while the gated pushes do not. The codegen treats the bootstrap pushes specially: still bumps `count_var` so push
//    and pop stay balanced, but skips the slot store (the bootstrap value is dead memory because no `load_top` ever
//    reads it back; writing through a possibly-unclaimed `row_id_var` would corrupt arbitrary heap rows).
//
// 3. An optional `StaticAdStackBoundExpr` capturing a single recognized gate predicate `BinaryOp(cmp,
//    GlobalLoadStmt(field[I]), ConstStmt(literal))` on the chain from the float LCA up to the task body root.
//    Recognizes both ndarray-backed (`ExternalPtrStmt -> ArgLoadStmt`) and SNode-backed (`GetChStmt -> output_snode`
//    leaf with `root -> dense -> place(scalar)` shape) field sources. Also handles the autodiff-spilled gate shape
//    `IfStmt(cond = AdStackLoadTopStmt(stack=S))` by walking back to the unique non-const push onto S in the same task.
//    Multi-gate chains, compound-predicate trees, and unfamiliar control-flow parents fall through to "no capture" so
//    the runtime falls back to dispatched-threads worst-case sizing.
//
// The IR pre-pass also produces the per-thread strides (`per_thread_stride_float`, `per_thread_stride_int`) and a stack
// count, all of which the codegens need for downstream metadata buffer layout.
//
// SNode descriptor resolution is parameterized via the `SNodeDescriptorResolver` callback so the analysis stays
// decoupled from any specific compiled SNode struct representation. The SPIR-V/Metal/Vulkan path resolves through
// `CompiledSNodeStructs::snode_descriptors`; the LLVM path uses its own runtime SNode tree. Resolvers that return
// `std::nullopt` cause the SNode-backed gate to be rejected, so only fields whose descriptors are known to the caller
// end up captured.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_set>
#include <vector>

#include "quadrants/common/serialization.h"
#include "quadrants/ir/adstack_size_expr.h"
#include "quadrants/ir/ir.h"

namespace quadrants::lang {

class AdStackPushStmt;
class Block;
class OffloadedStmt;

// Captured static gate predicate. Encoding mirrors what the runtime reducer kernel expects: one comparison op against a
// typed literal, one field load on the same SNode path or ndarray slot for every iteration, plus a polarity bit
// selecting the LCA's enter-on-true vs enter-on-false orientation.
struct StaticAdStackBoundExpr {
  // BinaryOpType (cmp_lt / cmp_le / cmp_gt / cmp_ge / cmp_eq / cmp_ne) cast to int. Stored as int rather than the enum
  // to keep the header dependency-light; the codegen and the runtime reducer both cast through `BinaryOpType` at use
  // site.
  int cmp_op{0};

  // Literal threshold. The active variant is selected by the GlobalLoad result's primitive type the IR pass observed;
  // the reducer kernel bitcasts / reads the right one based on `field_dtype` at dispatch time. f64 gates store the
  // literal in `literal_f64` so the reducer can read the source ndarray as `double*` without narrowing precision.
  bool field_dtype_is_float{true};
  bool field_dtype_is_double{false};
  float literal_f32{0.0f};
  double literal_f64{0.0};
  int32_t literal_i32{0};

  // True when the LCA enters on the gate condition holding (typical `if cmp:` shape); false when the LCA sits inside
  // the `else` branch (`if cmp: else: <gate>`). The reducer flips the predicate at dispatch time so the captured count
  // always matches the count of threads that reach the LCA.
  bool polarity{true};

  // Field source. SNode-backed fields (`qd.field(...)` placed under `qd.root.dense(...)`) are identified at dispatch
  // time by the descriptor triple below (`snode_root_id` + byte base / cell stride + iter count); ndarray-backed
  // kernel arguments (`qd.ndarray(...)`) are identified by the `arg_id` path pointing into the kernel arg buffer.
  enum class FieldSourceKind : int32_t { SNode = 0, NdArray = 1 };
  FieldSourceKind field_source_kind{FieldSourceKind::SNode};
  std::vector<int> ndarray_arg_id;
  // Number of axes on the captured gating ndarray (1 for `qd.ndarray(qd.f32, shape=(N,))`, 2 for `shape=(R, C)`, ...).
  // Set at capture time from `ExternalPtrStmt::indices.size()` so the host launcher can walk the right number of
  // `SHAPE_POS_IN_NDARRAY + axis` slots when computing the reducer's flat-element walk bound. Zero for SNode-backed
  // gates (where `snode_iter_count` carries the equivalent information).
  int ndarray_ndim{0};

  // SNode-source extras populated by the resolver callback when the field is SNode-backed. Combined byte offset (dense
  // within root cell + leaf within dense's per-cell layout) and the per-`gid` stride the reducer kernel walks the field
  // at. `snode_root_id` selects which root buffer to bind on the dispatch when a kernel has multiple roots. Set to -1 /
  // 0 for ndarray-backed gates and for SNode gates whose descriptors the resolver does not know (the IR analysis treats
  // those as "no capture").
  int snode_root_id{-1};
  uint32_t snode_byte_base_offset{0};
  uint32_t snode_byte_cell_stride{0};
  uint32_t snode_iter_count{0};

  // Compile-time loop trip count of the task whose pushes this gate dominates. Set when the task's
  // `range_for` has a constant `[begin, end)` (the same `static_bound` predicate the iteration-count check
  // uses). Zero means runtime-bound (analyzer cannot resolve the trip count from IR alone). The runtime
  // uses this as an upper bound when sizing the float adstack heap from the reducer's gate-passing-cell
  // count: each loop iteration claims at most one row at the LCA-block, so the heap needs at most
  // `loop_iter_static` rows regardless of how many cells of an oversized SNode the reducer counted.
  uint32_t loop_iter_static{0};

  // Pre-chunking loop trip-count `SizeExpr` for the same task, captured by `determine_ad_stack_size` and
  // copied here at codegen time. Covers the runtime-bounded shapes the static `loop_iter_static` cannot:
  // `for j in range(field[i])`, `for k in range(arr.shape[axis])`, two-arg ranges over either of those,
  // multi-axis ndrange products, and casts of a stashed loop index. Empty when the analyzer found no
  // bound shape (the user kernel is outside the SizeExpr grammar) or the task is not a range-for; in
  // those cases the runtime falls back to the unclipped reducer count. Evaluated at every kernel launch
  // (cheap - same `evaluate_adstack_size_expr` walk the per-thread `ad_stack_size` already runs) so the
  // clip tracks the live trip count even when it varies between launches.
  SerializedSizeExpr loop_iter_size_expr;

  QD_IO_DEF(cmp_op,
            field_dtype_is_float,
            field_dtype_is_double,
            literal_f32,
            literal_f64,
            literal_i32,
            polarity,
            field_source_kind,
            ndarray_arg_id,
            ndarray_ndim,
            snode_root_id,
            snode_byte_base_offset,
            snode_byte_cell_stride,
            snode_iter_count,
            loop_iter_static,
            loop_iter_size_expr);
};

// Captured `MaxOverRange` reducible by a dedicated parallel max-reducer dispatch at launch time. The recognized grammar
// `MaxOverRange(begin, end, body)` where `begin` and `end` evaluate to closed-form scalars after recursive substitution
// of any deeper captured `MaxOverRange`s, and `body` is integer-typed arithmetic (`Const`, `ExternalTensorRead(arg,
// [BoundVar(this_var)])`, `Add` / `Sub` / `Mul` / `Max` of those). The runtime dispatches one reducer per spec in
// dependency order (deepest first); the per-launch result is substituted as a `Const` into the SizeExpr tree so the
// per-thread sizer never walks the iteration domain. Anything outside the grammar is left for the existing capped path
// (silent truncation today; tracked as future work).
struct StaticAdStackMaxReducerSpec {
  // Index of the alloca within `AdStackSizingAttribs::allocas` (same indexing the per-thread sizer uses).
  int32_t stack_id{-1};
  // Index of the OUTERMOST `MaxOverRange` node in this alloca's `size_expr.nodes`. The runtime keys results by
  // `(task_id_in_kernel, stack_id, mor_node_idx)` and the substitution helper replaces `nodes[mor_node_idx]` with a
  // `Const` carrying the dispatched reducer's output. When a chain of nested `MaxOverRange`s is captured as a single
  // multi-axis spec, this is the outermost node (axis 0); the inner nodes collapse into the per-axis arrays below and
  // are not separately substituted.
  int32_t mor_node_idx{-1};
  // Body subtree root (the innermost `MaxOverRange`'s body for multi-axis specs). Walked at launch time to extract the
  // arg-id paths the reducer reads from. The body may reference any of the `axis_var_ids` below as bound variables; the
  // encoder remaps each to a dense device-scope slot in `[0, axis_var_ids.size())`.
  int32_t body_node_idx{-1};
  // Per-axis iteration ranges and bound-variable ids, ORDERED outermost-first (axis 0 = the spec's outermost
  // `MaxOverRange`, axis N-1 = the innermost). The dispatch iterates the cross-product of these ranges; each `[begin,
  // end)` must evaluate closed-form at dispatch time (after recursive substitution of any deeper captured
  // `MaxOverRange` ancestors). Single-axis specs have one entry per vector.
  std::vector<int32_t> axis_begin_node_idxs;
  std::vector<int32_t> axis_end_node_idxs;
  std::vector<int32_t> axis_var_ids;
  // Indices into `size_expr.nodes` that are deeper captured `MaxOverRange` specs this one depends on. The runtime
  // dispatches in topological order so all dependencies have been substituted before this spec's body is read.
  std::vector<int32_t> dependent_mor_node_idxs;
  QD_IO_DEF(stack_id,
            mor_node_idx,
            body_node_idx,
            axis_begin_node_idxs,
            axis_end_node_idxs,
            axis_var_ids,
            dependent_mor_node_idxs);
};

// SNode descriptor info the analysis needs to capture an SNode-backed gate. The resolver returns `std::nullopt` when
// the leaf / dense pair has no compile-time descriptor available (e.g. on backends that walk the SNode tree at
// runtime), in which case the analysis rejects the gate and the runtime falls back to worst-case sizing.
struct SNodeFieldDescriptor {
  int root_id{-1};
  uint32_t byte_base_offset{0};
  uint32_t byte_cell_stride{0};
  uint32_t iter_count{0};
};
using SNodeDescriptorResolver =
    std::function<std::optional<SNodeFieldDescriptor>(const SNode *leaf, const SNode *dense)>;

struct StaticAdStackAnalysisResult {
  // LCA of every f32 push/load-top site, or `nullptr` when the task has no f32 adstack push sites or the LCA reduces to
  // the task body's root. In the latter case the row-claim still runs from the root and the layout collapses to the
  // per-thread (worst-case) eager mapping, but emitting the claim is harmless.
  Block *lca_block_float{nullptr};
  // Set of autodiff-bootstrap const-init pushes identified by the pre-pass. Codegens skip the slot store at these
  // sites; only the `count_var` increment is kept so push and pop stay balanced.
  std::unordered_set<AdStackPushStmt *> bootstrap_pushes;
  // Captured static gate, when the analysis recognized exactly one IfStmt on the LCA -> root chain. `nullopt` falls
  // through to dispatched-threads worst-case sizing in the runtime.
  std::optional<StaticAdStackBoundExpr> bound_expr;
  // Per-thread strides in elements of each heap's element type, summed across every alloca in the task. The float
  // stride counts both primal and adjoint slots (`2 * max_size`); the int stride counts primal only (i32 / u1 adstacks
  // have no adjoint). Both are zero when the task declares no adstacks.
  uint32_t per_thread_stride_float{0};
  uint32_t per_thread_stride_int{0};
  // Per-thread float-heap byte stride, summed across every f32 / f64 alloca in the task as
  // `2 * sizeof(alloca->ret_type) * max_size` (primal + adjoint slots). Tracks the actual byte cost so the sparse-heap
  // threshold check stays accurate on f64 allocas (where `entry_size_bytes = 8` doubles the per-row footprint vs. the
  // entries-unit `per_thread_stride_float * sizeof(float)` estimate).
  uint64_t per_thread_stride_float_bytes{0};
  // Total adstack count, useful for sizing per-task metadata buffers downstream.
  int num_ad_stacks{0};
};

// Run the analysis on `task_ir`. `snode_descriptor_resolver` is consulted only on SNode-backed gates; pass an
// always-empty resolver to disable SNode capture (the analysis still captures ndarray-backed gates and emits the LCA +
// bootstrap set for both backends). `sparse_heap_threshold_bytes` is the conservative-heap cutoff below which a
// matched gate is NOT captured into `bound_expr`, so the codegen falls back to the eager `linear_thread_idx * stride`
// addressing and the launchers skip the per-launch reducer dispatch + DtoH; see `CompileConfig::
// ad_stack_sparse_threshold_bytes` for the user-facing knob (default 100 MiB; 0 forces capture, useful for tests
// that pin the reducer-backed sizing path).
// `task_range_is_original_loop`: true when `task_ir->{begin_value, end_value}` is the original user-loop trip
// count, false when an upstream pass (e.g. `make_cpu_multithreaded_range_for` on CPU LLVM) has rewritten it
// into a per-chunk subrange. The analyzer only fills `bound_expr.loop_iter_static` when this is true; on
// rewritten loops the original trip count is no longer recoverable from `task_ir` and the runtime falls back to
// the unclipped reducer count for that task.
StaticAdStackAnalysisResult analyze_adstack_static_bounds(OffloadedStmt *task_ir,
                                                          const SNodeDescriptorResolver &snode_descriptor_resolver,
                                                          std::size_t sparse_heap_threshold_bytes,
                                                          bool task_range_is_original_loop = true);

}  // namespace quadrants::lang
