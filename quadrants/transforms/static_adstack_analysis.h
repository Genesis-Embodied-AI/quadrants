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

  // Field source. SNode-backed fields (`qd.field(...)` placed under `qd.root.dense(...)`) are identified by the leaf
  // SNode's global id; ndarray-backed kernel arguments (`qd.ndarray(...)`) are identified by the `arg_id` path pointing
  // into the kernel arg buffer.
  enum class FieldSourceKind : int32_t { SNode = 0, NdArray = 1 };
  FieldSourceKind field_source_kind{FieldSourceKind::SNode};
  int snode_id{-1};
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

  QD_IO_DEF(cmp_op,
            field_dtype_is_float,
            field_dtype_is_double,
            literal_f32,
            literal_f64,
            literal_i32,
            polarity,
            field_source_kind,
            snode_id,
            ndarray_arg_id,
            ndarray_ndim,
            snode_root_id,
            snode_byte_base_offset,
            snode_byte_cell_stride,
            snode_iter_count);
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
  // `2 * sizeof(alloca->ret_type) * max_size` (primal + adjoint slots). Tracks the actual byte cost so
  // the sparse-heap threshold check stays accurate on f64 allocas (where `entry_size_bytes = 8` doubles the
  // per-row footprint vs. the entries-unit `per_thread_stride_float * sizeof(float)` estimate).
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
StaticAdStackAnalysisResult analyze_adstack_static_bounds(OffloadedStmt *task_ir,
                                                          const SNodeDescriptorResolver &snode_descriptor_resolver,
                                                          std::size_t sparse_heap_threshold_bytes);

}  // namespace quadrants::lang
