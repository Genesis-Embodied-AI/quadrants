// Per-construct FRONTEND split (no reuse tier).
//
// Runs the remaining pre-offload + offload frontend (simplify / merge_global_ptrs / offload) PER top-level construct
// and reassembles, instead of once over the whole kernel, so a later cross-process cache can key and reuse each
// construct's frontend output independently. Each construct is isolated by its BACKWARD SLICE and the pass falls back
// to the whole-kernel path for anything not recompute-safe.
//
// This lives in its own file (only `maybe_split_frontend_per_construct` is called from `compile_to_offloads`) to keep
// the central pass's contact area with this experimental feature small -- see AGENTS.md ("Minimize contact area").

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/program.h"
#include "quadrants/program/per_construct_cache.h"

#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quadrants::lang {
namespace {

// Run the whole pre-offload + offload frontend on ONE isolated top-level construct instead of once over the whole
// kernel. Mirrors the kNone / non-mesh portion of the whole-kernel sequence in `compile_to_offloads` between simplify_I
// and simplify_III, in the same order, so a per-construct compile produces the same tasks the whole-kernel path would
// for that construct. `cb` is a construct already isolated + recomputed (other constructs dropped, shared defs DIE'd)
// by `split_frontend_per_construct`.
void run_construct_frontend(IRNode *cb, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  const std::string &name = kernel->get_name();
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_I"});
  irpass::handle_external_ptr_boundary(cb, config);
  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(cb, config, {name});
  }
  irpass::merge_global_ptrs(cb);
  irpass::flag_access(cb);
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_II"});
  irpass::offload(cb, config);
  if (config.opt_level > 0) {
    irpass::cse_offloaded_tasks(cb);
  }
  irpass::flag_access(cb);
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_III"});
}

bool block_has_mesh_for(Block *block) {
  if (block == nullptr)
    return false;
  // `include_containers` is not optional here: `MeshForStmt` is a container, so without it this guard matches
  // nothing and silently passes every mesh kernel into the split, which produces IR that segfaults at launch.
  return !irpass::analysis::gather_statements(
              block, [](Stmt *s) { return s->is<MeshForStmt>(); }, /*include_containers=*/true)
              .empty();
}

// Resolve a local pointer (an AllocaStmt, or a MatrixPtrStmt into one) to its base AllocaStmt. Returns nullptr when the
// pointer is not alloca-based (e.g. a global pointer / global temporary), in which case it is not a local variable.
Stmt *resolve_local_alloca(Stmt *ptr) {
  while (ptr != nullptr) {
    if (ptr->is<AllocaStmt>())
      return ptr;
    if (auto *mp = ptr->cast<MatrixPtrStmt>()) {
      ptr = mp->origin;
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

// Find the top-level statement (direct child of `top`) that transitively contains `s`, and report via
// `*inside_container` whether the path from `s` up to `top` crosses any container statement (loop/while). Returns
// nullptr when `s` is not under `top`.
Stmt *top_level_owner(Stmt *s, Block *top, bool *inside_container) {
  *inside_container = false;
  Stmt *cur = s;
  Block *b = s->parent;
  while (b != nullptr && b != top) {
    Stmt *owner = b->parent_stmt();
    if (owner == nullptr)
      return nullptr;
    if (owner->is_container_statement())
      *inside_container = true;
    cur = owner;
    b = owner->parent;
  }
  return (b == top) ? cur : nullptr;
}

// Whether a top-level statement is a *real* observable effect that forces its segment to become a task. Pointer/address
// computations (GlobalPtr/ExternalPtr/MatrixPtr) report has_global_side_effect()==true because of sparse activation,
// but they are not standalone effects -- they are the address half of a load/store and get recomputed into whichever
// construct consumes them (e.g. a dynamic loop bound `range(lo[None], hi[None])` lowers to GlobalPtr+GlobalLoad in a
// serial segment that must NOT become its own task; it is recomputed into the loop construct via the backward slice).
bool stmt_is_task_effect(Stmt *s) {
  if (s->is<GlobalPtrStmt>() || s->is<ExternalPtrStmt>() || s->is<MatrixPtrStmt>())
    return false;
  return s->has_global_side_effect();
}

// A load from global memory (field / ndarray). These are what the backward slice may recompute into a later construct.
bool stmt_is_global_read(Stmt *s) {
  return s->is<GlobalLoadStmt>();
}

// A store to global memory (field / ndarray). Atomics count only when their destination is a global address; an atomic
// on a local alloca is not a global write.
bool stmt_is_global_write(Stmt *s) {
  if (s->is<GlobalStoreStmt>())
    return true;
  if (auto *at = s->cast<AtomicOpStmt>())
    return resolve_local_alloca(at->dest) == nullptr;
  return false;
}

// Segmentation of the top-level block the way `offload` will chunk it into tasks: every container statement
// (RangeFor/StructFor/While/...) is its own segment, and every maximal run of consecutive non-container (serial)
// statements is one serial segment. A construct = a segment that emits a task: containers always do; a serial run does
// iff it contains a real effect (`stmt_is_task_effect`). A serial run of only pure value defs / pointer chains (e.g.
// dynamic loop bounds) emits no task and is recomputed into whichever construct consumes it.
struct TopLevelSegments {
  std::vector<int> seg_id;           // per top-level statement index -> segment id
  std::vector<bool> seg_emits_task;  // per segment -> is it a construct (emits a task)
  int n_segs = 0;
};

TopLevelSegments segment_top_level(Block *block) {
  const int n = (int)block->statements.size();
  TopLevelSegments segs;
  segs.seg_id.assign(n, -1);
  int cur_seg = -1;
  bool in_serial_run = false;
  for (int j = 0; j < n; j++) {
    Stmt *s = block->statements[j].get();
    if (s->is_container_statement()) {
      cur_seg = (int)segs.seg_emits_task.size();
      segs.seg_emits_task.push_back(true);
      in_serial_run = false;
    } else {
      if (!in_serial_run) {
        cur_seg = (int)segs.seg_emits_task.size();
        segs.seg_emits_task.push_back(false);
        in_serial_run = true;
      }
      if (stmt_is_task_effect(s))
        segs.seg_emits_task[cur_seg] = true;
    }
    segs.seg_id[j] = cur_seg;
  }
  segs.n_segs = (int)segs.seg_emits_task.size();
  return segs;
}

// Top-level writers of each local variable, for the backward slice. A `LocalLoadStmt`'s only operand is the alloca (or
// a `MatrixPtrStmt` into it), never the store that gave it its value, so an operand-closed slice pulls in a bare,
// zero-initialized alloca and silently reads zeros. `split_is_recompute_safe` has already rejected kernels where a
// local is produced *inside* another construct's loop, so every writer that matters here is top-level and safe to
// recompute into the consuming construct.
std::unordered_map<Stmt *, std::vector<Stmt *>> gather_top_level_alloca_writers(Block *block) {
  std::unordered_map<Stmt *, std::vector<Stmt *>> alloca_writers;
  const int n = (int)block->statements.size();
  for (int j = 0; j < n; j++) {
    Stmt *s = block->statements[j].get();
    Stmt *dest = nullptr;
    if (auto *st = s->cast<LocalStoreStmt>())
      dest = st->dest;
    else if (auto *at = s->cast<AtomicOpStmt>())
      dest = at->dest;
    if (dest == nullptr)
      continue;
    if (Stmt *a = resolve_local_alloca(dest))
      alloca_writers[a].push_back(s);
  }
  return alloca_writers;
}

// Backward slice of construct `k`: segment k's whole subtree plus the transitive operand-def chain it reads
// (const/arg/binop/pointer/field-load chains from earlier segments, recomputed into this construct), including the
// top-level writers of any local it loads. Computed on the ORIGINAL block; returns the set of original statements the
// construct needs (no cloning -- callers clone the surviving statements themselves).
std::unordered_set<Stmt *> compute_construct_needed(
    Block *block,
    const std::vector<int> &seg_id,
    int k,
    const std::unordered_map<Stmt *, std::vector<Stmt *>> &alloca_writers) {
  const int n = (int)block->statements.size();
  std::unordered_set<Stmt *> needed;
  std::vector<Stmt *> worklist;
  auto add_with_subtree = [&needed, &worklist](Stmt *s) {
    if (needed.insert(s).second)
      worklist.push_back(s);
    // `include_containers`: a nested loop or branch has operands of its own (bounds, conditions) that the slice has to
    // follow, so it has to enter the worklist like any other statement.
    for (Stmt *sub : irpass::analysis::gather_statements(
             s, [](Stmt *) { return true; }, /*include_containers=*/true))
      if (needed.insert(sub).second)
        worklist.push_back(sub);
  };
  for (int j = 0; j < n; j++) {
    if (seg_id[j] != k)
      continue;
    add_with_subtree(block->statements[j].get());
  }
  while (!worklist.empty()) {
    Stmt *s = worklist.back();
    worklist.pop_back();
    for (Stmt *op : s->get_operands())
      if (op != nullptr && needed.insert(op).second)
        worklist.push_back(op);
    // Reaching an alloca means this construct reads the local; bring its top-level writers (and their own operand
    // chains) along, or the construct compiles against an uninitialized variable.
    if (s->is<AllocaStmt>()) {
      auto it = alloca_writers.find(s);
      if (it != alloca_writers.end()) {
        for (Stmt *w : it->second)
          add_with_subtree(w);
      }
    }
  }
  return needed;
}

// Split-safety gate. Cross-construct SSA that is purely recomputable -- consts, args, top-level pure arithmetic -- is
// fine: clone+die duplicates it into each construct. Two things are NOT recomputable and force a fall back to the
// whole-kernel path:
//
//   (1) Loop-carried locals. A local produced INSIDE a top-level loop and consumed by another construct cannot be
//       reconstructed, because the producing loop is dropped from the consumer's slice. It is safe only when every
//       access (read or loop-write) of that local stays within a SINGLE construct. Checking readers against the
//       *union* of writer constructs is unsound: when two constructs both read-modify-write the same local, each is
//       "covered" by the union, yet the second still consumes the value the first produced.
//
//   (2) Field loads snapshotted before a later construct. The backward slice recomputes a serial def -- including a
//       field/ndarray LOAD -- into every construct that consumes it. That is sound only if no construct BETWEEN the
//       original load and the consuming construct writes global memory; otherwise the recomputed load, now executing
//       inside the later construct, observes the mutation instead of the source-order snapshot (e.g.
//       `base = x[0]; for ..: x[0] = 2; for i: y[i] = base` would read 2 instead of the original x[0]). Field identity
//       is not tracked, so this is conservative -- a write to an unrelated field also triggers the fallback -- which
//       keeps correctness while costing only the rare shadowed-snapshot kernel a split.
bool split_is_recompute_safe(Block *block) {
  if (block == nullptr)
    return false;

  // (1) Loop-carried locals.
  auto accesses = irpass::analysis::gather_statements(block, [](Stmt *s) {
    return s->is<LocalLoadStmt>() || s->is<LocalStoreStmt>() || s->is<AtomicOpStmt>();
  });
  std::unordered_map<Stmt *, std::set<Stmt *>> read_owners;        // alloca -> top-level constructs that read it
  std::unordered_map<Stmt *, std::set<Stmt *>> loop_write_owners;  // alloca -> constructs that write it inside a loop
  for (Stmt *acc : accesses) {
    Stmt *read_ptr = nullptr;
    Stmt *write_ptr = nullptr;
    if (auto *ld = acc->cast<LocalLoadStmt>()) {
      read_ptr = ld->src;
    } else if (auto *st = acc->cast<LocalStoreStmt>()) {
      write_ptr = st->dest;
    } else if (auto *at = acc->cast<AtomicOpStmt>()) {
      read_ptr = at->dest;  // atomic read-modify-write counts as both a read and a write of dest
      write_ptr = at->dest;
    }
    bool inside_container = false;
    Stmt *owner = top_level_owner(acc, block, &inside_container);
    if (owner == nullptr)
      continue;
    if (Stmt *a = resolve_local_alloca(read_ptr))
      read_owners[a].insert(owner);
    if (Stmt *a = resolve_local_alloca(write_ptr)) {
      if (inside_container)
        loop_write_owners[a].insert(owner);
    }
  }
  for (auto &kv : loop_write_owners) {
    std::set<Stmt *> owners = kv.second;  // constructs that loop-write this local ...
    auto rit = read_owners.find(kv.first);
    if (rit != read_owners.end())
      owners.insert(rit->second.begin(), rit->second.end());  // ... plus every construct that reads it
    if (owners.size() > 1)
      return false;  // a loop-produced local is shared across constructs -> not recomputable
  }

  // (2) Field loads snapshotted before a later construct.
  const int n = (int)block->statements.size();
  std::unordered_map<Stmt *, int> top_index;
  for (int j = 0; j < n; j++)
    top_index[block->statements[j].get()] = j;
  std::vector<bool> writes_global(n, false);
  for (Stmt *w : irpass::analysis::gather_statements(
           block, [](Stmt *s) { return stmt_is_global_write(s); }, /*include_containers=*/true)) {
    bool inside = false;
    Stmt *owner = top_level_owner(w, block, &inside);
    if (owner == nullptr)
      continue;
    auto it = top_index.find(owner);
    if (it != top_index.end())
      writes_global[it->second] = true;
  }
  auto segs = segment_top_level(block);
  auto alloca_writers = gather_top_level_alloca_writers(block);
  for (int k = 0; k < segs.n_segs; k++) {
    if (!segs.seg_emits_task[k])
      continue;
    int b_lo = -1;
    for (int j = 0; j < n; j++)
      if (segs.seg_id[j] == k) {
        b_lo = j;
        break;
      }
    if (b_lo < 0)
      continue;
    auto needed = compute_construct_needed(block, segs.seg_id, k, alloca_writers);
    // Earliest top-level position of a global read this construct RECOMPUTES (i.e. one that lives in an earlier
    // segment; reads in the construct's own segment are not moved). If any global write sits strictly between that read
    // and the construct, the recomputed read would observe the mutation -> not safe.
    int min_recomputed_read_pos = -1;
    for (Stmt *s : needed) {
      if (!stmt_is_global_read(s))
        continue;
      bool inside = false;
      Stmt *owner = top_level_owner(s, block, &inside);
      if (owner == nullptr)
        continue;
      auto it = top_index.find(owner);
      if (it == top_index.end())
        continue;
      int pos = it->second;
      if (pos < b_lo && (min_recomputed_read_pos < 0 || pos < min_recomputed_read_pos))
        min_recomputed_read_pos = pos;
    }
    if (min_recomputed_read_pos < 0)
      continue;
    for (int q = min_recomputed_read_pos + 1; q < b_lo; q++)
      if (writes_global[q])
        return false;  // an intervening construct mutates global memory the recomputed load would re-read
  }
  return true;
}

// Split the flat top-level block into constructs right after lower_ast + the structural prefix, run the full
// per-construct frontend on each (recomputing shared top-level defs into every construct that reads them), and
// reassemble the produced OffloadedStmts in source order. Compiling each construct independently keeps its
// simplify/merge_global_ptrs working set tiny, and makes each construct's frontend output independently keyable --
// which the cross-process cache PR uses to skip unchanged constructs. This PR runs the split with NO reuse tier (every
// construct is recompiled every compile). Correctness for recompute-safe kernels rests on two things: cross-construct
// global-temp hubs dissolve via recompute, and cross-construct memory ordering is preserved by keeping tasks in
// original construct order. The caller (`maybe_split_frontend_per_construct`) has already restricted this to
// autodiff_mode==kNone / non-mesh / recompute-safe kernels.
void split_frontend_per_construct(IRNode *ir, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  auto *block = ir->cast<Block>();
  QD_ASSERT(block != nullptr);
  const int n = (int)block->statements.size();
  auto segs = segment_top_level(block);
  auto alloca_writers = gather_top_level_alloca_writers(block);

  // Program-scoped record of this split's stats (total / hit / recompiled per kernel), read back by the codegen driver
  // into the observability surface. This PR ships no reuse tier, so `hit` stays 0; the cross-process manifest PR turns
  // this into an actual per-construct cache.
  PerConstructCache *cc = (kernel->program != nullptr) ? &kernel->program->per_construct_cache() : nullptr;

  std::vector<std::unique_ptr<Stmt>> tasks;
  int n_constructs = 0, n_hit = 0, n_recompiled = 0;
  for (int k = 0; k < segs.n_segs; k++) {
    if (!segs.seg_emits_task[k])
      continue;  // pure-def-only serial run: no standalone task, recomputed into consuming constructs below
    n_constructs++;
    // Isolate construct k by its BACKWARD SLICE and keep only the surviving top-level statements. The slice is computed
    // on the ORIGINAL block; cloning the whole block first and deleting afterwards is O(constructs x block size) so we
    // clone only the construct that survives.
    auto needed = compute_construct_needed(block, segs.seg_id, k, alloca_writers);
    std::vector<int> keep_indices;
    for (int j = 0; j < n; j++) {
      if (needed.find(block->statements[j].get()) != needed.end())
        keep_indices.push_back(j);
    }
    auto cloned = irpass::analysis::clone_block_subset(block, keep_indices);
    auto *cb = cloned.get();
    irpass::die(cb);  // clean up anything left dead after slicing (recompute per construct)
    irpass::re_id(cb);

    // Run the full per-construct frontend on the isolated construct and take its produced tasks. This PR ships the
    // split with NO reuse tier, so every construct is recompiled here; the cross-process manifest PR keys this output
    // (via `get_hashed_per_construct_cache_key`) and reuses an unchanged construct's tasks instead.
    run_construct_frontend(cb, config, kernel, verbose);
    n_recompiled++;
    while (!cb->statements.empty())
      tasks.push_back(cb->extract(0));
  }
  while (!block->statements.empty())
    block->extract(0);
  for (auto &t : tasks)
    block->insert(std::move(t));
  if (cc != nullptr) {
    std::lock_guard<std::mutex> g(cc->mu);
    cc->last_stats[kernel->get_name()] = {n_constructs, n_hit, n_recompiled};
  }
}

}  // namespace

namespace irpass {

bool maybe_split_frontend_per_construct(IRNode *ir,
                                        const CompileConfig &config,
                                        const Kernel *kernel,
                                        bool verbose,
                                        AutodiffMode autodiff_mode) {
  if (autodiff_mode != AutodiffMode::kNone)
    return false;
  auto *block = ir->cast<Block>();
  if (block == nullptr)
    return false;
  if (block_has_mesh_for(block))
    return false;
  if (!split_is_recompute_safe(block))
    return false;
  split_frontend_per_construct(ir, config, kernel, verbose);
  return true;
}

}  // namespace irpass

}  // namespace quadrants::lang
