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
#include "quadrants/ir/visitors.h"
#include "quadrants/codegen/ir_dump.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/program.h"
#include "quadrants/program/per_construct_cache.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quadrants::lang {
namespace {

// Feature-local statement gather that ALSO offers container statements (loops / if / while / offloaded) to the
// predicate, not just leaf statements. The shared `irpass::analysis::gather_statements` never tests a container
// itself: `BasicStmtVisitor` claims those types with typed overloads that recurse into the body without consulting
// the predicate. The split needs the container statements too -- to detect container KINDS (`MeshForStmt`, dynamic
// -parallel loops) and to walk a construct's whole subtree including nested loops -- so it does that traversal here
// rather than widening the shared analysis API for this experimental pass (AGENTS.md: minimize contact area).
class ContainerAwareGatherer : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit ContainerAwareGatherer(std::function<bool(Stmt *)> test) : test_(std::move(test)) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (test_(stmt))
      results_.push_back(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (test_(stmt))
      results_.push_back(stmt);
  }

  std::vector<Stmt *> results_;

 private:
  std::function<bool(Stmt *)> test_;
};

// Like `irpass::analysis::gather_statements` but the predicate also sees container statements. See
// `ContainerAwareGatherer`.
std::vector<Stmt *> gather_stmts_incl_containers(IRNode *root, const std::function<bool(Stmt *)> &test) {
  ContainerAwareGatherer gatherer(test);
  root->accept(&gatherer);
  return gatherer.results_;
}

// Two-phase operand-remapping cloner for a SUBSET of a block's top-level statements. Kept local to this experimental
// pass (AGENTS.md: minimize contact area) rather than widening the shared cloner in analysis/clone.cpp, since only the
// split needs subset cloning. Mirrors that file's `IRCloner`: phase 1 records original->clone for every statement
// (including nested ones, by walking container bodies in lockstep), phase 2 rewrites each clone's operands to the clone
// when the referent is in the subset and leaves it pointing at the original otherwise -- so a caller that needs a
// self-contained block must pass a subset closed under operands.
class SubsetCloner : public IRVisitor {
 public:
  enum Phase { register_operand_map, replace_operand } phase;

  explicit SubsetCloner(IRNode *other) : phase(register_operand_map), other_(other) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void set_other(IRNode *n) {
    other_ = n;
  }

  void generic_visit(Stmt *stmt) {
    auto *other_stmt = other_->as<Stmt>();
    if (phase == register_operand_map) {
      operand_map_[stmt] = other_stmt;
      // The hand-written container clones (IfStmt / RangeForStmt / StructForStmt / WhileStmt) rebuild via a typed
      // constructor, not the copy ctor, so they drop two fields the whole-kernel path keeps; restore both here for
      // every cloned statement (a no-op re-copy for leaf stmts, whose QD_DEFINE_CLONE copy-ctor clone already has
      // them). region_tag marks the graph region (`qd.graph.do_while`) a serial statement belongs to; losing it makes
      // the serial offloader tag the task at level -1 so a conditional/loop nested in a host loop runs once instead of
      // once per host-loop iteration. dbg_info carries the source traceback; losing it empties `get_tb()` so later
      // per-construct diagnostics (e.g. offload.cpp's block-dim-too-large warning on a cloned StructForStmt) print
      // without a source location.
      other_stmt->region_tag = stmt->region_tag;
      other_stmt->dbg_info = stmt->dbg_info;
    } else {
      QD_ASSERT(stmt->num_operands() == other_stmt->num_operands());
      for (int i = 0; i < stmt->num_operands(); i++) {
        auto it = operand_map_.find(stmt->operand(i));
        other_stmt->set_operand(i, it == operand_map_.end() ? stmt->operand(i) : it->second);
      }
    }
  }

  void visit(Stmt *stmt) override {
    generic_visit(stmt);
  }

  void visit(IfStmt *stmt) override {
    generic_visit(stmt);
    auto *other = other_->as<IfStmt>();
    if (stmt->true_statements) {
      other_ = other->true_statements.get();
      stmt->true_statements->accept(this);
      other_ = other;
    }
    if (stmt->false_statements) {
      other_ = other->false_statements.get();
      stmt->false_statements->accept(this);
      other_ = other;
    }
  }

  void visit(WhileStmt *stmt) override {
    generic_visit(stmt);
    auto *other = other_->as<WhileStmt>();
    other_ = other->body.get();
    stmt->body->accept(this);
    other_ = other;
  }

  void visit(RangeForStmt *stmt) override {
    generic_visit(stmt);
    auto *other = other_->as<RangeForStmt>();
    other_ = other->body.get();
    stmt->body->accept(this);
    other_ = other;
  }

  void visit(StructForStmt *stmt) override {
    generic_visit(stmt);
    auto *other = other_->as<StructForStmt>();
    other_ = other->body.get();
    stmt->body->accept(this);
    other_ = other;
  }

  void visit(Block *block) override {
    auto *other = other_->as<Block>();
    for (int i = 0; i < (int)block->size(); i++) {
      other_ = other->statements[i].get();
      block->statements[i]->accept(this);
    }
    other_ = other;
  }

 private:
  IRNode *other_;
  std::unordered_map<Stmt *, Stmt *> operand_map_;
};

// Clone only the listed top-level statements (in the given order) into a fresh Block. The per-construct frontend split
// needs one isolated copy per construct, and cloning the entire block each time is O(constructs x block size) -- ~5.9 s
// on a 130-construct / 2685-statement kernel; the slice a construct actually needs is tiny by comparison. Offload has
// not run at the split seam, so the top-level statements are leaves or serial/for/if/while containers (mesh-for kernels
// fall back before this point), which `SubsetCloner` handles.
std::unique_ptr<Block> clone_block_subset(Block *block, const std::vector<int> &indices) {
  auto nb = std::make_unique<Block>();
  nb->set_parent_callable(block->parent_callable());
  std::vector<Stmt *> srcs;
  srcs.reserve(indices.size());
  for (int i : indices) {
    Stmt *s = block->statements[i].get();
    srcs.push_back(s);
    nb->insert(s->clone());  // deep-clones nested blocks; operands still point at the originals until remapped below
  }
  // Same two-phase walk as analysis/clone.cpp's IRCloner, driven per statement pair because the source and target
  // blocks no longer line up index-for-index.
  SubsetCloner cloner(nb.get());
  for (int p = 0; p < 2; p++) {
    cloner.phase = (p == 0) ? SubsetCloner::register_operand_map : SubsetCloner::replace_operand;
    for (std::size_t j = 0; j < srcs.size(); j++) {
      cloner.set_other(nb->statements[j].get());
      srcs[j]->accept(&cloner);
    }
  }
  return nb;
}

// Run the whole pre-offload + offload frontend on ONE isolated top-level construct instead of once over the whole
// kernel. Mirrors the kNone / non-mesh portion of the whole-kernel sequence in `compile_to_offloads` between simplify_I
// and simplify_III, in the same order, so a per-construct compile produces the same tasks the whole-kernel path would
// for that construct. `cb` is a construct already isolated + recomputed (other constructs dropped, shared defs DIE'd)
// by `split_frontend_per_construct`.
void run_construct_frontend(IRNode *cb,
                            const CompileConfig &config,
                            const Kernel *kernel,
                            bool verbose,
                            int construct_index) {
  const std::string &name = kernel->get_name();
  // print_ir / QD_DUMP_IR are observation-only under the split: instead of forcing the whole-kernel path, we let the
  // split run and make its console prints / IR dumps reflect what was actually compiled -- one set per construct.
  // print_ir already flows through `verbose` into each pass printer below; this banner just labels whose IR follows.
  if (verbose)
    std::cout << "[per-construct frontend split] " << name << " construct " << construct_index << std::endl;
  // QD_DUMP_IR: dump this construct's post-simplify_I IR to <kernel>_construct<i>_after_simplify_I.ll -- the
  // intermediate stage the whole-kernel path can no longer show once the split runs the simplify stages per construct
  // (the construct index keeps constructs from colliding on one filename). We deliberately do NOT emit a per-construct
  // after_offload dump: compile_to_offloads.cpp still writes the whole-kernel reassembled <kernel>_after_offload.ll
  // (every construct's offloaded tasks), so a per-construct copy would only duplicate it -- and would double-count in
  // consumers that glob "*after_offload*" (e.g. test_loop_config_name, test_stable_gtmp_offsets). No-op unless
  // QD_DUMP_IR=1.
  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  const bool dump_ir = dump_ir_env != nullptr && std::string(dump_ir_env) == "1";
  auto dump_stage = [&](const std::string &stage) {
    if (!dump_ir)
      return;
    std::filesystem::path dir = config.debug_dump_path;
    std::filesystem::create_directories(dir);
    std::filesystem::path filename =
        dir / (name + "_construct" + std::to_string(construct_index) + "_" + stage + ".ll");
    std::string ir_str;
    irpass::print(cb, &ir_str);
    std::ofstream ofs(filename.string());
    if (ofs.good())
      ofs << ir_str;
  };
  // Mirror the whole-kernel path's `verify_if_debug` after each stage (compile_to_offloads.cpp). It is a no-op unless
  // config.debug, so it is free in release builds; under debug=True it keeps the per-construct path catching malformed
  // IR at the responsible pass instead of letting it slip through to codegen.
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_I"});
  irpass::analysis::verify_if_debug(cb, config);
  dump_stage("after_simplify_I");
  irpass::handle_external_ptr_boundary(cb, config);
  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(cb, config, {name});
    irpass::analysis::verify_if_debug(cb, config);
  }
  irpass::merge_global_ptrs(cb);
  irpass::analysis::verify_if_debug(cb, config);
  irpass::flag_access(cb);
  irpass::analysis::verify_if_debug(cb, config);
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_II"});
  irpass::analysis::verify_if_debug(cb, config);
  irpass::offload(cb, config);
  irpass::analysis::verify_if_debug(cb, config);
  if (config.opt_level > 0) {
    irpass::cse_offloaded_tasks(cb);
  }
  irpass::flag_access(cb);
  irpass::full_simplify(cb, config, {false, /*autodiff_enabled*/ false, name, verbose, "simplify_III"});
  irpass::analysis::verify_if_debug(cb, config);
}

bool block_has_mesh_for(Block *block) {
  if (block == nullptr)
    return false;
  // Gathering containers is not optional here: `MeshForStmt` is a container, so a leaf-only gather would match
  // nothing and silently pass every mesh kernel into the split, which produces IR that segfaults at launch.
  return !gather_stmts_incl_containers(block, [](Stmt *s) { return s->is<MeshForStmt>(); }).empty();
}

// Whether the kernel contains a concurrently-executed region: a `qd.stream_parallel()` block or a
// `qd.graph.parallel()` section, both of which tag their loops with a nonzero `stream_parallel_group_id` /
// `graph_parallel_region_id`. Those regions run on separate streams while sharing one kernel context and its single
// global-temporary buffer. The per-construct split runs `offload` INDEPENDENTLY per construct, and each run restarts
// global-temporary offset allocation from zero, so two concurrent constructs (e.g. dynamic-bound loops whose bounds
// flow through a global temp) would alias the same offset and race on the shared buffer -- a loop could consume the
// other section's bound and skip iterations or read out of bounds. The whole-kernel offload instead assigns unique
// offsets across all tasks, so fall back for any kernel with a concurrent region. Gathers containers so loops nested
// inside the region's block are inspected too.
bool block_has_concurrent_region(Block *block) {
  if (block == nullptr)
    return false;
  return !gather_stmts_incl_containers(block, [](Stmt *s) {
            if (auto *rf = s->cast<RangeForStmt>())
              return rf->stream_parallel_group_id != 0 || rf->graph_parallel_region_id != 0;
            if (auto *sf = s->cast<StructForStmt>())
              return sf->stream_parallel_group_id != 0 || sf->graph_parallel_region_id != 0;
            return false;
          }).empty();
}

// Resolve a local pointer to its base AllocaStmt: an AllocaStmt directly, a MatrixPtrStmt into one (matrix/vector
// element), or a GetElementStmt into one (`qd.Struct` member). Returns nullptr when the pointer is not alloca-based
// (e.g. a global pointer / global temporary), in which case it is not a local variable. Chasing GetElementStmt matters
// for the cross-construct analyses: a struct member is stored via `LocalStoreStmt(GetElementStmt(alloca), ...)`, so
// without following `src` the member's writer (and any producing loop) is dropped from a consumer's slice and it reads
// the stale value.
Stmt *resolve_local_alloca(Stmt *ptr) {
  while (ptr != nullptr) {
    if (ptr->is<AllocaStmt>())
      return ptr;
    if (auto *mp = ptr->cast<MatrixPtrStmt>()) {
      ptr = mp->origin;
      continue;
    }
    if (auto *ge = ptr->cast<GetElementStmt>()) {
      ptr = ge->src;
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

// A load from global memory (field / ndarray). These are side-effect-free, so the backward slice may recompute them
// into a later construct -- subject to the shadowing check in `split_is_recompute_safe`.
bool stmt_is_global_read(Stmt *s) {
  return s->is<GlobalLoadStmt>();
}

// A volatile global load (`qd.volatile_load`). Volatile semantics require the access to happen exactly once, in place,
// on every execution -- so, unlike a plain load, it must NEVER be recomputed into another construct: cloning it turns
// one source-level observation of a concurrently-updated cell into several and reorders it relative to surrounding
// work.
bool stmt_is_volatile_load(Stmt *s) {
  auto *ld = s->cast<GlobalLoadStmt>();
  return ld != nullptr && ld->is_volatile;
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

// The locals (allocas) a statement writes. Reads the shared `Store` trait (`get_store_destination`) so it covers EVERY
// way the frontend mutates a function-scope variable without enumerating statement types: a `LocalStoreStmt`, an
// `AtomicOpStmt`, a `qd.append` (`SNodeOpStmt::allocate`, which writes the returned index into its `val` alloca WITHOUT
// a `LocalStoreStmt`), and a `@qd.real_func` call that mutates a local through a `qd.ref` argument (`FuncCallStmt`).
// Non-local destinations (global pointers, ndarrays, snode roots) resolve to nullptr and are dropped. Missing any of
// these would let a consuming construct's backward slice keep a bare zero-initialized alloca (reading the stale value)
// while the effectful-producer gate never sees the writer to force a fallback.
std::vector<Stmt *> stmt_local_write_allocas(Stmt *s) {
  std::vector<Stmt *> out;
  for (Stmt *dest : irpass::analysis::get_store_destination(s))
    if (Stmt *a = resolve_local_alloca(dest))
      out.push_back(a);
  return out;
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
    for (Stmt *a : stmt_local_write_allocas(s))
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
    // Gather containers too: a nested loop or branch has operands of its own (bounds, conditions) that the slice has
    // to follow, so it has to enter the worklist like any other statement.
    for (Stmt *sub : gather_stmts_incl_containers(s, [](Stmt *) { return true; }))
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
// fine: clone+die duplicates it into each construct. Three things are NOT recomputable and force a fall back to the
// whole-kernel path:
//
//   (1) Loop-carried locals. A local produced INSIDE a top-level loop and consumed by another construct cannot be
//       reconstructed, because the producing loop is dropped from the consumer's slice. It is safe only when every
//       access (read or loop-write) of that local stays within a SINGLE construct. Checking readers against the
//       *union* of writer constructs is unsound: when two constructs both read-modify-write the same local, each is
//       "covered" by the union, yet the second still consumes the value the first produced.
//
//   (2) Non-recomputable producers cloned into a consumer. The backward slice pulls the writers (and operand chains)
//       of any local a construct reads. Cloning such a producer into a consuming construct is only sound if it is a
//       pure, deterministic function of recomputed operands. Three kinds are not: an EFFECTFUL statement (e.g. a local
//       capturing the return of a global `AtomicOpStmt`, or a sparse op) already emits its own task in its home
//       segment, so cloning it runs the effect again (`old = atomic_add(counter, 1); for i: out[i] = old` increments
//       `counter` twice); a NON-DETERMINISTIC statement (`RandStmt`) resamples and advances the PRNG once per clone
//       (`r = random(); for i: x[i] = r; for i: y[i] = r` gives the two loops different values); and a VOLATILE load
//       (`qd.volatile_load`) must be observed exactly once in place, so cloning it into several constructs turns one
//       read of a concurrently-updated cell into several. Any statement the slice would recompute from a DIFFERENT
//       segment must therefore be none of effectful, a `RandStmt`, or a volatile load.
//
//   (3) Field loads snapshotted before a later construct. The backward slice recomputes a serial def -- including a
//       field/ndarray LOAD -- into every construct that consumes it. That is sound only if no construct BETWEEN the
//       original load and the consuming construct performs a real effect (any `stmt_is_task_effect`: a global/ndarray
//       store, a global atomic, or a sparse activate/deactivate); otherwise the recomputed load, now executing inside
//       the later construct, observes the mutation instead of the source-order snapshot (e.g.
//       `base = x[0]; for ..: x[0] = 2; for i: y[i] = base` would read 2 instead of the original x[0]). Neither field
//       identity nor effect target is tracked, so this is conservative -- an unrelated store/sparse op also triggers
//       the fallback -- which keeps correctness while costing only the rare shadowed-snapshot kernel a split.
bool split_is_recompute_safe(Block *block) {
  if (block == nullptr)
    return false;

  // (1) Loop-carried locals. Writers are found generically via the shared `Store` trait (`stmt_local_write_allocas`),
  // so any statement that mutates a local counts -- `LocalStoreStmt`, `AtomicOpStmt`, a `qd.append`
  // (`SNodeOpStmt::allocate`), a `@qd.real_func` writing through a `qd.ref` arg (`FuncCallStmt`), and an external /
  // bitcode call (`ExternalFuncCallStmt`) -- without enumerating statement types. A read-modify-write / call with
  // in-out args (`AtomicOpStmt`, `FuncCallStmt`, `ExternalFuncCallStmt`) also READS the local it writes.
  auto accesses = irpass::analysis::gather_statements(
      block, [](Stmt *s) { return s->is<LocalLoadStmt>() || !stmt_local_write_allocas(s).empty(); });
  std::unordered_map<Stmt *, std::set<Stmt *>> read_owners;        // alloca -> top-level constructs that read it
  std::unordered_map<Stmt *, std::set<Stmt *>> loop_write_owners;  // alloca -> constructs that write it inside a loop
  for (Stmt *acc : accesses) {
    bool inside_container = false;
    Stmt *owner = top_level_owner(acc, block, &inside_container);
    if (owner == nullptr)
      continue;
    auto writes = stmt_local_write_allocas(acc);
    for (Stmt *a : writes)
      if (inside_container)
        loop_write_owners[a].insert(owner);
    if (auto *ld = acc->cast<LocalLoadStmt>()) {
      if (Stmt *a = resolve_local_alloca(ld->src))
        read_owners[a].insert(owner);
    } else if (acc->is<AtomicOpStmt>() || acc->is<FuncCallStmt>() || acc->is<ExternalFuncCallStmt>()) {
      for (Stmt *a : writes)  // atomic rmw / call in-out arg reads the same local it writes
        read_owners[a].insert(owner);
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

  // (2) + (3): both need each construct's backward slice and the top-level positions of global writes.
  const int n = (int)block->statements.size();
  std::unordered_map<Stmt *, int> top_index;
  for (int j = 0; j < n; j++)
    top_index[block->statements[j].get()] = j;
  // Any real effect (`stmt_is_task_effect`) counts as an intervening mutation of global state: global/ndarray stores,
  // global atomics, AND sparse-structure ops (`SNodeOpStmt` activate/deactivate inherit has_global_side_effect==true).
  std::vector<bool> writes_global(n, false);
  for (Stmt *w : gather_stmts_incl_containers(block, [](Stmt *s) { return stmt_is_task_effect(s); })) {
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
    // (2) Every statement this construct recomputes from ANOTHER segment must be purely recomputable. An effectful
    // statement (atomic, global store, sparse op, ...) already emits its own task in its home segment, so cloning it
    // here would run the effect twice; a RandStmt would draw a fresh sample and advance the PRNG per clone.
    int min_recomputed_read_pos = -1;
    for (Stmt *s : needed) {
      bool inside = false;
      Stmt *owner = top_level_owner(s, block, &inside);
      if (owner == nullptr)
        continue;
      auto it = top_index.find(owner);
      if (it == top_index.end())
        continue;
      int pos = it->second;
      if (segs.seg_id[pos] != k && (stmt_is_task_effect(s) || s->is<RandStmt>() || stmt_is_volatile_load(s)))
        return false;  // recomputing another segment's effect, PRNG draw, or volatile load would change behavior
      // (3) Earliest top-level position of a global read this construct RECOMPUTES (one that lives in an earlier
      // segment; reads in the construct's own segment are not moved). If a global write sits strictly between that read
      // and the construct, the recomputed read would observe the mutation -> not safe.
      if (stmt_is_global_read(s) && pos < b_lo && (min_recomputed_read_pos < 0 || pos < min_recomputed_read_pos))
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

  // Program-scoped record of this split's stats (total / hit / recompiled per kernel), read back by the compilation
  // manager (`KernelCompilationManager::load_or_compile`) into the observability surface -- reading it there rather
  // than in a backend codegen driver keeps the counts backend-agnostic (LLVM and SPIR-V alike). This PR ships no reuse
  // tier, so `hit` stays 0; the cross-process manifest PR turns this into an actual per-construct cache.
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
    auto cloned = clone_block_subset(block, keep_indices);
    auto *cb = cloned.get();
    irpass::die(cb);  // clean up anything left dead after slicing (recompute per construct)
    irpass::re_id(cb);

    // Run the full per-construct frontend on the isolated construct and take its produced tasks. This PR ships the
    // split with NO reuse tier, so every construct is recompiled here; the cross-process manifest PR keys this output
    // (by a stable per-construct cache key it introduces) and reuses an unchanged construct's tasks instead.
    run_construct_frontend(cb, config, kernel, verbose, /*construct_index=*/n_constructs - 1);
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
  auto env_is_enabled = [](std::string_view env) {
    const char *v = std::getenv(env.data());
    return v != nullptr && std::string(v) == "1";
  };
  // QD_DUMP_CFG still forces the whole-kernel path. Unlike the other dump flags it is NOT observation-only under the
  // split: cfg_optimization (a pre-existing pass, not touched by this PR) both changes its own scope under QD_DUMP_CFG
  // (dumping the whole-kernel CFG instead of the per-task CFGs a normal build uses) and names its dump files by phase,
  // not construct -- so running the split under it would collide every construct's CFG into one filename. Making
  // QD_DUMP_CFG observation-only means reworking cfg_optimization, which is deferred to a separate PR; until then keep
  // the whole-kernel fallback for it so its dump matches the whole graph.
  //
  // QD_DUMP_IR, QD_DUMP_SIMPLIFY and print_ir, by contrast, are observation-only: the split still runs and their output
  // is emitted PER CONSTRUCT, reflecting what the split actually compiled rather than perturbing it. QD_DUMP_IR ->
  // run_construct_frontend writes <kernel>_construct<i>_*.ll; QD_DUMP_SIMPLIFY -> simplify.cpp keys its filenames off a
  // global call counter, so each construct's passes already land in distinct files; print_ir -> each construct's pass
  // printer prints under a per-construct banner. See docs/source/user_guide/optimization_passes.md.
  if (env_is_enabled(DUMP_CFG_ENV))
    return false;
  // QD_KERNEL_COVERAGE (python/quadrants/lang/_kernel_coverage.py) rewrites every kernel to store a probe into a global
  // coverage field before each source line. Those probe stores add top-level global-write constructs (so the partition
  // no longer matches the source structure) and, in graph/checkpoint kernels, land between a yield gate and its loop --
  // per-construct reassembly then moves them and corrupts both the coverage signal and yield/resume behavior. Coverage
  // is a measurement mode, so keep the split transparent by falling back to the whole-kernel path.
  if (env_is_enabled("QD_KERNEL_COVERAGE"))
    return false;
  if (block_has_mesh_for(block))
    return false;
  if (block_has_concurrent_region(block))
    return false;  // concurrent constructs share one global-temp buffer; per-construct offload would alias offsets
  if (!split_is_recompute_safe(block))
    return false;
  split_frontend_per_construct(ir, config, kernel, verbose);
  return true;
}

}  // namespace irpass

}  // namespace quadrants::lang
