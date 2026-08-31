// Per-construct FRONTEND split.
//
// Runs the pre-offload + offload frontend (simplify / merge_global_ptrs / offload) per top-level construct and
// reassembles, instead of once over the whole kernel, so each construct's compilation can be cached independently.
// Each construct is isolated by its BACKWARD SLICE, and the pass falls back to the whole-kernel path for anything not
// recompute-safe. It lives in its own file (only `maybe_split_frontend_per_construct` is called from
// `compile_to_offloads`) to keep the central pass's contact area small -- see AGENTS.md ("Minimize contact area").

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

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace quadrants::lang {
namespace {

// Statement gather whose predicate also sees CONTAINER statements. The shared `gather_statements` only tests leaves:
// its typed container overloads recurse into the body without consulting the predicate.
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

std::vector<Stmt *> gather_stmts_incl_containers(IRNode *root, const std::function<bool(Stmt *)> &test) {
  ContainerAwareGatherer gatherer(test);
  root->accept(&gatherer);
  return gatherer.results_;
}

// Two-phase operand-remapping clone of a SUBSET of a block's top-level statements. The subset must be closed under
// operands: an operand outside it is left pointing at the original rather than a clone.
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
      // Typed container clones (If/RangeFor/StructFor/While) rebuild via a constructor that drops these two fields;
      // restore them. region_tag sets the serial offloader's task scheduling (checkpoint/do_while level); dbg_info the
      // source traceback.
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

// Clone the listed top-level statements (in order) into a fresh Block. Cloning the whole block per construct would be
// O(constructs x block size); a construct's slice is tiny by comparison.
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
  // Driven per statement pair: after subsetting, source and target blocks no longer line up index-for-index.
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

// Run the pre-offload + offload frontend on ONE isolated construct, mirroring the kNone/non-mesh whole-kernel sequence
// in `compile_to_offloads` (simplify_I .. simplify_III, same order) so the construct yields the same tasks.
void run_construct_frontend(IRNode *cb,
                            const CompileConfig &config,
                            const Kernel *kernel,
                            bool verbose,
                            int construct_index) {
  const std::string &name = kernel->get_name();
  // print_ir / QD_DUMP_IR are observation-only: the split still runs, and its prints/dumps come out per construct.
  if (verbose)
    std::cout << "[per-construct frontend split] " << name << " construct " << construct_index << std::endl;
  // Dump only the post-simplify_I stage per construct -- the intermediate the whole-kernel path can no longer show
  // once simplify runs per construct. No per-construct after_offload dump: the reassembled whole-kernel
  // <kernel>_after_offload.ll already covers it, and a copy would double-count in consumers that glob after_offload.
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
  // `verify_if_debug` after each stage mirrors the whole-kernel path; a no-op unless config.debug.
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
  // Must gather containers: `MeshForStmt` is a container, so a leaf-only gather would pass every mesh kernel through.
  return !gather_stmts_incl_containers(block, [](Stmt *s) { return s->is<MeshForStmt>(); }).empty();
}

// A concurrently-executed region: a `qd.stream_parallel()` / `qd.graph.parallel()` loop (nonzero
// stream_parallel_group_id / graph_parallel_region_id). Such regions run on separate streams sharing one
// global-temporary buffer, but the split runs `offload` per construct and each run restarts global-temp offsets from
// zero, so two concurrent constructs would alias the same offset and race. Fall back for these.
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

// Resolve a local pointer to its base AllocaStmt, chasing MatrixPtrStmt (matrix/vector element) and GetElementStmt
// (`qd.Struct` member) origins. Returns nullptr for non-local pointers (global ptr / global temp). Following
// GetElementStmt matters: a struct member store is `LocalStoreStmt(GetElementStmt(alloca), ...)`, so otherwise the
// member's writer is dropped from a consumer's slice.
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

// A real observable effect that forces its segment to become a task. Pointer/address stmts (GlobalPtr/ExternalPtr/
// MatrixPtr) report has_global_side_effect()==true but are only the address half of a load/store, so exclude them --
// they get recomputed into whichever construct consumes them.
bool stmt_is_task_effect(Stmt *s) {
  if (s->is<GlobalPtrStmt>() || s->is<ExternalPtrStmt>() || s->is<MatrixPtrStmt>())
    return false;
  return s->has_global_side_effect();
}

bool stmt_is_global_read(Stmt *s) {
  return s->is<GlobalLoadStmt>();
}

// A volatile load must be observed exactly once in place, so unlike a plain load it must never be recomputed into
// another construct.
bool stmt_is_volatile_load(Stmt *s) {
  auto *ld = s->cast<GlobalLoadStmt>();
  return ld != nullptr && ld->is_volatile;
}

// Segment the top-level block the way `offload` chunks it into tasks: each container is its own segment; each maximal
// run of serial statements is one segment. A construct is a segment that emits a task -- containers always do, a serial
// run only if it contains a real effect. A pure serial run emits no task and is recomputed into its consumers.
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

// Sentinel from `construct_gdw_level`: leave the construct's per-task levels as offload assigned them.
constexpr int kKeepOffloadGdwLevel = INT_MIN;

// The graph_do_while level every one of construct k's reassembled tasks must carry. The host graph driver rebuilds a
// `qd.graph.do_while` loop from a flat task list, grouping a contiguous run of same-level tasks as one loop body; a
// task at the wrong level wedged into that run strands the loop counter outside the body so it never decrements.
// Offloading a construct in isolation can misplace a serial task whose `region_tag` is unset (level -1) -- e.g. a
// pass-made global-temp materialization -- because it no longer has a correctly-tagged neighbour to borrow from. We
// recover the intended level from k's own source segment, still frontend-tagged at the split seam: a container's
// `graph_do_while_level_id`, else its side-effecting serial statements' `region_tag`. Returns kKeepOffloadGdwLevel
// (no re-stamp) when a serial segment's side effects straddle >1 level (offload already split those per level) and for
// non-graph kernels (all level -1, as offload assigned).
int construct_gdw_level(Block *block, const std::vector<int> &seg_id, int k) {
  const int n = (int)block->statements.size();
  int level = kKeepOffloadGdwLevel;
  for (int j = 0; j < n; j++) {
    if (seg_id[j] != k)
      continue;
    Stmt *s = block->statements[j].get();
    // A container is its own segment, so its own level tags the whole construct (matching the whole-kernel offloader).
    if (auto *rf = s->cast<RangeForStmt>())
      return rf->graph_do_while_level_id;
    if (auto *sf = s->cast<StructForStmt>())
      return sf->graph_do_while_level_id;
    if (auto *mf = s->cast<MeshForStmt>())
      return mf->graph_do_while_level_id;
    if (stmt_is_task_effect(s) && s->region_tag.is_set) {
      const int lv = s->region_tag.graph_do_while_level_id;
      if (level == kKeepOffloadGdwLevel)
        level = lv;
      else if (level != lv)
        return kKeepOffloadGdwLevel;  // serial run straddles levels: trust offload's per-region task split
    }
  }
  return level;
}

// The locals (allocas) a statement writes, via the shared `Store` trait so it covers every mutation path without
// enumerating statement types (LocalStore, AtomicOp, `qd.append`, a `qd.ref` arg mutated by a `@qd.real_func`).
// Non-local destinations resolve to nullptr and are dropped.
std::vector<Stmt *> stmt_local_write_allocas(Stmt *s) {
  std::vector<Stmt *> out;
  for (Stmt *dest : irpass::analysis::get_store_destination(s))
    if (Stmt *a = resolve_local_alloca(dest))
      out.push_back(a);
  return out;
}

// The locals (allocas) a statement reads, via the shared `Load` trait. Covers LocalLoad, AtomicOp (read-modify-write),
// and crucially `ReferenceStmt` -- a read-only `qd.ref` arg into a `@qd.real_func` is a Load with no store
// destination, so a store-only scan would miss it and the loop-carried-local gate could wrongly accept the split.
std::vector<Stmt *> stmt_local_read_allocas(Stmt *s) {
  std::vector<Stmt *> out;
  for (Stmt *ptr : irpass::analysis::get_load_pointers(s))
    if (Stmt *a = resolve_local_alloca(ptr))
      out.push_back(a);
  return out;
}

// Top-level writers of each local, for the backward slice: a `LocalLoadStmt`'s operand is the alloca, not the store
// that gave it its value, so an operand-closed slice would otherwise read a bare zero-initialized alloca. Each writer
// is paired with its top-level source index so the slice keeps only writers preceding the consumer.
std::unordered_map<Stmt *, std::vector<std::pair<int, Stmt *>>> gather_top_level_alloca_writers(Block *block) {
  std::unordered_map<Stmt *, std::vector<std::pair<int, Stmt *>>> alloca_writers;
  const int n = (int)block->statements.size();
  for (int j = 0; j < n; j++) {
    Stmt *s = block->statements[j].get();
    for (Stmt *a : stmt_local_write_allocas(s))
      alloca_writers[a].push_back({j, s});
  }
  return alloca_writers;
}

// Backward slice of construct `k`: k's subtree, the transitive operand chain it reads, and (for any local it loads)
// that local's top-level writers from EARLIER segments. Computed on the original block; returns the needed original
// statements, which callers clone.
std::unordered_set<Stmt *> compute_construct_needed(
    Block *block,
    const std::vector<int> &seg_id,
    int k,
    const std::unordered_map<Stmt *, std::vector<std::pair<int, Stmt *>>> &alloca_writers) {
  const int n = (int)block->statements.size();
  std::unordered_set<Stmt *> needed;
  std::vector<Stmt *> worklist;
  auto add_with_subtree = [&needed, &worklist](Stmt *s) {
    if (needed.insert(s).second)
      worklist.push_back(s);
    // Containers too: a nested loop/branch has its own operands (bounds, conditions) the slice must follow.
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
    // This construct reads a local: pull its writers from EARLIER segments only (a writer in k's own or a later
    // segment must not be cloned in; own-segment writers are already in `needed`).
    if (s->is<AllocaStmt>()) {
      auto it = alloca_writers.find(s);
      if (it != alloca_writers.end()) {
        for (const auto &[w_pos, w] : it->second) {
          if (seg_id[w_pos] >= k)
            continue;
          add_with_subtree(w);
        }
      }
    }
  }
  return needed;
}

// Split-safety gate. Purely recomputable cross-construct SSA (consts, args, top-level pure arithmetic) is fine --
// clone+die duplicates it into each construct. Three things are not, and force a fall back to the whole-kernel path:
//
//   (1) Loop-carried locals: a local produced inside a top-level loop and consumed by another construct cannot be
//       reconstructed once the producing loop is dropped from the consumer's slice. Safe only when every access of the
//       local stays within a SINGLE construct -- checking readers against the union of writer constructs is unsound
//       (two constructs that each read-modify-write it are both "covered", yet the second consumes the first's value).
//
//   (2) Non-recomputable producers the slice would clone into a consumer: an EFFECTFUL statement (its task already
//       runs in its home segment, so a clone runs the effect twice), a NON-DETERMINISTIC `RandStmt` (resamples /
//       advances the PRNG per clone), or a VOLATILE load (must be observed exactly once in place).
//
//   (3) A field/ndarray load recomputed into a later construct past an intervening ALIASING write: if a construct
//       between the load and the consumer writes memory the load may re-read, the recomputed load observes the mutation
//       instead of the source-order snapshot. Aliasing is by SNode / ndarray (the shared alias analysis), so an
//       unrelated field's store does NOT force fallback; a write we cannot pin to one address (structural / external)
//       does.
Stmt *write_effect_dest(Stmt *w) {
  if (auto *st = w->cast<GlobalStoreStmt>())
    return st->dest;
  if (auto *at = w->cast<AtomicOpStmt>())
    return at->dest;
  // Structural snode ops, list/gc, and external / real-func calls have no single target pointer -> may-alias-all.
  return nullptr;
}

// InternalFuncStmt is a runtime intrinsic. Most -- warp/block sync + memory fences, shuffle / ballot / broadcast /
// elect, thread / invocation index, clocks, register composite-extract -- write NO field/ndarray/structural memory, so
// they cannot change a recomputed load and are not a recompute hazard (condition 3), even though they report a side
// effect (to stay un-reordered). The few internal ops that DO write global memory (sparse-matrix insert_triplet,
// refresh_counter, the test_* allocator probes) are deliberately absent, so they stay may-alias-all. This is an
// allowlist: any name not listed (a new/unknown internal op) is treated as a writer, costing at most a missed split.
bool internal_func_is_memory_free(const std::string &name) {
  static const std::unordered_set<std::string> kMemoryFree = {
      "composite_extract_0", "composite_extract_1",  "composite_extract_2",  "composite_extract_3",
      "linear_thread_idx",   "block_thread_idx",      "do_nothing",           "workgroupBarrier",
      "workgroupMemoryBarrier", "gridMemoryBarrier",  "localInvocationId",    "globalInvocationId",
      "vkGlobalThreadIdx",   "subgroupBarrier",       "subgroupMemoryBarrier", "subgroupElect",
      "subgroupBroadcast",   "subgroupShuffle",       "subgroupShuffleDown",  "subgroupShuffleUp",
      "subgroupBallotU32",   "subgroupBallotU64",     "subgroupInvocationId", "spirv_clock_i64",
      "cuda_clock_i64",      "block_barrier",         "block_barrier_and_i32", "block_barrier_or_i32",
      "block_barrier_count_i32", "block_mem_fence",   "grid_mem_fence",       "warp_barrier",
      "cuda_all_sync_i32",   "cuda_any_sync_i32",     "cuda_uni_sync_i32",    "cuda_ballot_i32",
      "cuda_shfl_sync_i32",  "cuda_shfl_sync_f32",    "cuda_shfl_up_sync_i32", "cuda_shfl_up_sync_f32",
      "cuda_shfl_down_sync_i32", "cuda_shfl_down_sync_f32", "cuda_shfl_xor_sync_i32", "cuda_match_any_sync_i32",
      "cuda_match_all_sync_i32", "cuda_active_mask",  "cuda_fns_u32",         "amdgpu_clock_i64",
      "cpu_clock_i64",
  };
  return kMemoryFree.count(name) > 0;
}

bool split_is_recompute_safe(Block *block) {
  if (block == nullptr)
    return false;

  // (1) Loop-carried locals: reject a loop-written local shared across more than one construct.
  auto accesses = irpass::analysis::gather_statements(
      block, [](Stmt *s) { return !stmt_local_read_allocas(s).empty() || !stmt_local_write_allocas(s).empty(); });
  std::unordered_map<Stmt *, std::set<Stmt *>> read_owners;        // alloca -> top-level constructs that read it
  std::unordered_map<Stmt *, std::set<Stmt *>> loop_write_owners;  // alloca -> constructs that write it inside a loop
  for (Stmt *acc : accesses) {
    bool inside_container = false;
    Stmt *owner = top_level_owner(acc, block, &inside_container);
    if (owner == nullptr)
      continue;
    if (inside_container)
      for (Stmt *a : stmt_local_write_allocas(acc))
        loop_write_owners[a].insert(owner);
    for (Stmt *a : stmt_local_read_allocas(acc))
      read_owners[a].insert(owner);
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
  // Every intervening global write that could change a recomputed field/ndarray load, tagged with its top-level slot
  // and the pointer it targets (null => a footprint we cannot pin to one address: structural / external -> may-alias-
  // all). Gather LEAF effects only -- a container (loop / if) reports has_global_side_effect but its real writes are its
  // leaves -- and drop memory-free runtime intrinsics (barriers, shuffles, thread-index), which have a side effect but
  // write no addressable memory, so a construct sitting after a barrier still splits.
  struct GlobalWrite {
    int pos;
    Stmt *dest;
  };
  std::vector<GlobalWrite> global_writes;
  for (Stmt *w : gather_stmts_incl_containers(block, [](Stmt *s) {
         if (!stmt_is_task_effect(s) || s->is_container_statement())
           return false;
         if (auto *ifs = s->cast<InternalFuncStmt>())
           return !internal_func_is_memory_free(ifs->func_name);
         return true;
       })) {
    bool inside = false;
    Stmt *owner = top_level_owner(w, block, &inside);
    if (owner == nullptr)
      continue;
    auto it = top_index.find(owner);
    if (it != top_index.end())
      global_writes.push_back({it->second, write_effect_dest(w)});
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
    // Scan the slice: (2) reject a non-recomputable producer recomputed from another segment; (3) collect the global
    // reads recomputed from an earlier segment, checked against intervening aliasing writes below.
    struct RecomputedRead {
      int pos;
      Stmt *src;
    };
    std::vector<RecomputedRead> recomputed_reads;
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
      // Global read recomputed from an earlier segment (reads in k's own segment are not moved).
      if (stmt_is_global_read(s) && pos < b_lo)
        recomputed_reads.push_back({pos, s->cast<GlobalLoadStmt>()->src});
    }
    // (3) A recomputed read is unsafe only if a MAY-ALIAS write runs between the read's original slot and this
    // construct -- recomputing the read there would then observe that write. A write to a provably different address
    // (different SNode / ndarray, via the shared alias analysis) is harmless, which is what lets a multi-array kernel
    // split. A write we cannot pin to one address, or a null read pointer, may-aliases everything.
    for (const RecomputedRead &r : recomputed_reads)
      for (const GlobalWrite &w : global_writes)
        if (w.pos > r.pos && w.pos < b_lo)
          if (w.dest == nullptr || r.src == nullptr || irpass::analysis::maybe_same_address(r.src, w.dest))
            return false;
  }
  return true;
}

// Frontend work the split would do relative to the whole-kernel path. Each construct is compiled over its backward
// slice, so a shared serial prefix is recompiled once per construct: cost is `sum over constructs of |slice|` vs the
// whole-kernel single pass. The ratio is ~1 for near-disjoint constructs and grows toward ~#constructs when they share
// a large prefix. With no reuse cache that recompute is pure overhead, so `maybe_split_frontend_per_construct` falls
// back above a ratio cap.
struct SplitCost {
  long long sum_slice_stmts = 0;
  int total_stmts = 0;
  int n_constructs = 0;
  double ratio() const {
    return total_stmts > 0 ? (double)sum_slice_stmts / (double)total_stmts : 1.0;
  }
};

SplitCost estimate_split_cost(Block *block) {
  SplitCost c;
  c.total_stmts = (int)gather_stmts_incl_containers(block, [](Stmt *) { return true; }).size();
  auto segs = segment_top_level(block);
  auto alloca_writers = gather_top_level_alloca_writers(block);
  for (int k = 0; k < segs.n_segs; k++) {
    if (!segs.seg_emits_task[k])
      continue;
    c.n_constructs++;
    c.sum_slice_stmts += (long long)compute_construct_needed(block, segs.seg_id, k, alloca_writers).size();
  }
  return c;
}

// Split the flat top-level block into constructs, run the full per-construct frontend on each (recomputing shared
// top-level defs into every construct that reads them), and reassemble the produced OffloadedStmts in source order.
// Correctness for recompute-safe kernels rests on cross-construct global-temp hubs dissolving via recompute and
// cross-construct memory ordering being preserved by the source-order reassembly. The caller has already restricted
// this to autodiff_mode==kNone / non-mesh / recompute-safe kernels.
void split_frontend_per_construct(IRNode *ir, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  auto *block = ir->cast<Block>();
  QD_ASSERT(block != nullptr);
  const int n = (int)block->statements.size();
  auto segs = segment_top_level(block);
  auto alloca_writers = gather_top_level_alloca_writers(block);

  // Program-scoped stats (total / hit / recompiled per kernel), read back by the compilation manager to stay
  // backend-agnostic. `hit` stays 0 until a reuse tier lands.
  PerConstructCache *cc = (kernel->program != nullptr) ? &kernel->program->per_construct_cache() : nullptr;

  std::vector<std::unique_ptr<Stmt>> tasks;
  int n_constructs = 0, n_hit = 0, n_recompiled = 0;
  for (int k = 0; k < segs.n_segs; k++) {
    if (!segs.seg_emits_task[k])
      continue;  // pure-def-only serial run: no standalone task, recomputed into consuming constructs below
    n_constructs++;
    // Isolate construct k by its backward slice, keeping only the surviving top-level statements.
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

    run_construct_frontend(cb, config, kernel, verbose, /*construct_index=*/n_constructs - 1);
    n_recompiled++;
    // Reattach the construct's graph_do_while level onto its tasks so a do_while body stays one contiguous same-level
    // run after isolated offload (see `construct_gdw_level`).
    const int level = construct_gdw_level(block, segs.seg_id, k);
    while (!cb->statements.empty()) {
      auto t = cb->extract(0);
      if (level != kKeepOffloadGdwLevel)
        if (auto *off = t->cast<OffloadedStmt>())
          off->graph_do_while_level_id = level;
      tasks.push_back(std::move(t));
    }
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
  // QD_DUMP_CFG forces the whole-kernel path: cfg_optimization dumps the whole-kernel CFG and names files by phase, not
  // construct, so running the split under it would collide every construct's CFG into one filename. The other dump
  // flags (QD_DUMP_IR / QD_DUMP_SIMPLIFY / print_ir) are observation-only -- the split still runs and emits output per
  // construct. See docs/source/user_guide/optimization_passes.md.
  if (env_is_enabled(DUMP_CFG_ENV))
    return false;
  // QD_KERNEL_COVERAGE inserts per-line probe stores that add global-write constructs and, in graph/checkpoint kernels,
  // land between a yield gate and its loop; per-construct reassembly would move them and corrupt the coverage signal
  // and yield/resume behavior. It is a measurement mode, so keep it transparent by falling back.
  if (env_is_enabled("QD_KERNEL_COVERAGE"))
    return false;
  if (block_has_mesh_for(block))
    return false;
  if (block_has_concurrent_region(block))
    return false;  // concurrent constructs share one global-temp buffer; per-construct offload would alias offsets
  if (!split_is_recompute_safe(block))
    return false;
  // Cost guard: recompiling a large shared prefix once per construct is pure overhead with no reuse tier, so fall back
  // above a ratio cap. QD_SPLIT_MAX_COST_RATIO overrides the cap; QD_SPLIT_STATS logs the per-kernel estimate.
  SplitCost cost = estimate_split_cost(block);
  double max_ratio = 4.0;
  if (const char *r = std::getenv("QD_SPLIT_MAX_COST_RATIO"))
    max_ratio = std::atof(r);
  const bool would_fall_back = cost.ratio() > max_ratio;
  // QD_SPLIT_STATS: "1" prints the per-kernel cost; any other value is a file path to append to. Debug/tuning only.
  if (const char *sf = std::getenv("QD_SPLIT_STATS")) {
    std::ostringstream line;
    line << kernel->get_name() << " constructs=" << cost.n_constructs << " total_stmts=" << cost.total_stmts
         << " sum_slice_stmts=" << cost.sum_slice_stmts << " ratio=" << cost.ratio() << " max_ratio=" << max_ratio
         << (would_fall_back ? " -> FALLBACK" : " -> SPLIT") << "\n";
    if (std::string(sf) == "1") {
      std::cout << "[per-construct split] " << line.str();
    } else {
      std::ofstream ofs(sf, std::ios::app);
      if (ofs.good())
        ofs << line.str();
    }
  }
  if (would_fall_back)
    return false;
  split_frontend_per_construct(ir, config, kernel, verbose);
  return true;
}

}  // namespace irpass

}  // namespace quadrants::lang
