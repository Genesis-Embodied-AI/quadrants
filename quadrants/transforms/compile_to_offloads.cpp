#include "quadrants/ir/ir.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/pass.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/program/extension.h"
#include "quadrants/program/function.h"
#include "quadrants/program/kernel.h"
#include "quadrants/program/program.h"
#include "quadrants/codegen/llvm/per_task_artifact_cache.h"
#include "quadrants/program/per_construct_cache.h"
#include "quadrants/analysis/offline_cache_util.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/codegen/ir_dump.h"
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>

namespace quadrants::lang {

namespace {
// Run the whole pre-offload + offload frontend on ONE isolated top-level construct instead of once over the whole
// kernel. Mirrors the kNone / non-mesh portion of the whole-kernel sequence in
// `compile_to_offloads` between simplify_I and simplify_III, in the same order, so a per-construct compile produces the
// same tasks the whole-kernel path would for that construct. `cb` is a construct already isolated + recomputed (other
// constructs dropped, shared defs DIE'd) by `split_frontend_per_construct`.
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

// Deliberately a visitor rather than `gather_statements`: that helper only ever runs its predicate from
// `visit(Stmt *)`, and `BasicStmtVisitor` claims the container statements (`MeshForStmt`, `StructForStmt`,
// `RangeForStmt`, `IfStmt`, `WhileStmt`, `OffloadedStmt`) with typed overloads that recurse into the body without
// consulting the predicate. A `gather_statements` test for any of those types therefore never matches -- which is
// exactly how this guard silently passed every mesh kernel into the split.
class MeshForFinder : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  bool found{false};

  MeshForFinder() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(MeshForStmt *stmt) override {
    found = true;
  }
};

bool block_has_mesh_for(Block *block) {
  if (block == nullptr)
    return false;
  MeshForFinder finder;
  block->accept(&finder);
  return finder.found;
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

// Split-safety gate for the per-construct frontend split. A kernel is NOT safe to split when a top-level local variable's value is *produced inside one construct* (written by a store/atomic
// nested inside a top-level container / loop) and *consumed by a different construct* (read outside that producing
// loop). Whole-kernel offload carries such values across tasks via a cross-construct GlobalTemporary; isolating the
// consuming construct would drop the producing loop, so the value cannot be reconstructed (it is not recomputable from
// pure top-level defs). Such mutable shared allocas are rare in practice, so falling back to the whole-kernel path
// here costs little while keeping correctness in the general case. Recomputable cross-construct SSA -- consts, args,
// field-loads, top-level pure defs -- is fine: clone+die duplicates it into each construct.
bool split_is_recompute_safe(Block *block) {
  if (block == nullptr)
    return false;
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
    auto rit = read_owners.find(kv.first);
    if (rit == read_owners.end())
      continue;
    for (Stmt *r : rit->second) {
      if (kv.second.find(r) == kv.second.end())
        return false;  // a loop-produced local is read outside its producing loop -> cross-construct, not recomputable
    }
  }
  return true;
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

// Split the flat top-level block into constructs right after lower_ast + the structural prefix, run the full
// per-construct frontend on each (recomputing shared top-level defs into every construct that reads them), and
// reassemble the produced OffloadedStmts in source order. Compiling each construct independently keeps its
// simplify/merge_global_ptrs working set tiny, and is what lets its output be keyed, cached and skipped.
// Correctness for recompute-safe kernels rests on two things: cross-construct global-temp hubs dissolve via
// recompute, and cross-construct memory ordering is preserved by keeping tasks in original construct order. Restricted to autodiff_mode==kNone / non-mesh / recompute-safe kernels by the
// caller; anything else falls back to the whole-kernel path.
void split_frontend_per_construct(IRNode *ir, const CompileConfig &config, const Kernel *kernel, bool verbose) {
  auto *block = ir->cast<Block>();
  QD_ASSERT(block != nullptr);
  const int n = (int)block->statements.size();

  // Segment the top-level block exactly the way `offload` will chunk it into tasks: every container statement
  // (RangeFor/StructFor/While/...) is its own segment, and every maximal run of consecutive non-container (serial)
  // statements is one serial segment. A construct = a segment that emits a task: containers always do; a serial run
  // does iff it contains a real effect (stmt_is_task_effect). A serial run of only pure value defs / pointer chains
  // (e.g. dynamic loop bounds) emits no task and is recomputed into whichever construct consumes it.
  std::vector<int> seg_id(n, -1);
  std::vector<bool> seg_emits_task;  // per segment
  int cur_seg = -1;
  bool in_serial_run = false;
  for (int j = 0; j < n; j++) {
    Stmt *s = block->statements[j].get();
    if (s->is_container_statement()) {
      cur_seg = (int)seg_emits_task.size();
      seg_emits_task.push_back(true);
      in_serial_run = false;
    } else {
      if (!in_serial_run) {
        cur_seg = (int)seg_emits_task.size();
        seg_emits_task.push_back(false);
        in_serial_run = true;
      }
      if (stmt_is_task_effect(s))
        seg_emits_task[cur_seg] = true;
    }
    seg_id[j] = cur_seg;
  }
  const int n_segs = (int)seg_emits_task.size();

  // Top-level writers of each local variable, for the backward slice below. A `LocalLoadStmt`'s only operand is the
  // alloca (or a `MatrixPtrStmt` into it), never the store that gave it its value, so an operand-closed slice pulls in
  // a bare, zero-initialized alloca and silently reads zeros. `split_is_recompute_safe` has already rejected kernels
  // where a local is produced *inside* another construct's loop, so every writer that matters here is top-level and
  // safe to recompute into the consuming construct.
  std::unordered_map<Stmt *, std::vector<Stmt *>> alloca_writers;
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

  // Program-scoped per-construct FRONTEND cache. On a warm compile only the changed construct's
  // frontend re-runs; unchanged constructs are cloned out of the cache, skipping simplify/mgp/offload. Content-keyed,
  // so identical constructs across kernels (same ABI/config) share entries.
  PerConstructCache *cc = (kernel->program != nullptr) ? &kernel->program->per_construct_cache() : nullptr;

  // EXPERIMENT (no-construct-cache variation): bypass the in-memory per-construct FRONTEND cache to measure how much
  // the in-process frontend dedup (identical constructs within one kernel, and warm same-process recompiles) is worth
  // once the on-disk construct manifest is relied on alone. The cross-process manifest (`try_load`/store), the disk
  // artifact tier, `last_stats` and the `disk_plans` are all deliberately left intact -- only the in-process
  // `cc->entries` read/write is disabled. Flip to `true` to restore the baseline behaviour.
  constexpr bool kUseInMemoryConstructCache = false;

  // Cross-process construct manifests, naming the compiled tasks held by the per-task artifact cache.
  std::unique_ptr<ConstructManifestCache> manifest_store;
  std::unique_ptr<PerTaskArtifactCache> artifact_store;
  if (kernel->program != nullptr) {
    manifest_store =
        std::make_unique<ConstructManifestCache>(construct_manifest_dir_for(config.offline_cache_file_path));
    artifact_store =
        std::make_unique<PerTaskArtifactCache>(pertask_artifact_dir_for(config.offline_cache_file_path));
  }
  ConstructManifestCache *manifests = manifest_store.get();
  const DeviceCapabilityConfig dev_caps =
      (manifests != nullptr) ? kernel->program->get_device_caps() : DeviceCapabilityConfig{};
  std::vector<std::string> plan_construct_keys, plan_artifact_keys;  // parallel to `tasks`
  std::unordered_map<int, std::string> pending_manifest_key;         // construct ordinal -> cross-process key

  std::vector<std::unique_ptr<Stmt>> tasks;
  int n_constructs = 0, n_hit = 0, n_recompiled = 0, n_manifest_hit = 0;
  for (int k = 0; k < n_segs; k++) {
    if (!seg_emits_task[k])
      continue;  // pure-def-only serial run: no standalone task, recomputed into consuming constructs below
    n_constructs++;
    // Isolate construct k by its BACKWARD SLICE: keep segment k's whole subtree plus the transitive operand-def chain
    // it reads (const/arg/binop/pointer/field-load chains from earlier segments, recomputed into this construct), and
    // drop every other top-level statement (other constructs' loops and stores). The slice is closed under operands, so
    // no kept statement can reference a dropped one. This both keeps a store together with its own pointer operand and
    // recomputes cross-construct recomputable values (e.g. dynamic loop bounds), without the has_global_side_effect
    // heuristic that mis-stripped pointer chains.
    //
    // The slice is computed on the ORIGINAL block and only the surviving top-level statements are cloned. Cloning the
    // whole block first and deleting afterwards is O(constructs x block size) -- ~5.9 s on this kernel, and paid even
    // when every construct is a cache hit, since the isolated construct is what the key is computed from.
    std::unordered_set<Stmt *> needed;
    std::vector<Stmt *> worklist;
    auto add_with_subtree = [&needed, &worklist](Stmt *s) {
      if (needed.insert(s).second)
        worklist.push_back(s);
      for (Stmt *sub : irpass::analysis::gather_statements(s, [](Stmt *) { return true; }))
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
    std::vector<int> keep_indices;
    for (int j = 0; j < n; j++) {
      if (needed.find(block->statements[j].get()) != needed.end())
        keep_indices.push_back(j);
    }
    auto cloned = irpass::analysis::clone_block_subset(block, keep_indices);
    auto *cb = cloned.get();
    irpass::die(cb);  // clean up anything left dead after slicing (recompute per construct)

    // Key the isolated, re-id'd construct and consult the cache. Key + clone are cheap; the expensive
    // run_construct_frontend on a miss runs OUTSIDE the cache lock so parallel kernel compiles never serialize on it.
    bool hit = false;
    std::string ckey;
    if (cc != nullptr || manifests != nullptr) {
      irpass::re_id(cb);
      ckey = get_hashed_per_construct_cache_key(config, cb, kernel);
    }

    // Cross-process manifest. If a previous process already compiled this exact construct, it recorded the ordered
    // per-task artifact keys it produced. Reusing them lets us skip this construct's ENTIRE frontend
    // (merge_global_ptrs / full_simplify / offload) and its codegen: we emit one placeholder task per key and the
    // codegen driver loads the artifacts directly. This is the only path that removes the whole-kernel
    // merge_global_ptrs on a warm edit.
    if (manifests != nullptr) {
      const std::string dkey = get_hashed_per_construct_disk_key(config, dev_caps, cb, kernel);
      ConstructManifest man;
      if (manifests->try_load(dkey, &man) && !man.task_keys.empty()) {
        // The keys embed each task's index in the kernel (`...#<i>`), and that index is baked into the compiled
        // entry-fn / shared-array / adstack-counter names. Reuse is only valid if this construct lands at the same
        // offset, so verify before committing; a shift (an edit changed some earlier construct's task count) is a
        // miss and we just recompile.
        //
        // Every named artifact must also still be on disk. Committing to a placeholder is irreversible -- it discards
        // the construct's IR, so the codegen driver has nothing to fall back on and can only abort. The two tiers are
        // independent stores and either can lag the other (a pruned or half-written artifact directory, a manifest
        // recorded for tasks that were served from the in-memory cache and so never persisted), so treat a dangling
        // reference as an ordinary miss here, while the IR is still available.
        const int base = (int)tasks.size();
        bool usable = true;
        for (int t = 0; t < (int)man.task_keys.size(); t++) {
          const auto &kk = man.task_keys[t];
          const auto pos = kk.rfind('#');
          if (pos == std::string::npos || kk.substr(pos + 1) != std::to_string(base + t)) {
            usable = false;
            break;
          }
          if (artifact_store != nullptr && !artifact_store->exists(kk)) {
            usable = false;
            break;
          }
        }
        if (usable) {
          for (int t = 0; t < (int)man.task_keys.size(); t++) {
            // Placeholder: an empty serial task purely to hold the slot. It is never lowered -- codegen sees the
            // artifact key recorded for this index and loads the compiled task instead.
            auto ph = std::make_unique<OffloadedStmt>(OffloadedStmt::TaskType::serial, config.arch,
                                                      const_cast<Kernel *>(kernel));
            tasks.push_back(std::move(ph));
            plan_construct_keys.push_back(dkey);
            plan_artifact_keys.push_back(man.task_keys[t]);
          }
          n_manifest_hit++;
          continue;
        }
      }
      pending_manifest_key[n_constructs] = dkey;  // record so codegen can write the manifest after keying its tasks
    }

    if (kUseInMemoryConstructCache && cc != nullptr) {
      std::lock_guard<std::mutex> g(cc->mu);
      auto it = cc->entries.find(ckey);
      if (it != cc->entries.end()) {
        for (auto &t : it->second)
          tasks.push_back(irpass::analysis::clone(t.get()));
        hit = true;
      }
    }
    if (hit) {
      n_hit++;
      // Even on an in-memory hit these tasks belong to this construct, so tag them for the manifest write.
      for (size_t t = plan_construct_keys.size(); t < tasks.size(); t++) {
        plan_construct_keys.push_back(pending_manifest_key.count(n_constructs) ? pending_manifest_key[n_constructs]
                                                                              : std::string());
        plan_artifact_keys.push_back(std::string());
      }
      continue;
    }

    run_construct_frontend(cb, config, kernel, verbose);
    n_recompiled++;
    std::vector<std::unique_ptr<Stmt>> produced;
    while (!cb->statements.empty())
      produced.push_back(cb->extract(0));
    if (kUseInMemoryConstructCache && cc != nullptr) {
      std::vector<std::unique_ptr<Stmt>> stored;
      stored.reserve(produced.size());
      for (auto &t : produced)
        stored.push_back(irpass::analysis::clone(t.get()));
      std::lock_guard<std::mutex> g(cc->mu);
      cc->entries.emplace(ckey, std::move(stored));  // emplace: a racing store of the same key wins harmlessly
    }
    for (auto &t : produced)
      tasks.push_back(std::move(t));
    // Tag the freshly produced tasks with this construct so codegen can record its manifest once it has keyed them.
    for (size_t t = plan_construct_keys.size(); t < tasks.size(); t++) {
      plan_construct_keys.push_back(pending_manifest_key.count(n_constructs) ? pending_manifest_key[n_constructs]
                                                                            : std::string());
      plan_artifact_keys.push_back(std::string());
    }
  }
  while (!block->statements.empty())
    block->extract(0);
  for (auto &t : tasks)
    block->insert(std::move(t));
  if (cc != nullptr) {
    std::lock_guard<std::mutex> g(cc->mu);
    // A manifest hit is a cache hit from the caller's point of view: the construct's frontend was reused, just from
    // disk rather than from this process's memory. Same convention as the artifact-hit accounting in the codegen
    // driver, and it keeps `hit == total - 1` on a one-construct edit whether the reuse came from memory or disk.
    cc->last_stats[kernel->get_name()] = {n_constructs, n_hit + n_manifest_hit, n_recompiled};
    if (manifests != nullptr) {
      ConstructDiskPlan plan;
      plan.construct_key_by_task = std::move(plan_construct_keys);
      plan.artifact_key_by_task = std::move(plan_artifact_keys);
      cc->disk_plans[kernel->get_name()] = std::move(plan);
    }
  }
}
}  // namespace

namespace irpass {

void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         const Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);
  print("Initial IR");

  if (!verbose && config.print_preprocessed_ir && start_from_ast) {
    QD_INFO("[{}] {}:", kernel->get_name(), "Preprocessed IR");
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  }

  if (autodiff_mode == AutodiffMode::kReverse) {
    irpass::reverse_segments(ir);
    print("Segment reversed (for autodiff)");
  }

  const char *dump_ir_env = std::getenv(DUMP_IR_ENV.data());
  std::filesystem::path ir_dump_dir = config.debug_dump_path;
  bool should_dump = (dump_ir_env != nullptr && std::string(dump_ir_env) == "1");

  auto dump_ir = [&](const std::string &stage_name) {
    if (!should_dump)
      return;
    std::filesystem::create_directories(ir_dump_dir);
    std::filesystem::path filename = ir_dump_dir / (kernel->name + "_" + stage_name + ".ll");
    std::string ir_str;
    irpass::print(ir, &ir_str);
    std::ofstream ofs(filename.string());
    if (ofs.good()) {
      ofs << ir_str;
    }
  };

  dump_ir("from_ast");

  if (start_from_ast) {
    irpass::frontend_type_check(ir);
    irpass::lower_ast(ir);
  }

  dump_ir("quadrants1");
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::BeforeLowerAccess);
  irpass::analysis::gather_func_store_dests(ir);
  irpass::compile_quadrants_functions(ir, config, Function::IRStage::OptimizedIR);
  irpass::analysis::gather_func_store_dests(ir);

  irpass::eliminate_immutable_local_vars(ir);

  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  // TODO: strictly enforce bit vectorization for x86 cpu and CUDA now
  //       create a separate CompileConfig flag for the new pass
  if (arch_is_cpu(config.arch) || config.arch == Arch::cuda || config.arch == Arch::amdgpu) {
    irpass::bit_loop_vectorize(ir);
    irpass::type_check(ir, config);
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
  irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
    irpass::die(ir);
  }

  dump_ir("before_simplify_I");

  // For recompute-safe kernels (forward-only, non-mesh), run the remaining pre-offload + offload frontend PER
  // top-level construct and reassemble, instead of once over the whole kernel. The seam is here, right after
  // lower_ast + the structural prefix (function inlining, matrix-ptr lowering, bit-loop vectorize) and before the
  // expensive simplify/merge_global_ptrs/offload passes. Anything not recompute-safe (autodiff, mesh-for) falls
  // through to the whole-kernel path below.
  if (autodiff_mode == AutodiffMode::kNone && !block_has_mesh_for(ir->cast<Block>()) &&
      split_is_recompute_safe(ir->cast<Block>())) {
    split_frontend_per_construct(ir, config, kernel, verbose);
    dump_ir("after_offload");
    return;
  }

  irpass::full_simplify(
      ir, config,
      {false, /*autodiff_enabled*/ autodiff_mode != AutodiffMode::kNone, kernel->get_name(), verbose, "simplify_I"});
  irpass::analysis::verify_if_debug(ir, config);
  dump_ir("after_simplify_I");

  irpass::handle_external_ptr_boundary(ir, config);

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::analysis::gather_meshfor_relation_types(ir);
  }

  if (config.debug && autodiff_mode == AutodiffMode::kCheckAutodiffValid) {
    // Check whether the kernel obeys the autodiff limitation e.g., gloabl data
    // access rule
    // This check should be performed in the forward kernel i.e., autodiff_mode
    // == AutodiffMode::kCheckAutodiffValid
    irpass::demote_atomics(ir, config);
    irpass::differentiation_validation_check(ir, config, kernel->get_name());
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (autodiff_mode == AutodiffMode::kReverse || autodiff_mode == AutodiffMode::kForward) {
    // Remove local atomics here so that we don't have to handle their gradients
    irpass::demote_atomics(ir, config);

    irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ true, kernel->get_name(), verbose, "pre_autodiff"});
    irpass::auto_diff(ir, config, autodiff_mode, ad_use_stack);
    // TODO: Be carefull with the full_simplify when do high-order autodiff
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "post_autodiff"});
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.check_out_of_bound) {
    irpass::check_out_of_bound(ir, config, {kernel->get_name()});
    irpass::analysis::verify_if_debug(ir, config);
  }

  // Merge a global's separate read/write GlobalPtrStmts (same address) into one shared, activate=true pointer BEFORE
  // this first flag_access, so flag_access cannot stamp a read-only (activate=false) copy that the CSE eliminability
  // rule then refuses to re-merge with the in-loop write. Without it, cache_loop_invariant_global_vars sees a split
  // read/write and cannot cache conditional/in-if stores -> the -88% solver break-flag bug + the lost duck_in_box
  // optimization. On main this fell out of whole_kernel_cse running inside every full_simplify fixpoint; per-task CSE
  // does no pre-offload whole-kernel CSE, so we do this one cheap, pointers-only pass here instead (arithmetic is
  // already canonical after simplify_I, so a single call is enough; running it in the fixpoint was a +12-22s
  // compile regression for no extra benefit).
  irpass::merge_global_ptrs(ir);
  irpass::analysis::verify_if_debug(ir, config);

  irpass::flag_access(ir);
  irpass::analysis::verify_if_debug(ir, config);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_II"});
  irpass::analysis::verify_if_debug(ir, config);

  irpass::offload(ir, config);
  irpass::analysis::verify_if_debug(ir, config);

  dump_ir("after_offload");

  // Full per-task CSE now, before flag_access #2 splits a global's read/write pointers by access flag and before
  // simplify_III's LICM hoists the read pointer out of the loop. This restores the pointer-unification that main
  // gets from whole_kernel_cse running inside the post-offload full_simplify (per-task CSE otherwise defers to the
  // codegen workers, which run after cache_loop_invariant_global_vars). Needed for ndarrays, which only become
  // ExternalPtrStmts during offload and so cannot be reached by the pre-offload merge_global_ptrs. See the pass.
  // Gated on opt_level like all other CSE (per_task_cse / upstream whole_kernel_cse): at opt_level 0 there is no CSE
  // to require pointer unification, matching upstream behaviour.
  if (config.opt_level > 0) {
    irpass::cse_offloaded_tasks(ir);
  }

  // NOTE: There was an additional CFG pass here, removed in
  // https://github.com/taichi-dev/taichi/pull/8691
  irpass::flag_access(ir);

  irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_III"});
  irpass::analysis::verify_if_debug(ir, config);

  dump_ir("after_simplify_III");

  // Run the adstack-size pre-pass here, before the per-task split in `KernelCodeGen::compile_kernel_to_module`
  // and before `make_cpu_multithreaded_range_for` in `offload_to_executable` rewrites user ranges into chunk
  // wrappers. The kernel IR still has every `OffloadedStmt` as a sibling in the top-level block, so the pre-
  // pass can resolve a `GlobalLoadStmt(GlobalTemporaryStmt)` source by walking across tasks: prep serial tasks
  // that store a dynamic range bound (e.g. `arr.shape[0]` lowered via `offload::PromoteIntermediateToGlobalTmp`)
  // are still visible alongside the consuming range-for task. Gated on the same reverse+ad_use_stack predicate
  // the per-task call used so compile behaviour is unchanged for forward-only kernels.
  if (autodiff_mode == AutodiffMode::kReverse && ad_use_stack) {
    irpass::determine_ad_stack_size(ir, config);
    print("Autodiff stack size determined");
  }
}

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local) {
  QD_AUTO_PROF;

  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, kernel->get_name(), ir);

  // TODO: This is just a proof that we can demote struct-fors after offloading.
  // Eventually we might want the order to be TLS/BLS -> demote struct-for.
  // For now, putting this after TLS will disable TLS, because it can only
  // handle range-fors at this point.

  auto amgr = std::make_unique<AnalysisManager>();

  print("Start offload_to_executable");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.detect_read_only) {
    irpass::detect_read_only(ir);
    print("Detect read-only accesses");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted I");
  irpass::analysis::verify_if_debug(ir, config);

  if (config.cache_loop_invariant_global_vars) {
    irpass::cache_loop_invariant_global_vars(ir, config);
    print("Cache loop-invariant global vars");
  }

  if (config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::type_check(ir, config);
    print("Dense struct-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (config.make_cpu_multithreading_loop && arch_is_cpu(config.arch)) {
    irpass::make_cpu_multithreaded_range_for(ir, config);
    irpass::type_check(ir, config);
    print("Make CPU multithreaded range-for");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (is_extension_supported(config.arch, Extension::mesh) && config.demote_no_access_mesh_fors) {
    irpass::demote_no_access_mesh_fors(ir);
    irpass::type_check(ir, config);
    print("No-access mesh-for demoted");
    irpass::analysis::verify_if_debug(ir, config);
  }

  if (make_thread_local) {
    irpass::make_thread_local(ir, config);
    print("Make thread local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::make_mesh_thread_local(ir, config, {kernel->get_name()});
    print("Make mesh thread local");
    if (config.make_mesh_block_local && config.arch == Arch::cuda) {
      irpass::make_mesh_block_local(ir, config, {kernel->get_name()});
      print("Make mesh block local");
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_X"});
      print("Simplified X");
    }
  }

  if (make_block_local) {
    irpass::make_block_local(ir, config, {kernel->get_name(), verbose});
    print("Make block local");
  }

  if (is_extension_supported(config.arch, Extension::mesh)) {
    irpass::demote_mesh_statements(ir, config, {kernel->get_name()});
    print("Demote mesh statements");
  }

  irpass::demote_atomics(ir, config);
  print("Atomics demoted II");
  irpass::analysis::verify_if_debug(ir, config);

  if (is_extension_supported(config.arch, Extension::quant) && config.quant_opt_atomic_demotion) {
    irpass::analysis::gather_uniquely_accessed_bit_structs(ir, amgr.get());
  }

  irpass::remove_range_assumption(ir);
  print("Remove range assumption");

  irpass::remove_loop_unique(ir);
  print("Remove loop_unique");
  irpass::analysis::verify_if_debug(ir, config);

  if (lower_global_access) {
    irpass::full_simplify(ir, config,
                          {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "before_lower_access"});
    print("Simplified before lower access");
    irpass::lower_access(ir, config, {kernel->no_activate, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);
  }

  irpass::demote_operations(ir, config);
  print("Operations demoted");

  irpass::full_simplify(ir, config,
                        {lower_global_access, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "simplify_IV"});
  print("Simplified IV");

  // `determine_ad_stack_size` used to run here, but the pre-pass needs the full kernel IR (all offloaded
  // tasks as siblings) so cross-task `GlobalTemporaryStmt` sources can be resolved. It now runs at the end
  // of `compile_to_offloads`, before the per-task split in `KernelCodeGen::compile_kernel_to_module`. The
  // `determine_ad_stack_size` parameter is kept in the signature for API stability but is no longer used.
  (void)determine_ad_stack_size;

  if (is_extension_supported(config.arch, Extension::quant)) {
    irpass::optimize_bit_struct_stores(ir, config, amgr.get());
    print("Bit struct stores optimized");
  }

  bool half2_optimization_enabled =
      (config.arch == Arch::cuda && config.half2_vectorization && !get_custom_cuda_library_path().empty());
  if (config.real_matrix_scalarize) {
    if (irpass::scalarize(ir, half2_optimization_enabled)) {
      irpass::die(ir);
      print("DIE");

      // Remove redundant MatrixInitStmt inserted during scalarization
      irpass::full_simplify(ir, config, {false, /*autodiff_enabled*/ false, kernel->get_name(), verbose, "scalarize"});
      print("Scalarized");
    }
  }

  // Final field registration correctness & type checking
  irpass::type_check(ir, config);
  irpass::analysis::verify_if_debug(ir, config);
}

void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           const Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local,
                           bool start_from_ast) {
  QD_AUTO_PROF;

  compile_to_offloads(ir, config, kernel, verbose, autodiff_mode, ad_use_stack, start_from_ast);

  offload_to_executable(ir, config, kernel, verbose,
                        /*determine_ad_stack_size=*/autodiff_mode == AutodiffMode::kReverse && ad_use_stack,
                        lower_global_access, make_thread_local, make_block_local);
}

void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      Function::IRStage target_stage) {
  QD_AUTO_PROF;

  auto current_stage = func->ir_stage();
  auto print = make_pass_printer(verbose, config.print_ir_dbg_info, func->get_name(), ir);
  print("Initial IR");

  if (target_stage >= Function::IRStage::BeforeLowerAccess && current_stage < Function::IRStage::BeforeLowerAccess) {
    if (autodiff_mode == AutodiffMode::kReverse) {
      irpass::reverse_segments(ir);
      print("Segment reversed (for autodiff)");
    }

    if (current_stage < Function::IRStage::InitialIR) {
      irpass::frontend_type_check(ir);
      irpass::lower_ast(ir);
      print("Lowered");
    }

    // Removes MatrixOfMatrixPtrStmt & MatrixOfGlobalPtrStmt
    irpass::lower_matrix_ptr(ir, config.force_scalarize_matrix);
    print("Matrix ptr lowered");

    irpass::demote_atomics(ir, config);
    print("Atomics demoted");
    irpass::associate_continue_scope(ir, config);
    print("Associated continue scope");
    func->set_ir_stage(Function::IRStage::BeforeLowerAccess);
  }

  if (config.force_scalarize_matrix) {
    irpass::scalarize(ir, false /*half2_optimization_enabled*/);
  }

  if (target_stage >= Function::IRStage::OptimizedIR && current_stage < Function::IRStage::OptimizedIR) {
    irpass::lower_access(ir, config, {{}, true});
    print("Access lowered");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::die(ir);
    print("DIE");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::flag_access(ir);
    print("Access flagged III");
    irpass::analysis::verify_if_debug(ir, config);

    irpass::type_check(ir, config);
    print("Typechecked");

    irpass::demote_operations(ir, config);
    print("Operations demoted");

    if (config.real_matrix_scalarize) {
      if (irpass::scalarize(ir)) {
        // Remove redundant MatrixInitStmt inserted during scalarization
        irpass::die(ir);
        print("Scalarized");
      }
    }

    irpass::full_simplify(ir, config, {true, autodiff_mode != AutodiffMode::kNone, func->get_name(), verbose, "final"});
    print("Simplified");
    irpass::analysis::verify_if_debug(ir, config);
    func->set_ir_stage(Function::IRStage::OptimizedIR);
  }
}

}  // namespace irpass

}  // namespace quadrants::lang
