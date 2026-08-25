#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/control_flow_graph.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/system/profiler.h"
#include "quadrants/codegen/ir_dump.h"

namespace quadrants::lang {

namespace irpass {

namespace {

// Collect the top-level offloaded tasks of |root| iff |root| is an already-offloaded kernel body, i.e. a Block
// whose statements are all OffloadedStmt. Returns an empty vector otherwise (pre-offload IR, function bodies,
// non-Block roots). This is what lets the caller tell "post-offload" (run per-task cfg) from "pre-offload /
// other" (ditch cfg), since "an offloaded task" only exists post-offload.
std::vector<OffloadedStmt *> collect_offloaded_tasks(IRNode *root) {
  std::vector<OffloadedStmt *> tasks;
  auto *block = root->cast<Block>();
  if (block == nullptr || block->statements.empty()) {
    return tasks;
  }
  for (auto &stmt : block->statements) {
    if (!stmt->is<OffloadedStmt>()) {
      return {};  // not a pure offloaded kernel body -> whole-kernel path
    }
  }
  for (auto &stmt : block->statements) {
    tasks.push_back(stmt->as<OffloadedStmt>());
  }
  return tasks;
}

std::string cfg_dump_suffix(const std::string &phase, bool post, std::optional<int> task_index = std::nullopt) {
  std::string suffix;
  if (!phase.empty()) {
    suffix += "_" + phase;
  }
  if (task_index.has_value()) {
    suffix += "_task" + std::to_string(task_index.value());
  }
  suffix += post ? "_post_cfg_opt" : "_before_cfg_opt";
  return suffix;
}

// Kernel-wide task index for naming |root|'s CFG dump, or nullopt when |root| has none. Set only when |root| is a
// bare offloaded task (make_block_local's per-task full_simplify runs on a clone that carries OffloadedStmt::
// task_index); a Block or never-offloaded root has no index, so its dump keeps the unsuffixed whole-kernel name.
std::optional<int> root_task_index(IRNode *root) {
  auto *off = root->cast<OffloadedStmt>();
  if (off != nullptr && off->task_index >= 0) {
    return off->task_index;
  }
  return std::nullopt;
}

// Build and optimize the CFG for a SINGLE offloaded task, scoped to that task alone.
//
// The task is moved into a throwaway wrapper block, run through the normal Block -> OffloadedStmt CFG
// construction, then moved back (IR shape unchanged). Building through a wrapper -- instead of stitching
// together per-sub-block CFGs -- yields byte-for-byte the slice the whole-kernel CFG would build for this task:
// the offloaded for-body's implicit-loop `continue` edges (wired by visit(OffloadedStmt), not visit(Block)), the
// prologue/body/epilogue chaining, and the body's is_parallel_executed flag. Optimizing each sub-block in
// isolation would drop the `continue` loop-back edges and wrongly dead-store-eliminate a global store preceding
// a `continue` (regression: test_cfg_continue).
//
// Scoping to one task is safe because each task is a separate device launch and CFG boundary seeding is
// conservative across it: reaching-definition seeds every global pointer live-in and live-variable seeds every
// global store destination live-out, so nothing is forwarded or eliminated across the launch boundary.
bool optimize_one_task(Block *parent,
                       OffloadedStmt *off,
                       bool after_lower_access,
                       bool autodiff_enabled,
                       bool run_optimization,
                       const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &lva_config_opt,
                       bool dump_cfg,
                       const CompileConfig &config,
                       const std::string &kernel_name,
                       const std::string &phase,
                       int task_id) {
  const int location = parent->locate(off);
  QD_ASSERT(location != -1);
  Block wrapper;
  wrapper.insert(parent->extract(off));
  bool modified = false;
  {
    // |cfg| holds raw pointers into |wrapper| (its container nodes) and into the task's own sub-blocks; keep
    // both alive until the analyses are done, then move the task back before |wrapper| leaves scope.
    auto cfg = analysis::build_cfg(&wrapper);
    if (dump_cfg) {
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, /*post=*/false, task_id));
    }
    if (run_optimization) {
      cfg->simplify_graph();
      if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
        modified = true;
      }
      if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
        modified = true;
      }
      if (dump_cfg) {
        cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, /*post=*/true, task_id));
      }
    }
  }
  parent->insert(wrapper.extract(off), location);
  return modified;
}

}  // namespace

bool cfg_optimization(const CompileConfig &config,
                      IRNode *root,
                      bool after_lower_access,
                      bool autodiff_enabled,
                      bool real_matrix_enabled,
                      const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &lva_config_opt,
                      const std::string &kernel_name,
                      const std::string &phase) {
  QD_AUTO_PROF;

  const char *dump_cfg_env = std::getenv(DUMP_CFG_ENV.data());
  const bool dump_cfg = dump_cfg_env != nullptr && std::string(dump_cfg_env) == "1";

  // Post-offload we optimize each task's CFG independently; pre-offload (no tasks yet) we DITCH the whole-kernel
  // cfg_optimization, whose super-linear reaching-definition / forwarding analyses on the monolithic IR dominate
  // compile time.
  auto tasks = collect_offloaded_tasks(root);
  if (!tasks.empty()) {
    // Per-task store-to-load forwarding + dead-store elimination, skipped on the real-matrix path (like the
    // whole-kernel path).
    const bool run_optimization = !real_matrix_enabled;
    bool result_modified = false;
    auto *block = root->as<Block>();
    int task_index = 0;
    for (auto *off : tasks) {
      // Prefer the task's own kernel-wide index (set for the isolated per-task codegen path, where the block
      // holds one task); before codegen numbers them the tasks sit here in kernel order, so the loop index fits.
      const int dump_task_id = off->task_index >= 0 ? off->task_index : task_index;
      ++task_index;
      // Nothing to optimize or dump: skip the CFG build (keeps QD_DUMP_CFG zero-cost when off).
      if (!run_optimization && !dump_cfg) {
        continue;
      }
      result_modified |= optimize_one_task(block, off, after_lower_access, autodiff_enabled, run_optimization,
                                           lva_config_opt, dump_cfg, config, kernel_name, phase, dump_task_id);
    }
    // TODO: implement cfg->dead_instruction_elimination()
    die(root);  // remove unused allocas across the whole kernel
    return result_modified;
  }
  // No offloaded tasks yet. For the pre-offload compile phases below the whole-kernel cfg is redundant (the
  // post-offload per-task cfg redoes the intra-task forwarding + DSE once tasks exist), so we ditch it and keep
  // only the cheap dead-alloca cleanup. Any OTHER non-offloaded caller (unit tests, standalone blocks / function
  // bodies that are never offloaded) must still run the whole-kernel cfg below, or lose its forwarding/DSE.
  const bool pre_offload_compile_phase =
      phase == "simplify_I" || phase == "simplify_II" || phase == "pre_autodiff" || phase == "post_autodiff";
  if (pre_offload_compile_phase) {
    // Optimization is skipped here; QD_DUMP_CFG still builds the CFG only to dump the "before" graph (build_cfg
    // does not mutate IR). No "post" dump -- nothing changed it.
    if (dump_cfg) {
      auto cfg = analysis::build_cfg(root);
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, /*post=*/false, root_task_index(root)));
    }
    die(root);
    return false;
  }
  // else: fall through to the whole-kernel cfg path below.

  // Also handles a bare OffloadedStmt root (e.g. make_block_local's full_simplify on a codegen worker): its
  // task_index scopes the dump name so concurrent tasks do not collide; nullopt for whole-kernel roots.
  auto cfg = analysis::build_cfg(root);

  if (dump_cfg) {
    cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, /*post=*/false, root_task_index(root)));
  }

  bool result_modified = false;
  if (!real_matrix_enabled) {
    cfg->simplify_graph();

    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
      result_modified = true;
    }
    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
      result_modified = true;
    }

    if (dump_cfg) {
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, /*post=*/true, root_task_index(root)));
    }
  }
  // TODO: implement cfg->dead_instruction_elimination()
  die(root);  // remove unused allocas
  return result_modified;
}
}  // namespace irpass

}  // namespace quadrants::lang
