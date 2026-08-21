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

// Build the QD_DUMP_CFG filename suffix. |task_index| >= 0 tags the file with the offloaded task it belongs to
// (post-offload, one CFG per task); |task_index| < 0 is the whole-kernel graph (the non-offloaded fall-through
// path) and gets no task tag. |stage| is e.g. "_before_cfg_opt" / "_post_cfg_opt".
std::string cfg_dump_suffix(const std::string &phase, int task_index, const std::string &stage) {
  std::string suffix;
  if (!phase.empty()) {
    suffix += "_" + phase;
  }
  if (task_index >= 0) {
    suffix += "_task" + std::to_string(task_index);
  }
  suffix += stage;
  return suffix;
}

// Build and optimize a control-flow graph for a SINGLE offloaded task, scoped to that task alone.
//
// The task is temporarily moved into a throwaway wrapper block and run through the normal Block ->
// OffloadedStmt CFG construction, then moved back, leaving the IR shape unchanged. Building through a wrapper
// (instead of stitching together per-sub-block CFGs) is what makes this correct: the resulting CFG is
// byte-for-byte the slice that the whole-kernel CFG would build for this one task -- including the offloaded
// for-body's implicit-loop `continue` edges (which are wired by visit(OffloadedStmt), not by visit(Block)), the
// prologue/body/epilogue chaining, and the body's is_parallel_executed flag. Optimizing each sub-block in
// isolation would drop the `continue` loop-back edges and wrongly dead-store-eliminate a global store that
// precedes a `continue` (regression caught by test_cfg_continue).
//
// Scoping the analyses to one task is semantics-preserving because each offloaded task is a separate device
// launch and the existing CFG boundary seeding is conservative across the launch boundary:
// reaching_definition_analysis seeds the start node with all global pointers ("may already hold data") and
// live_variable_analysis seeds the final node with all global store destinations ("may be read later"). With
// the CFG spanning only one task, every global address -- fields, external tensors, and the global-temporary
// buffer that carries scalars between tasks -- is therefore treated as live-in and live-out of the task, so no
// store a sibling task may read is eliminated and no value is forwarded across a task (device-launch) boundary.
//
// |dump_cfg| (QD_DUMP_CFG=1) only adds a debugging side effect: it writes this task's own CFG to a file
// (tagged with |task_index|) before and after optimization. It does NOT change which analyses run, so the
// dump reflects exactly the per-task graph the compiler actually optimizes.
bool optimize_one_task(Block *parent,
                       OffloadedStmt *off,
                       bool after_lower_access,
                       bool autodiff_enabled,
                       const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &lva_config_opt,
                       bool dump_cfg,
                       const CompileConfig &config,
                       const std::string &kernel_name,
                       const std::string &phase,
                       int task_index) {
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
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, task_index, "_before_cfg_opt"));
    }
    cfg->simplify_graph();
    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
      modified = true;
    }
    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
      modified = true;
    }
    if (dump_cfg) {
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, task_index, "_post_cfg_opt"));
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

  // Per-offloaded-task scoping. Once the kernel is offloaded we optimize each task's CFG independently; and we
  // deliberately DITCH the expensive whole-kernel cfg_optimization in the pre-offload phase, relying on the
  // post-offload per-task cfg below to do the store-to-load forwarding + dead-store elimination once tasks
  // exist. The expensive (super-linear) reaching-definition / forwarding analyses otherwise run on the monolithic
  // pre-offload kernel IR -- where there are no tasks to scope to -- and dominate compile time. cfg_optimization
  // is an optimization, not a correctness pass, so dropping it pre-offload is safe; the only thing lost is
  // cross-task forwarding/DSE on the monolithic IR, which is invalid across separate device launches anyway.
  //
  // QD_DUMP_CFG (|dump_cfg|) is purely a debugging side effect and must NOT change any of the decisions below:
  // whichever graphs the compiler builds and optimizes, we additionally dump. Post-offload that means one file
  // per offloaded task (dumped inside optimize_one_task); the pre-offload phases build no CFG at all, so there is
  // nothing to dump there; the non-offloaded fall-through builds and dumps the whole-kernel graph.
  auto tasks = collect_offloaded_tasks(root);
  if (!tasks.empty()) {
    // Post-offload: per-task store-to-load forwarding + dead-store elimination (skipped for the real-matrix
    // path, matching the whole-kernel path which runs no analyses there).
    bool result_modified = false;
    if (!real_matrix_enabled) {
      auto *block = root->as<Block>();
      for (int task_index = 0; task_index < static_cast<int>(tasks.size()); task_index++) {
        result_modified |= optimize_one_task(block, tasks[task_index], after_lower_access, autodiff_enabled,
                                             lva_config_opt, dump_cfg, config, kernel_name, phase, task_index);
      }
    }
    // TODO: implement cfg->dead_instruction_elimination()
    die(root);  // remove unused allocas across the whole kernel
    return result_modified;
  }
  // No offloaded tasks yet. Within compile_to_offloads these are the pre-offload full_simplify calls on the
  // monolithic kernel IR (the phases below, all *before* irpass::offload): their whole-kernel cfg is the
  // (super-linear) reaching-definition / store-to-load analysis that dominates compile time, and it is
  // redundant because the post-offload per-task cfg ("simplify_III" onward) redoes the intra-task
  // store-to-load forwarding + dead-store elimination once tasks exist. So for exactly those phases we ditch
  // cfg, keeping only the cheap dead-alloca cleanup. For ANY other caller of full_simplify on non-offloaded
  // IR (unit tests, standalone blocks / function bodies that are never offloaded), we must still run the
  // whole-kernel cfg below, or its forwarding/DSE would be silently lost -- so we fall through.
  const bool pre_offload_compile_phase =
      phase == "simplify_I" || phase == "simplify_II" || phase == "pre_autodiff" || phase == "post_autodiff";
  if (pre_offload_compile_phase) {
    die(root);
    return false;
  }
  // else: fall through to the whole-kernel cfg path below.

  auto cfg = analysis::build_cfg(root);

  if (dump_cfg) {
    cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, -1, "_before_cfg_opt"));
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
      cfg->dump_graph_to_file(config, kernel_name, cfg_dump_suffix(phase, -1, "_post_cfg_opt"));
    }
  }
  // TODO: implement cfg->dead_instruction_elimination()
  die(root);  // remove unused allocas
  return result_modified;
}
}  // namespace irpass

}  // namespace quadrants::lang
