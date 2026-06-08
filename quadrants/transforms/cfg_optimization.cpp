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
// non-Block roots), in which case the caller falls back to the whole-kernel CFG. This is what makes the
// per-task path activate only post-offload, where the notion of "an offloaded task" exists.
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
bool optimize_one_task(Block *parent,
                       OffloadedStmt *off,
                       bool after_lower_access,
                       bool autodiff_enabled,
                       const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &lva_config_opt) {
  const int location = parent->locate(off);
  QD_ASSERT(location != -1);
  Block wrapper;
  wrapper.insert(parent->extract(off));
  bool modified = false;
  {
    // |cfg| holds raw pointers into |wrapper| (its container nodes) and into the task's own sub-blocks; keep
    // both alive until the analyses are done, then move the task back before |wrapper| leaves scope.
    auto cfg = analysis::build_cfg(&wrapper);
    cfg->simplify_graph();
    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
      modified = true;
    }
    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
      modified = true;
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

  // Per-offloaded-task scoping: once the kernel is offloaded, optimize each task's CFG independently instead of
  // building one whole-kernel CFG across all tasks. This keeps the (super-linear) dataflow analyses small per
  // task without changing semantics -- see the comment on CompileConfig::cfg_optimization_per_task and
  // optimize_one_task above. Skipped for the real-matrix path (which runs no analyses) and for pre-offload IR
  // (no tasks yet); also skipped when CFG dumping is requested, so QD_DUMP_CFG keeps dumping the whole-kernel
  // graph. All of these fall through to the whole-kernel path below.
  const char *dump_cfg_env = std::getenv(DUMP_CFG_ENV.data());
  const bool dump_cfg = dump_cfg_env != nullptr && std::string(dump_cfg_env) == "1";
  if (config.cfg_optimization_per_task && !real_matrix_enabled && !dump_cfg) {
    auto tasks = collect_offloaded_tasks(root);
    if (!tasks.empty()) {
      auto *block = root->as<Block>();
      bool result_modified = false;
      for (auto *off : tasks) {
        result_modified |= optimize_one_task(block, off, after_lower_access, autodiff_enabled, lva_config_opt);
      }
      // TODO: implement cfg->dead_instruction_elimination()
      die(root);  // remove unused allocas across the whole kernel
      return result_modified;
    }
  }

  auto cfg = analysis::build_cfg(root);

  if (dump_cfg) {
    std::string suffix = phase.empty() ? "_before_cfg_opt" : ("_" + phase + "_before_cfg_opt");
    cfg->dump_graph_to_file(config, kernel_name, suffix);
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
      std::string suffix = phase.empty() ? "_post_cfg_opt" : ("_" + phase + "_post_cfg_opt");
      cfg->dump_graph_to_file(config, kernel_name, suffix);
    }
  }
  // TODO: implement cfg->dead_instruction_elimination()
  die(root);  // remove unused allocas
  return result_modified;
}
}  // namespace irpass

}  // namespace quadrants::lang
