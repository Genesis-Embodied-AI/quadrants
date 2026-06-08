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

// Run store-to-load forwarding + dead-store elimination over a single offloaded task's sub-block, scoped to
// that block alone. Correctness relies on the existing CFG boundary seeding: reaching_definition_analysis seeds
// the start node with all global pointers ("may contain data before this kernel") and live_variable_analysis
// seeds the final node with all global store destinations ("may be loaded after this kernel"). Because the CFG
// here spans only one task, every global address (fields, external tensors, global temporaries that carry data
// between tasks) is therefore conservatively treated as live-in and live-out of the task -- so no store that a
// sibling task may read is ever eliminated, and no value is forwarded across a task (device-launch) boundary.
bool optimize_offload_block(Block *block,
                            bool in_parallel_for,
                            bool after_lower_access,
                            bool autodiff_enabled,
                            const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &lva_config_opt) {
  if (block == nullptr || block->statements.empty()) {
    return false;
  }
  auto cfg = analysis::build_cfg(block, in_parallel_for);
  cfg->simplify_graph();
  bool modified = false;
  if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
    modified = true;
  }
  if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
    modified = true;
  }
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
  // optimize_offload_block above. Disabled for the real-matrix path (which skips the analyses entirely) and
  // pre-offload IR (no tasks yet), both of which fall through to the whole-kernel path below.
  if (config.cfg_optimization_per_task && !real_matrix_enabled) {
    auto tasks = collect_offloaded_tasks(root);
    if (!tasks.empty()) {
      bool result_modified = false;
      for (auto *off : tasks) {
        const bool body_parallel = off->task_type == OffloadedStmt::TaskType::range_for ||
                                    off->task_type == OffloadedStmt::TaskType::struct_for ||
                                    off->task_type == OffloadedStmt::TaskType::mesh_for;
        // Prologues/epilogues run serially; only the for-task body is parallel-executed.
        result_modified |= optimize_offload_block(off->tls_prologue.get(), false, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
        result_modified |= optimize_offload_block(off->mesh_prologue.get(), false, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
        result_modified |= optimize_offload_block(off->bls_prologue.get(), false, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
        result_modified |= optimize_offload_block(off->body.get(), body_parallel, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
        result_modified |= optimize_offload_block(off->bls_epilogue.get(), false, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
        result_modified |= optimize_offload_block(off->tls_epilogue.get(), false, after_lower_access,
                                                  autodiff_enabled, lva_config_opt);
      }
      // TODO: implement cfg->dead_instruction_elimination()
      die(root);  // remove unused allocas across the whole kernel
      return result_modified;
    }
  }

  auto cfg = analysis::build_cfg(root);

  const char *dump_cfg_env = std::getenv(DUMP_CFG_ENV.data());
  bool dump_cfg = dump_cfg_env != nullptr && std::string(dump_cfg_env) == "1";
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
