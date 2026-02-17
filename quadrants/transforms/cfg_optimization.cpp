#include "quadrants/ir/ir.h"
#include "quadrants/ir/control_flow_graph.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/system/profiler.h"
#include "quadrants/codegen/ir_dump.h"

namespace quadrants::lang {

namespace irpass {
bool cfg_optimization(
    const CompileConfig &config,
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    bool real_matrix_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt,
    const std::string &kernel_name,
    const std::string &phase) {
  QD_AUTO_PROF;
  auto cfg = analysis::build_cfg(root);

  const char *dump_cfg_env = std::getenv(DUMP_CFG_ENV.data());
  bool dump_cfg = dump_cfg_env != nullptr && std::string(dump_cfg_env) == "1";
  if (dump_cfg) {
    std::string suffix =
        phase.empty() ? "_before_cfg_opt" : ("_" + phase + "_before_cfg_opt");
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
      std::string suffix =
          phase.empty() ? "_post_cfg_opt" : ("_" + phase + "_post_cfg_opt");
      cfg->dump_graph_to_file(config, kernel_name, suffix);
    }
  }
  // TODO: implement cfg->dead_instruction_elimination()
  die(root);  // remove unused allocas
  return result_modified;
}
}  // namespace irpass

}  // namespace quadrants::lang
