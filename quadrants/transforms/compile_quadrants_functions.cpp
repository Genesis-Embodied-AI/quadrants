#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/ir/statements.h"
#include "quadrants/program/function.h"
#include "quadrants/program/compile_config.h"

namespace quadrants::lang {

class CompileQuadrantsFunctions : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  CompileQuadrantsFunctions(const CompileConfig &compile_config,
                            Function::IRStage target_stage)
      : compile_config_(compile_config), target_stage_(target_stage) {
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = stmt->func;
    const auto ir_type = func->ir_stage();
    if (ir_type < target_stage_) {
      irpass::compile_function(func->ir.get(), compile_config_, func,
                               /*autodiff_mode=*/AutodiffMode::kNone,
                               /*verbose=*/compile_config_.print_ir,
                               target_stage_);
      func->ir->accept(this);
    }
  }

  static void run(IRNode *ir,
                  const CompileConfig &compile_config,
                  Function::IRStage target_stage) {
    CompileQuadrantsFunctions ctf{compile_config, target_stage};
    ir->accept(&ctf);
  }

 private:
  const CompileConfig &compile_config_;
  Function::IRStage target_stage_;
};

namespace irpass {

void compile_quadrants_functions(IRNode *ir,
                                 const CompileConfig &compile_config,
                                 Function::IRStage target_stage) {
  QD_AUTO_PROF;
  CompileQuadrantsFunctions::run(ir, compile_config, target_stage);
}

}  // namespace irpass

}  // namespace quadrants::lang
