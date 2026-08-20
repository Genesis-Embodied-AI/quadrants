#include "quadrants/ir/ir.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/visitors.h"

namespace quadrants::lang {

class StmtSearcher : public BasicStmtVisitor {
 private:
  std::function<bool(Stmt *)> test_;
  bool include_containers_;
  std::vector<Stmt *> results_;

 public:
  using BasicStmtVisitor::visit;

  StmtSearcher(std::function<bool(Stmt *)> test, bool include_containers)
      : test_(test), include_containers_(include_containers) {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  void visit(Stmt *stmt) override {
    if (test_(stmt))
      results_.push_back(stmt);
  }

  void preprocess_container_stmt(Stmt *stmt) override {
    if (include_containers_ && test_(stmt))
      results_.push_back(stmt);
  }

  static std::vector<Stmt *> run(IRNode *root, const std::function<bool(Stmt *)> &test, bool include_containers) {
    StmtSearcher searcher(test, include_containers);
    root->accept(&searcher);
    return searcher.results_;
  }
};

namespace irpass::analysis {
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test,
                                      bool include_containers) {
  return StmtSearcher::run(root, test, include_containers);
}
}  // namespace irpass::analysis

}  // namespace quadrants::lang
