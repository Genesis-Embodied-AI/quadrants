#pragma once
#include "quadrants/ir/ir.h"

namespace quadrants::lang {

// Traverses blocks and the containers listed below, ignoring unhandled leaf statements by default.
// Overriding a container's visit replaces its traversal: call BasicStmtVisitor::visit(stmt) to retain it, or visit
// the desired child blocks explicitly. New container types need a traversal implementation here as well.
class BasicStmtVisitor : public IRVisitor {
 public:
  BasicStmtVisitor();

  // Called by the container visitors below before descending into child blocks; not called for Block itself.
  // Override this hook to inspect containers without replacing their traversal.
  virtual void preprocess_container_stmt(Stmt *stmt) {
  }

  void visit(Block *stmt_list) override;

  void visit(IfStmt *if_stmt) override;

  void visit(WhileStmt *stmt) override;

  void visit(RangeForStmt *for_stmt) override;

  void visit(StructForStmt *for_stmt) override;

  void visit(MeshForStmt *for_stmt) override;

  void visit(OffloadedStmt *stmt) override;

  void visit(FrontendWhileStmt *stmt) override;

  void visit(FrontendForStmt *stmt) override;

  void visit(FrontendIfStmt *stmt) override;
};

}  // namespace quadrants::lang
