#include "quadrants/ir/visitors.h"
#include "quadrants/ir/ir.h"

namespace quadrants::lang {

// The ExtractLocalPointers pass aims at removing redundant ConstStmts and
// MatrixPtrStmts generated for any (AllocaStmt, integer) pair by extracting
// a unique copy for any future usage.
//
// Example for redundant stmts:
//   <i32> $0 = const 0
//   <i32> $1 = const 1
//   ...
//   <[Tensor (3, 3) f32]> $47738 = alloca
//   <i32> $47739 = const 0  [REDUNDANT]
//   <*f32> $47740 = shift ptr [$47738 + $47739]
//   $47741 : local store [$47740 <- $47713]
//   <i32> $47742 = const 1  [REDUNDANT]
//   <*f32> $47743 = shift ptr [$47738 + $47742]
//   $47744 : local store [$47743 <- $47716]
//   ...
//   <i32> $47812 = const 1  [REDUNDANT]
//   <*f32> $47813 = shift ptr [$47738 + $47812]  [REDUNDANT]
//   <f32> $47814 = local load [$47813]
class ExtractLocalPointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_map<std::pair<Stmt *, int>,
                     Stmt *,
                     hashing::Hasher<std::pair<Stmt *, int>>>
      first_matrix_ptr_;  // mapping an (AllocaStmt, integer) pair to the
                          // first MatrixPtrStmt representing it
  std::unordered_map<int, Stmt *>
      first_const_;  // mapping an integer to the first ConstStmt representing
                     // it
  Block *top_level_;

  explicit ExtractLocalPointers(IRNode *root);
  void visit(OffloadedStmt *stmt) override;
  void visit(MatrixPtrStmt *stmt) override;
  static bool run(IRNode *node);

 private:
  using BasicStmtVisitor::visit;
};

namespace irpass {
bool scalarize(IRNode *root, bool half2_optimization_enabled);
}  // namespace irpass

}  // namespace quadrants::lang
