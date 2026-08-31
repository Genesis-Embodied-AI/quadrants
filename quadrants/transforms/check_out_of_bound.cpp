#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/transforms/check_out_of_bound.h"
#include "quadrants/transforms/utils.h"
#include <set>

namespace quadrants::lang {

// TODO: also check RangeAssumptionStmt

class CheckOutOfBound : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  std::set<int> visited;
  DelayedIRModifier modifier;
  std::string kernel_name;
  // AMDGPU only: the in-kernel assert is non-terminating there (see runtime.cpp), so a wave that fails a
  // bounds check keeps executing instead of being killed. Clamp the offending access index into
  // [0, size-1] so the subsequent load/store stays in bounds (no GPU memory fault, which on some AMD
  // arches would escalate to an uncatchable HSA exception) while the recorded assertion still surfaces on
  // the host after sync. Other backends (CPU early-returns, CUDA `asm("exit;")`) never continue, so they
  // leave the access untouched and this is false for them.
  bool clamp_oob_index_ = false;

  explicit CheckOutOfBound(const std::string &kernel_name, bool clamp_oob_index)
      : kernel_name(kernel_name), clamp_oob_index_(clamp_oob_index) {
  }

  // Return clamp(index, 0, size_minus_one) = min(max(index, 0), size_minus_one), appending the ops to
  // `stmts`. Used to keep a post-assert AMDGPU access in bounds.
  Stmt *clamp_index_to_bounds(VecStatement &stmts, Stmt *index, Stmt *zero, Stmt *size_minus_one) {
    auto lo = stmts.push_back<BinaryOpStmt>(BinaryOpType::max, index, zero);
    return stmts.push_back<BinaryOpStmt>(BinaryOpType::min, lo, size_minus_one);
  }

  // Point `ptr`'s operand slot that currently holds `*index_slot` at `clamped` (updates use-def correctly).
  void redirect_index_operand(Stmt *ptr, Stmt **index_slot, Stmt *clamped) {
    int slot = ptr->locate_operand(index_slot);
    ptr->set_operand(slot, clamped);
  }

  bool is_done(Stmt *stmt) {
    return visited.find(stmt->instance_id) != visited.end();
  }

  void set_done(Stmt *stmt) {
    visited.insert(stmt->instance_id);
  }

  void visit(SNodeOpStmt *stmt) override {
    if (stmt->ptr != nullptr) {
      QD_ASSERT(stmt->ptr->is<GlobalPtrStmt>());
      // We have already done the check on its ptr argument. No need to do
      // anything here.
      return;
    }

    // TODO: implement bound check here for other situations.
  }

  void visit(ExternalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    auto new_stmts = VecStatement();
    auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));
    Stmt *result = new_stmts.push_back<ConstStmt>(TypedConstant(true));
    std::string msg = fmt::format("[kernel={}] Out of bound access to ndarray at arg {} with indices [", kernel_name,
                                  fmt::join(stmt->base_ptr->as<ArgLoadStmt>()->arg_id, ", "));
    std::vector<Stmt *> args;
    int flattened_element = 1;
    for (int i = 0; i < stmt->element_shape.size(); i++) {
      flattened_element *= stmt->element_shape[i];
    }
    for (int i = 0; i < stmt->indices.size(); i++) {
      auto lower_bound = zero;
      auto check_lower_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_ge, stmt->indices[i], lower_bound);
      Stmt *upper_bound{nullptr};

      auto ndim = stmt->ndim;
      if (i < ndim) {
        // Check for External Shape
        auto axis = i;
        upper_bound = new_stmts.push_back<ExternalTensorShapeAlongAxisStmt>(
            /*axis=*/axis,
            /*arg_id=*/stmt->base_ptr->as<ArgLoadStmt>()->arg_id);
      } else {
        // Check for Element Shape
        upper_bound = new_stmts.push_back<ConstStmt>(TypedConstant(flattened_element));
      }

      auto check_upper_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_lt, stmt->indices[i], upper_bound);
      auto check_i = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, check_lower_bound, check_upper_bound);
      result = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, result, check_i);

      auto input_index = stmt->indices[i];
      args.emplace_back(input_index);

      if (clamp_oob_index_) {
        // Clamp against the runtime external shape so the post-assert AMDGPU access stays in bounds.
        auto one = new_stmts.push_back<ConstStmt>(TypedConstant(1));
        auto size_minus_one = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::sub, upper_bound, one);
        auto clamped = clamp_index_to_bounds(new_stmts, stmt->indices[i], zero, size_minus_one);
        redirect_index_operand(stmt, &stmt->indices[i], clamped);
      }
    }

    for (int i = 0; i < stmt->indices.size(); i++) {
      if (i > 0)
        msg += ", ";
      msg += "%d";
    }
    msg += "]\n" + stmt->get_tb();

    new_stmts.push_back<AssertStmt>(result, msg, args);
    modifier.insert_before(stmt, std::move(new_stmts));
    set_done(stmt);
  }

  void visit(GlobalPtrStmt *stmt) override {
    if (is_done(stmt))
      return;
    auto snode = stmt->snode;
    bool has_offset = !(snode->index_offsets.empty());
    auto new_stmts = VecStatement();
    auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));
    Stmt *result = new_stmts.push_back<ConstStmt>(TypedConstant(true));

    std::string msg =
        fmt::format("(kernel={}) Accessing field ({}) of size (", kernel_name, snode->get_node_type_name_hinted());
    std::string offset_msg = "offset (";
    std::vector<Stmt *> args;
    for (int i = 0; i < stmt->indices.size(); i++) {
      int offset_i = has_offset ? snode->index_offsets[i] : 0;

      // Note that during lower_ast, index arguments to GlobalPtrStmt are
      // already converted to [0, +inf) range.

      auto lower_bound = zero;
      auto check_lower_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_ge, stmt->indices[i], lower_bound);
      int size_i = snode->shape_along_axis(i);
      int upper_bound_i = size_i;
      auto upper_bound = new_stmts.push_back<ConstStmt>(TypedConstant(upper_bound_i));
      auto check_upper_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_lt, stmt->indices[i], upper_bound);
      auto check_i = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, check_lower_bound, check_upper_bound);
      result = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, result, check_i);
      if (i > 0) {
        msg += ", ";
        offset_msg += ", ";
      }
      msg += std::to_string(size_i);
      offset_msg += std::to_string(offset_i);

      auto input_index = stmt->indices[i];
      if (offset_i != 0) {
        auto offset = new_stmts.push_back<ConstStmt>(TypedConstant(offset_i));
        input_index = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::add, input_index, offset);
      }
      args.emplace_back(input_index);

      if (clamp_oob_index_) {
        // Clamp against the field size so the post-assert AMDGPU access stays in bounds.
        auto size_minus_one = new_stmts.push_back<ConstStmt>(TypedConstant(size_i - 1));
        auto clamped = clamp_index_to_bounds(new_stmts, stmt->indices[i], zero, size_minus_one);
        redirect_index_operand(stmt, &stmt->indices[i], clamped);
      }
    }
    offset_msg += ") ";
    msg += ") " + (has_offset ? offset_msg : "") + "with indices (";
    for (int i = 0; i < stmt->indices.size(); i++) {
      if (i > 0)
        msg += ", ";
      msg += "%d";
    }
    msg += ")";
    msg += "\n" + stmt->get_tb();

    new_stmts.push_back<AssertStmt>(result, msg, args);
    modifier.insert_before(stmt, std::move(new_stmts));
    set_done(stmt);
  }

  // TODO: As offset information per dimension is lacking, only the accumulated
  // index is checked.
  void visit(MatrixPtrStmt *stmt) override {
    if (is_done(stmt) || !stmt->offset_used_as_index())
      return;

    auto const &matrix_shape = stmt->get_origin_shape();
    int max_valid_index = 1;
    for (int i = 0; i < matrix_shape.size(); i++) {
      max_valid_index *= matrix_shape[i];
    }
    // index starts from 0, max_valid_index = size(matrix) - 1
    max_valid_index -= 1;

    auto index = stmt->offset;
    auto new_stmts = VecStatement();
    auto zero = new_stmts.push_back<ConstStmt>(TypedConstant(0));
    Stmt *result = new_stmts.push_back<ConstStmt>(TypedConstant(true));

    auto lower_bound = zero;
    auto check_lower_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_ge, index, lower_bound);
    auto upper_bound = new_stmts.push_back<ConstStmt>(TypedConstant(max_valid_index));
    auto check_upper_bound = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::cmp_le, index, upper_bound);
    auto check_i = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, check_lower_bound, check_upper_bound);
    result = new_stmts.push_back<BinaryOpStmt>(BinaryOpType::bit_and, result, check_i);

    std::string msg = fmt::format("(kernel={}) Out of bound access to a [", kernel_name);
    for (int i = 0; i < matrix_shape.size(); i++) {
      if (i > 0)
        msg += ", ";
      msg += std::to_string(matrix_shape[i]);
    }
    msg += "] matrix with index [%d]\n" + stmt->get_tb();

    std::vector<Stmt *> args = {index};
    new_stmts.push_back<AssertStmt>(result, msg, args);

    if (clamp_oob_index_) {
      // Clamp the flattened matrix offset into [0, size-1] for the post-assert AMDGPU access.
      auto clamped = clamp_index_to_bounds(new_stmts, index, zero, upper_bound);
      redirect_index_operand(stmt, &stmt->offset, clamped);
    }

    modifier.insert_before(stmt, std::move(new_stmts));
    set_done(stmt);
  }

  void visit(BinaryOpStmt *stmt) override {
    // Insert assertions if debug is on
    if (is_done(stmt)) {
      return;
    }
    if (stmt->op_type == BinaryOpType::pow) {
      if (is_integral(stmt->rhs->ret_type) && is_integral(stmt->lhs->ret_type)) {
        auto compare_rhs = Stmt::make<ConstStmt>(TypedConstant(0));
        auto compare = std::make_unique<BinaryOpStmt>(BinaryOpType::cmp_ge, stmt->rhs, compare_rhs.get());
        compare->ret_type = PrimitiveType::i32;
        std::string msg = "Negative exponent in pow(int, int) is not allowed.";
        msg += "\n" + stmt->get_tb();
        auto assert_stmt = std::make_unique<AssertStmt>(compare.get(), msg, std::vector<Stmt *>());
        assert_stmt->accept(this);
        modifier.insert_before(stmt, std::move(compare_rhs));
        modifier.insert_before(stmt, std::move(compare));
        modifier.insert_before(stmt, std::move(assert_stmt));
        set_done(stmt);
      }
    }
  }

  static bool run(IRNode *node, const CompileConfig &config, const std::string &kernel_name) {
    // Only AMDGPU needs the post-assert index clamp: its in-kernel assert is non-terminating, so a failed
    // bounds check falls through to the access. CPU/CUDA stop the thread at the assert and never do.
    CheckOutOfBound checker(kernel_name, arch_is_amdgpu(config.arch));
    bool modified = false;
    while (true) {
      node->accept(&checker);
      if (checker.modifier.modify_ir()) {
        modified = true;
      } else {
        break;
      }
    }
    if (modified)
      irpass::type_check(node, config);
    return modified;
  }
};

const PassID CheckOutOfBoundPass::id = "CheckOutOfBoundPass";

namespace irpass {

bool check_out_of_bound(IRNode *root, const CompileConfig &config, const CheckOutOfBoundPass::Args &args) {
  QD_AUTO_PROF;
  return CheckOutOfBound::run(root, config, args.kernel_name);
}

}  // namespace irpass

}  // namespace quadrants::lang
