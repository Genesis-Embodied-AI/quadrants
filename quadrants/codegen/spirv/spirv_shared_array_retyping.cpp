#include "quadrants/codegen/spirv/spirv_shared_array_retyping.h"

#include "quadrants/ir/type_utils.h"

namespace quadrants::lang {
namespace spirv {
namespace {

// Follow MatrixPtrStmt::origin chains back to the source AllocaStmt.
// Assumes only MatrixPtrStmt's on the chain to get to the AllocStmt.
const AllocaStmt *trace_to_alloca(const Stmt *stmt) {
  if (auto *alloca = stmt->cast<AllocaStmt>())
    return alloca;
  if (auto *matrix_ptr = stmt->cast<MatrixPtrStmt>())
    return trace_to_alloca(matrix_ptr->origin);
  return nullptr;
}

}  // namespace

void scan_shared_atomic_allocs(Block *ir_block,
                               std::unordered_map<const Stmt *, bool> &out) {
  for (auto &s : ir_block->statements) {
    if (auto *atomic_stmt = s->cast<AtomicOpStmt>()) {
      if (auto *alloca = trace_to_alloca(atomic_stmt->dest)) {
        if (alloca->is_shared) {
          auto alloca_dtype = alloca->ret_type.ptr_removed();
          if (auto *tensor_type = alloca_dtype->cast<TensorType>()) {
            auto scalar_dtype = tensor_type->get_element_type();
            if (auto *nested = scalar_dtype->cast<TensorType>())
              scalar_dtype = nested->get_element_type();
            if (is_real(scalar_dtype)) {
              bool has_non_add = (atomic_stmt->op_type != AtomicOpType::add);
              auto [it, inserted] = out.emplace(alloca, has_non_add);
              if (!inserted)
                it->second = it->second || has_non_add;
            }
          }
        }
      }
    }
    // Recurse into sub-blocks.
    // StructForStmt and MeshForStmt are lowered before codegen.
    QD_ASSERT(!s->cast<StructForStmt>());
    QD_ASSERT(!s->cast<MeshForStmt>());
    if (auto *if_stmt = s->cast<IfStmt>()) {
      if (if_stmt->true_statements)
        scan_shared_atomic_allocs(if_stmt->true_statements.get(), out);
      if (if_stmt->false_statements)
        scan_shared_atomic_allocs(if_stmt->false_statements.get(), out);
    } else if (auto *range_for = s->cast<RangeForStmt>()) {
      scan_shared_atomic_allocs(range_for->body.get(), out);
    } else if (auto *while_stmt = s->cast<WhileStmt>()) {
      scan_shared_atomic_allocs(while_stmt->body.get(), out);
    }
  }
}

}  // namespace spirv
}  // namespace quadrants::lang
