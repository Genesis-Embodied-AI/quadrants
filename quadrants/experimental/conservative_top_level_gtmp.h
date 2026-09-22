#pragma once

#include "quadrants/inc/constants.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/type_utils.h"
#include "quadrants/ir/statements.h"

namespace quadrants::lang::experimental {

// Benchmark-only, always-on stress test. Snapshot every scalar result directly in
// the kernel's top-level block before simplification can discard it. Do not walk
// parallel loop bodies or nested serial control flow. Graph regions have already
// been flattened into tagged top-level statements by lower_ast.
//
// This deliberately includes constants and IR intermediates, not just named
// source variables. Existing cross-task promotion still runs normally; its slots
// start after these snapshots. This is an over-allocation stress test, not the
// proposed source-level ABI allocator.
inline void snapshot_top_level_scalars(IRNode *ir) {
  auto *block = ir->as<Block>();
  std::vector<Stmt *> values;
  for (auto &stmt : block->statements) {
    // Store statements carry their destination's type but produce no SSA value.
    if (stmt->is<LocalStoreStmt>() || stmt->is<GlobalStoreStmt>())
      continue;
    const auto type = stmt->ret_type;
    if (!type.is_pointer() && (is_integral(type) || is_real(type))) {
      values.push_back(stmt.get());
    }
  }
  std::size_t offset = 0;
  for (auto *value : values) {
    const auto size = std::max<std::size_t>(4, data_type_size(value->ret_type));
    offset = (offset + size - 1) / size * size;
    QD_ASSERT_INFO(offset + size < quadrants_global_tmp_buffer_size,
                   "Conservative top-level gtmp experiment exceeded the temporary buffer");
    auto *ptr = value->insert_after_me(Stmt::make<GlobalTemporaryStmt>(offset, value->ret_type));
    ptr->region_tag = value->region_tag;
    auto *store = ptr->insert_after_me(Stmt::make<GlobalStoreStmt>(ptr, value));
    store->region_tag = value->region_tag;
    offset += size;
  }
}

// Account for snapshots before the normal offload allocator assigns shared slots.
// Scan the surviving IR rather than keeping pointers to pre-optimization nodes.
inline std::size_t reserved_gtmp_bytes(IRNode *ir) {
  std::size_t end = 0;
  for (auto *stmt :
       irpass::analysis::gather_statements(ir, [](Stmt *stmt) { return stmt->is<GlobalTemporaryStmt>(); })) {
    auto *ptr = stmt->as<GlobalTemporaryStmt>();
    end = std::max(end, static_cast<std::size_t>(ptr->offset) +
                            std::max<std::size_t>(4, data_type_size(ptr->ret_type.ptr_removed())));
  }
  return end;
}

}  // namespace quadrants::lang::experimental
