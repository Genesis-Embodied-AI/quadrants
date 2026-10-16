#include <functional>

#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/transforms/utils.h"

namespace quadrants::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

/* This pass groups CPU loop iterations into blocks so the compiler can optimize each block as a serial loop. It creates
 * one block per configured worker. A parallel outer loop runs the blocks, and a serial inner loop executes the original
 * loop body over each block's consecutive iterations.
 *
 * For example, the following code:
 *
 *   @qd.kernel
 *   def foo():
 *     for i in range(1024):
 *       a[i] = i
 *
 * becomes:
 *
 *   @qd.kernel
 *   def foo():
 *     for block_index in range(8):
 *       block_begin = block_index * 128
 *       block_end = min(block_begin + 128, 1024)
 *       for i in range(block_begin, block_end):
 *           a[i] = i
 *
 * where 8 is the number of generated blocks and 1 <= cpu_per_worker_min_block_dim <= 128.
 *
 * This pass is only applied to range-for loops that are offloaded to CPUs. The number of blocks is determined by the
 * config option "cpu_max_num_threads", and the minimum block size by "cpu_per_worker_min_block_dim" (default 512).
 *
 * The effect is that more invarants in the inner most can be identified and
 * moved outside, so that LLVM has more chance to vectorize the innermost
 * loop. This pass especially accelerates simple single level loops, e.g.
 * memcpy and vecadd, even when the loop bounds are determined at runtime.
 */

class MakeCPUMultithreadedRangeFor : public BasicStmtVisitor {
 public:
  explicit MakeCPUMultithreadedRangeFor(const CompileConfig &config) : config_(config) {
  }

  void visit(Block *block) override {
    for (auto &s_ : block->statements) {
      s_->accept(this);
    }
  }

  void visit(OffloadedStmt *offloaded) override {
    if (offloaded->task_type != TaskType::range_for) {
      return;
    }

    auto offloaded_body = std::make_unique<Block>();
    auto one = offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 1)));
    // The C++ comments below illustrate the generated code. Configuration values are embedded at compile time.
    // const int32_t cpu_per_worker_min_block_dim = config_.cpu_per_worker_min_block_dim;
    Stmt *cpu_per_worker_min_block_dim =
        offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(config_.cpu_per_worker_min_block_dim)));
    // const int32_t num_blocks = config_.cpu_max_num_threads;
    Stmt *num_blocks = offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(config_.cpu_max_num_threads)));
    // block_index is the index in: for (int32_t block_index = 0; block_index < num_blocks; ++block_index).
    Stmt *block_index = offloaded_body->insert(Stmt::make_typed<LoopIndexStmt>(offloaded, 0));

    // Retrieve range-for bounds.
    Stmt *original_range_begin;
    Stmt *original_range_end;
    if (offloaded->const_begin) {
      original_range_begin = offloaded_body->insert(
          Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, offloaded->begin_value)));
    } else {
      original_range_begin =
          offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(offloaded->begin_offset, PrimitiveType::i32));
      original_range_begin = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(original_range_begin));
    }
    if (offloaded->const_end) {
      original_range_end =
          offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, offloaded->end_value)));
    } else {
      original_range_end =
          offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(offloaded->end_offset, PrimitiveType::i32));
      original_range_end = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(original_range_end));
    }

    // int32_t next_block_index = block_index + one;
    Stmt *next_block_index =
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, block_index, one));
    std::function<Stmt *(Stmt *)> get_cpu_per_worker_block_start_index_fn = [&](Stmt *index) -> Stmt * {
      return offloaded_body->insert(
          Stmt::make_typed<InternalFuncStmt>("get_cpu_per_worker_block_start_index",
                                             std::vector<Stmt *>{original_range_begin, original_range_end, num_blocks,
                                                                 cpu_per_worker_min_block_dim, index},
                                             PrimitiveType::i32,
                                             /*with_runtime_context=*/false));
    };
    Stmt *block_begin = get_cpu_per_worker_block_start_index_fn(block_index);
    Stmt *block_end = get_cpu_per_worker_block_start_index_fn(next_block_index);

    // Create the serial inner loop.
    auto inner_loop = offloaded_body->insert(
        Stmt::make_typed<RangeForStmt>(block_begin, block_end, std::move(offloaded->body),
                                       /*is_bit_vectorized*/ false, /*num_cpu_threads*/ 1, /*block_dim*/ 1,
                                       /*strictly_serialized*/ true, offloaded->range_hint));

    irpass::replace_all_usages_with(inner_loop, offloaded, inner_loop);

    // The outer loop now iterates over block indices from 0 to config_.cpu_max_num_threads - 1.
    // Its bounds are constant because the number of blocks is fixed at compile time.
    offloaded->const_begin = true;
    offloaded->const_end = true;
    offloaded->begin_value = 0;
    offloaded->end_value = config_.cpu_max_num_threads;
    offloaded->body = std::move(offloaded_body);
    offloaded->body->set_parent_stmt(offloaded);
    offloaded->block_dim = 1;
    modified_ = true;
  }

  static bool run(IRNode *root, const CompileConfig &config) {
    MakeCPUMultithreadedRangeFor pass(config);
    root->accept(&pass);
    return pass.modified_;
  }

 private:
  const CompileConfig &config_;
  bool modified_{false};
};
}  // namespace

namespace irpass {

void make_cpu_multithreaded_range_for(IRNode *root, const CompileConfig &config) {
  MakeCPUMultithreadedRangeFor::run(root, config);
}

}  // namespace irpass

}  // namespace quadrants::lang
