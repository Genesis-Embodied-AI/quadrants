#include "quadrants/ir/ir.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/ir/visitors.h"
#include "quadrants/transforms/utils.h"

namespace quadrants::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

/* This pass divides the range-for loop into multithreading blocks, and
 * inserts a new range-for loop that iterates over the number of threads
 * available on the CPU. The computing logics in the original range-for
 * loop are then packed into an inner serial for loop. In a nutshell,
 * the outer offloaded range-for loop is used to parallelize the computation,
 * and inner range-for loop conducts real computation logics.
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
 *     for __thread_id in range(8):
 *       block_begin = __thread_id * 128
 *       block_end = min(block_begin + 128, 1024)
 *       for i in range(block_begin, block_end):
 *           a[i] = i
 *
 * where 8 is the number of threads available on the CPU and cpu_per_worker_min_block_dim is set to 1.
 *
 * This pass is only applied to range-for loops that are offloaded to
 * CPUs. The number of threads is determined by the config option
 * "cpu_max_num_threads", and the minimum chunk size by
 * "cpu_per_worker_min_block_dim" (default 512).
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
    auto cpu_per_worker_min_block_dim =
        offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(config_.cpu_per_worker_min_block_dim)));
    auto num_threads = offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(config_.cpu_max_num_threads)));
    auto block_index = offloaded_body->insert(Stmt::make_typed<LoopIndexStmt>(offloaded, 0));

    // Retrieve range-for bounds.
    Stmt *begin_stmt;
    Stmt *end_stmt;
    if (offloaded->const_begin) {
      begin_stmt = offloaded_body->insert(
          Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, offloaded->begin_value)));
    } else {
      begin_stmt = offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(offloaded->begin_offset, PrimitiveType::i32));
      begin_stmt = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(begin_stmt));
    }
    if (offloaded->const_end) {
      end_stmt =
          offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, offloaded->end_value)));
    } else {
      end_stmt = offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(offloaded->end_offset, PrimitiveType::i32));
      end_stmt = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(end_stmt));
    }

    auto next_block_index = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, block_index, one));
    auto get_cpu_block_start_index_fn = [&](Stmt *index) {
      return offloaded_body->insert(Stmt::make_typed<InternalFuncStmt>(
          "get_cpu_block_start_index",
          std::vector<Stmt *>{begin_stmt, end_stmt, num_threads, cpu_per_worker_min_block_dim, index}, PrimitiveType::i32,
          /*with_runtime_context=*/false));
    };
    auto block_begin = get_cpu_block_start_index_fn(block_index);
    auto block_end = get_cpu_block_start_index_fn(next_block_index);

    // Create the serial inner loop.
    auto inner_loop = offloaded_body->insert(
        Stmt::make_typed<RangeForStmt>(block_begin, block_end, std::move(offloaded->body),
                                       /*is_bit_vectorized*/ false, /*num_cpu_threads*/ 1, /*block_dim*/ 1,
                                       /*strictly_serialized*/ true, offloaded->range_hint));

    irpass::replace_all_usages_with(inner_loop, offloaded, inner_loop);

    // Update the offloaded stmt.
    // The statement now iterates over max CPU thread numbers.
    // Therefore it has constant begin and end values.
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
