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
 * where 8 is the number of threads available on the CPU and cpu_min_range_for_block is set to 1.
 *
 * This pass is only applied to range-for loops that are offloaded to
 * CPUs. The number of threads is determined by the config option
 * "cpu_max_num_threads", and the minimum chunk size by
 * "cpu_min_range_for_block" (default 512).
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
    auto cast = [&](Stmt *value, DataType type) -> Stmt * {
      auto stmt = Stmt::make_typed<UnaryOpStmt>(UnaryOpType::cast_value, value);
      stmt->cast_type = type;
      return offloaded_body->insert(std::move(stmt));
    };
    // Scheduling offsets can exceed i32 even when the original loop bounds fit in i32.
    auto one = offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(int64(1))));
    auto minimal_block_range =
        offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(int64(config_.cpu_min_range_for_block))));
    auto num_threads =
        offloaded_body->insert(Stmt::make_typed<ConstStmt>(TypedConstant(int64(config_.cpu_max_num_threads))));
    auto thread_index = cast(offloaded_body->insert(Stmt::make_typed<LoopIndexStmt>(offloaded, 0)), PrimitiveType::i64);

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

    begin_stmt = cast(begin_stmt, PrimitiveType::i64);
    end_stmt = cast(end_stmt, PrimitiveType::i64);

    // Inner serial block range is
    // max(((end - begin) + (num_threads - 1)) / num_threads,
    // minimal_block_range)
    auto total_range = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::sub, end_stmt, begin_stmt));
    auto saturated_total_range = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
        BinaryOpType::sub,
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, total_range, num_threads)), one));
    auto block_range = offloaded_body->insert(
        Stmt::make_typed<BinaryOpStmt>(BinaryOpType::floordiv, saturated_total_range, num_threads));
    block_range =
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::max, block_range, minimal_block_range));

    // Inner loop begins at
    // begin + block_range * thread_id
    auto block_begin =
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::mul, block_range, thread_index));
    block_begin = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, begin_stmt, block_begin));

    // Inner loop ends at
    // min(block_begin + block_range), end))
    auto block_end =
        offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::add, block_begin, block_range));
    block_end = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::min, end_stmt, block_end));

    // Clamp unused chunks to the original end before converting back to i32. Otherwise a large minimum or bounds
    // near INT32_MAX could wrap an unused chunk's start and turn it into a huge or overlapping range.
    block_begin = offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(BinaryOpType::min, end_stmt, block_begin));
    block_begin = cast(block_begin, PrimitiveType::i32);
    block_end = cast(block_end, PrimitiveType::i32);

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
