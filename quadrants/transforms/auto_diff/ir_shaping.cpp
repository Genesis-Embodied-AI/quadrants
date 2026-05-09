#include "quadrants/transforms/auto_diff/auto_diff_common.h"
#include "quadrants/transforms/auto_diff/ir_shaping.h"

namespace quadrants::lang {

namespace {

// ============================================================================
// RegulateTensorTypedStatements: rewrite tensor-typed local/global stores that touch a sub-tensor through MatrixPtr
// into an explicit gather + matrix-init + scalar store, so downstream passes never see a partial-tensor store.
// ============================================================================

class RegulateTensorTypedStatements : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;
  DelayedIRModifier delayed_modifier_;

  explicit RegulateTensorTypedStatements() {
  }

  template <typename Store, typename Load>
  void process_store_stmt(Store *stmt) {
    QD_ASSERT(stmt->template is<LocalStoreStmt>() || stmt->template is<GlobalStoreStmt>());

    if (stmt->dest->template is<MatrixPtrStmt>()) {
      auto matrix_ptr_stmt = stmt->dest->template as<MatrixPtrStmt>();
      auto orig_stmt = matrix_ptr_stmt->origin;

      if (!orig_stmt->ret_type.ptr_removed()->template is<TensorType>()) {
        return;
      }

      auto tensor_type = orig_stmt->ret_type.ptr_removed()->template as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();

      if (matrix_ptr_stmt->offset->template is<ConstStmt>()) {
        /*
          [Static index]
          Fwd:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, 2 // offset = 2
          $3 : local store $2, $val

          Replaced:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, 2 // --> erase

          $3 = matrix ptr $1, 0
          $4 = load $3

          $5 = matrix ptr $1, 1
          $6 = load $5

          $7 = matrix ptr $1, 3
          $8 = load $7

          $9 = matrix init [$4, $6, $val, $8]

          $10 : store $0, $9
        */
        int offset = matrix_ptr_stmt->offset->template as<ConstStmt>()->val.val_int32();

        QD_ASSERT(offset < num_elements);

        std::vector<Stmt *> values;
        for (int i = 0; i < num_elements; i++) {
          if (i == offset) {
            values.push_back(stmt->val);
            continue;
          }

          auto const_i = insert_const(PrimitiveType::i32, stmt, i, true);
          auto matrix_ptr_stmt_i = Stmt::make<MatrixPtrStmt>(orig_stmt, const_i);
          matrix_ptr_stmt_i->ret_type = tensor_type->get_element_type();

          auto local_load_stmt_i = Stmt::make<Load>(matrix_ptr_stmt_i.get());
          local_load_stmt_i->ret_type = tensor_type->get_element_type();

          values.push_back(local_load_stmt_i.get());

          stmt->insert_before_me(std::move(matrix_ptr_stmt_i));
          stmt->insert_before_me(std::move(local_load_stmt_i));
        }

        auto matrix_init_stmt = Stmt::make<MatrixInitStmt>(values);
        matrix_init_stmt->ret_type = tensor_type;

        auto store_stmt = Stmt::make<Store>(orig_stmt, matrix_init_stmt.get());
        stmt->insert_before_me(std::move(matrix_init_stmt));
        stmt->replace_with(std::move(store_stmt));

        return;

      } else {
        /*
          [Dynamic index]
          Fwd:
          $0 = alloca <4 x i32>
          $1 = load $0
          $2 = matrix ptr $1, $offset // offset = 2
          $3 : local store $2, $val

          Replaced:
          $0 = alloca <4 x i32>

          $1 = load $0
          $2 = matrix init [$val, $val, $val, $val]

          $3 = matrix init [$offset, $offset, $offset, $offset]
          $4 = matrix init [0, 1, 2, 3]

          $5 = bin_eq $3, $4
          $6 = select $5, $2, $1

          $7 : store $0, $6
        */
        auto tensor_type = orig_stmt->ret_type.ptr_removed()->template as<TensorType>();
        auto num_elements = tensor_type->get_num_elements();

        auto tensor_shape = tensor_type->get_shape();
        auto index_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::i32);

        std::vector<Stmt *> val_values(num_elements, stmt->val);
        std::vector<Stmt *> offset_values(num_elements, matrix_ptr_stmt->offset);
        std::vector<Stmt *> index_values(num_elements);
        for (int i = 0; i < num_elements; i++) {
          index_values[i] = insert_const(PrimitiveType::i32, stmt, i, true);
        }

        auto matrix_val = Stmt::make<MatrixInitStmt>(val_values);
        matrix_val->ret_type = tensor_type;

        auto matrix_offset = Stmt::make<MatrixInitStmt>(offset_values);
        matrix_offset->ret_type = index_tensor_type;

        auto matrix_index = Stmt::make<MatrixInitStmt>(index_values);
        matrix_index->ret_type = index_tensor_type;
        auto cmp_tensor_type = TypeFactory::get_instance().get_tensor_type(tensor_shape, PrimitiveType::u1);
        auto matrix_eq = Stmt::make<BinaryOpStmt>(BinaryOpType::cmp_eq, matrix_offset.get(), matrix_index.get());
        matrix_eq->ret_type = cmp_tensor_type;

        auto orig_value = Stmt::make<Load>(orig_stmt);
        orig_value->ret_type = tensor_type;

        auto matrix_select =
            Stmt::make<TernaryOpStmt>(TernaryOpType::select, matrix_eq.get(), matrix_val.get(), orig_value.get());
        matrix_select->ret_type = tensor_type;

        auto store_stmt = Stmt::make<Store>(orig_stmt, matrix_select.get());

        stmt->insert_before_me(std::move(matrix_val));
        stmt->insert_before_me(std::move(matrix_offset));
        stmt->insert_before_me(std::move(matrix_index));
        stmt->insert_before_me(std::move(matrix_eq));
        stmt->insert_before_me(std::move(orig_value));
        stmt->insert_before_me(std::move(matrix_select));
        stmt->replace_with(std::move(store_stmt));
        return;
      }
    }
  }

  void visit(LocalStoreStmt *stmt) override {
    process_store_stmt<LocalStoreStmt, LocalLoadStmt>(stmt);
  }

  void visit(GlobalStoreStmt *stmt) override {
    process_store_stmt<GlobalStoreStmt, GlobalLoadStmt>(stmt);
  }

  static void run(IRNode *root) {
    RegulateTensorTypedStatements pass;
    root->accept(&pass);
  }
};

// ============================================================================
// Independent-Blocks discovery: IBJudger / DupCleaner / IdentifyIBs.
//
// Independent Block (IB): a block (i.e. loop body) whose iterations are independent of previous iterations and outer
// scopes. IBs are where MakeAdjoint emits the reverse pass; outside an IB only iteration order matters and
// ReverseOuterLoops handles that.
// ============================================================================

class IndependentBlocksJudger : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(LocalLoadStmt *stmt) override {
    QD_ASSERT(stmt->src->is<AllocaStmt>() || stmt->src->is<MatrixPtrStmt>() || stmt->src->is<MatrixOfMatrixPtrStmt>());
    touched_allocas_.insert(stmt->src);
  }

  void visit(LocalStoreStmt *stmt) override {
    QD_ASSERT(stmt->dest->is<AllocaStmt>() || stmt->dest->is<MatrixPtrStmt>() ||
              stmt->dest->is<MatrixOfMatrixPtrStmt>());
    touched_allocas_.insert(stmt->dest);
  }

  void visit(AtomicOpStmt *stmt) override {
    // We don't need to check the global atomics inside the range for-loops because:
    //   1. If the range for-loop is innermost, they are captured by MakeAdjoint anyway.
    //   2. If the range for-loop is not innermost, they are processed by another IndependentBlocksJudger.
    if (is_inside_loop_)
      return;

    Stmt *dest = stmt->dest;
    if (dest->is<MatrixPtrStmt>()) {
      dest = dest->as<MatrixPtrStmt>()->origin;
    }

    if (dest->is<ExternalPtrStmt>()) {
      if (dest->as<ExternalPtrStmt>()
              ->base_ptr->as<ArgLoadStmt>()
              ->ret_type.ptr_removed()
              ->as<StructType>()
              ->elements()
              .size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) {
        qualified_glb_operations_ = true;
      }
    } else {
      QD_ASSERT(dest->is<GlobalPtrStmt>());
      if (dest->as<GlobalPtrStmt>()->snode->has_adjoint()) {
        qualified_glb_operations_ = true;
      }
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    // We don't need to check the global load inside the range for-loops because:
    //   1. If the range for-loop is innermost, they are captured by MakeAdjoint anyway.
    //   2. If the range for-loop is not innermost, they are processed by another IndependentBlocksJudger.
    if (is_inside_loop_)
      return;

    Stmt *src = stmt->src;
    if (src->is<MatrixPtrStmt>()) {
      src = src->as<MatrixPtrStmt>()->origin;
    }

    if ((src->is<ExternalPtrStmt>() && src->as<ExternalPtrStmt>()
                                               ->base_ptr->as<ArgLoadStmt>()
                                               ->ret_type.ptr_removed()
                                               ->as<StructType>()
                                               ->elements()
                                               .size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY) ||
        (src->is<GlobalPtrStmt>() && src->as<GlobalPtrStmt>()->snode->has_adjoint())) {
      qualified_glb_operations_ = true;
    }
  }

  void visit(RangeForStmt *stmt) override {
    inner_most_loop_ = false;
    is_inside_loop_ = true;
    stmt->body->accept(this);
    is_inside_loop_ = false;
  }

  static void run(IRNode *root, IndependentBlockMetaData &ib_meta_data) {
    IndependentBlocksJudger Judger;
    Block *block = root->as<Block>();
    root->accept(&Judger);
    std::set<Block *> outside_blocks;
    // Collect all parent blocks (i.e. outside blocks) of the current block for local load/store stmt checks.
    for (auto b = block->parent_block(); b; b = b->parent_block()) {
      if (b)
        outside_blocks.insert(b);
    }
    for (const auto &alloca : Judger.touched_allocas_) {
      // Test if the alloca belongs to the current block.
      if (outside_blocks.find(alloca->parent) != outside_blocks.end()) {
        // This block is not an IB since it loads/modifies outside variables.
        ib_meta_data.is_ib = false;
      }
    }

    // IB classification rules:
    //   - To be an IB, the block must have no local load/store to allocas outside itself (enforced above).
    //   - To be a smallest IB on top of that, it must be an inner-most loop or a block without qualified global
    //     atomics / global loads.
    ib_meta_data.is_smallest_ib = ib_meta_data.is_ib && (Judger.qualified_glb_operations_ || Judger.inner_most_loop_);
  }

 private:
  std::set<Stmt *> touched_allocas_;
  bool qualified_glb_operations_ = false;
  bool inner_most_loop_ = true;
  bool is_inside_loop_ = false;
};

// Remove duplicated IBs and remove blocks that are others' children, so each block is processed at most once.
class DuplicateIndependentBlocksCleaner : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void check_children_ib(Block *target_block) {
    // Remove the block if it is the child of the block being visiting
    if (independent_blocks_cleaned_.find(target_block) != independent_blocks_cleaned_.end()) {
      independent_blocks_cleaned_.erase(target_block);
    }
  }

  void visit(StructForStmt *stmt) override {
    check_children_ib(stmt->body.get());
    stmt->body->accept(this);
  }
  void visit(RangeForStmt *stmt) override {
    check_children_ib(stmt->body.get());
    stmt->body->accept(this);
  }

  static std::set<Block *> run(const std::vector<std::pair<int, Block *>> &raw_IBs) {
    DuplicateIndependentBlocksCleaner cleaner;
    // Remove duplicate IBs
    for (auto const &item : raw_IBs) {
      cleaner.independent_blocks_cleaned_.insert(item.second);
    }
    // No clean is needed if only one IB exists
    if (cleaner.independent_blocks_cleaned_.size() > 1) {
      // Check from the block with smallest depth, ensure no duplicate visit happens.
      for (const auto &block : cleaner.independent_blocks_cleaned_) {
        block->accept(&cleaner);
      }
    }
    return cleaner.independent_blocks_cleaned_;
  }

 private:
  std::set<Block *> independent_blocks_cleaned_;
};

class IdentifyIndependentBlocks : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  void visit(WhileStmt *stmt) override {
    QD_ERROR("WhileStmt is not supported in AutoDiff.");
  }

  void visit(ContinueStmt *stmt) override {
    QD_ERROR("ContinueStmt is not supported in AutoDiff.");
  }

  void visit(WhileControlStmt *stmt) override {
    QD_ERROR("WhileControlStmt (break) is not supported in AutoDiff.");
  }

  void visit_loop_body(Block *block) {
    auto ib_meta_data = IndependentBlockMetaData();
    // An IB has no local load/store to allocas *outside* itself. Note:
    //   - Local atomics must have been demoted before this pass.
    //   - It is OK for an IB to have more than two for loops.
    //   - No global load/atomics operations to global variables that require gradient.
    if (block->statements.empty()) {
      // A empty block shoud be a smallest IB
      ib_meta_data.is_ib = true;
      ib_meta_data.is_smallest_ib = true;
    } else {
      IndependentBlocksJudger::run(block, ib_meta_data);
    }

    if (ib_meta_data.is_smallest_ib) {
      independent_blocks_.push_back({depth_, block});
    } else if (ib_meta_data.is_ib) {
      current_ib_ = block;
      block->accept(this);
    } else {
      if (depth_ <= 1) {
        QD_ASSERT(depth_ == 1);
        // The top level block is already not an IB, store it
        independent_blocks_.push_back({depth_ - 1, block});
      } else {
        independent_blocks_.push_back({depth_ - 1, block->parent_block()});
      }
    }
  }

  void visit(StructForStmt *stmt) override {
    QD_ASSERT(depth_ == 0);
    depth_++;
    current_ib_ = stmt->body.get();
    visit_loop_body(stmt->body.get());
    depth_--;
  }

  void visit(RangeForStmt *stmt) override {
    if (depth_ == 0) {
      current_ib_ = stmt->body.get();
    }
    depth_++;
    visit_loop_body(stmt->body.get());
    depth_--;
  }

  static std::set<Block *> run(IRNode *root) {
    IdentifyIndependentBlocks pass;
    Block *block = root->as<Block>();
    bool has_for = false;
    for (auto &s : block->statements) {
      if (s->is<StructForStmt>() || s->is<RangeForStmt>()) {
        has_for = true;
      }
    }
    if (!has_for) {
      // The whole block is an IB
      pass.independent_blocks_.push_back({0, block});
    } else {
      root->accept(&pass);
    }
    // Sort the IBs by their depth from shallow to deep
    std::sort(
        pass.independent_blocks_.begin(), pass.independent_blocks_.end(),
        [](const std::pair<int, Block *> &a, const std::pair<int, Block *> &b) -> bool { return a.first < b.first; });

    QD_ASSERT(!pass.independent_blocks_.empty());
    return DuplicateIndependentBlocksCleaner::run(pass.independent_blocks_);
  }

 private:
  std::vector<std::pair<int, Block *>> independent_blocks_;
  int depth_{0};
  Block *current_ib_{nullptr};
};

// ============================================================================
// ReverseOuterLoops: flip iteration direction on outer (non-IB) for-loops and reorder sibling for-loops in non-IB
// container blocks so the reverse pass walks the iteration trace backward.
// ============================================================================

class ReverseOuterLoops : public BasicStmtVisitor {
  using BasicStmtVisitor::visit;

 private:
  explicit ReverseOuterLoops(const std::set<Block *> &IB) : loop_depth_(0), ib_(IB) {
  }

  bool is_ib(Block *block) const {
    return std::find(ib_.begin(), ib_.end(), block) != ib_.end();
  }

  // Sibling for-loops inside a non-IB container block execute their reverse-mode companions in the container's forward
  // order by default, because MakeAdjoint only touches IB-level bodies and nothing else permutes the enclosing order.
  // Reverse-mode AD requires the opposite: if the forward body runs `for_A; for_B` and for_B's reverse depends on reads
  // produced by for_A's forward run, the reverse pass must execute `rev-for_B; rev-for_A` so for_A's reverse sees the
  // adjoints for_B has populated (e.g. `a[i]=x[i]; b[i]=a[i]*y[i]` silently returns x.grad=0 otherwise: rev-for_A
  // clears a.grad before rev-for_B has populated it).
  //
  // Naive pairwise swap of for-loop positions is unsafe whenever a non-loop stmt between two for-loops feeds the later
  // sibling's SSA operand chain (e.g. a GlobalLoad that supplies a dynamic trip count): after the swap, the consumer
  // for-loop ends up before its producer and the IR verifier rejects the block. Before swapping, hoist any such
  // producer (and its transitive in-block dependencies) to the slot just before the first sibling for-loop. Non-loop
  // stmts unrelated to for-loop operands stay at their original indices; memory ordering between non-loop stmts is
  // preserved because the hoist keeps them in their original relative order and only moves them upward over for-loops
  // (which produce no SSA value and cannot be the source of a missed memory read for a non-loop that gets hoisted above
  // them).
  //
  // The top-level kernel block is handled by `reverse_segments` before this pass, so we only reorder inside nested
  // non-IB blocks here.
  static void reverse_for_loop_order_in_place(Block *block) {
    const int n = (int)block->statements.size();
    std::vector<int> for_indices;
    for (int i = 0; i < n; ++i) {
      Stmt *s = block->statements[i].get();
      if (s->is<RangeForStmt>() || s->is<StructForStmt>()) {
        for_indices.push_back(i);
      }
    }
    if (for_indices.size() < 2) {
      return;
    }
    const int first_for = for_indices.front();

    std::unordered_map<Stmt *, int> pos_of;
    pos_of.reserve(n);
    for (int i = 0; i < n; ++i) {
      pos_of[block->statements[i].get()] = i;
    }

    // Walk the SSA operand graph of every for-loop (restricted to this block). Any in-block stmt that (a) the operand
    // closure reaches and (b) sits at or after `first_for` gets flagged for hoisting: after swap, that stmt must
    // precede every for-loop, not just the ones it feeds.
    std::unordered_set<Stmt *> must_hoist;
    std::vector<Stmt *> stack;
    auto push_if_internal = [&](Stmt *s) {
      if (s == nullptr) {
        return;
      }
      auto it = pos_of.find(s);
      if (it == pos_of.end() || it->second < first_for) {
        return;
      }
      if (must_hoist.insert(s).second) {
        stack.push_back(s);
      }
    };
    // Seed the hoist frontier from both the for-loop's direct SSA operands (`begin`, `end`) and from every stmt nested
    // inside the for-loop's body that references an outer-block stmt as a free variable. The body-use gather catches
    // the case where the later sibling for-loop consumes a non-loop outer-block stmt `S` inside its body (e.g. `for_B:
    // body reads S`) rather than through `for_B`'s range bound: `RangeForStmt::get_operands()` returns only `{begin,
    // end}`, so without walking the body `S` would miss `must_hoist`, the pairwise swap would place `for_B` ahead of
    // `S`, and the IR verifier would reject the SSA violation.
    for (int fi : for_indices) {
      for (Stmt *op : block->statements[fi]->get_operands()) {
        push_if_internal(op);
      }
      Stmt *for_stmt = block->statements[fi].get();
      irpass::analysis::gather_statements(for_stmt, [&](Stmt *body_stmt) {
        for (Stmt *op : body_stmt->get_operands()) {
          push_if_internal(op);
        }
        return false;
      });
    }
    while (!stack.empty()) {
      Stmt *s = stack.back();
      stack.pop_back();
      for (Stmt *op : s->get_operands()) {
        push_if_internal(op);
      }
    }
    // For-loops themselves end up in `must_hoist` only because their own operand-closure reached them; they do not get
    // hoisted as non-loop producers - strip them here to keep `must_hoist` to "non-loop stmts that need to move above
    // all for-loops".
    for (int fi : for_indices) {
      must_hoist.erase(block->statements[fi].get());
    }

    std::vector<std::unique_ptr<Stmt>> new_stmts;
    new_stmts.reserve(n);
    // Stmts strictly before `first_for` keep their original slot.
    for (int i = 0; i < first_for; ++i) {
      new_stmts.push_back(std::move(block->statements[i]));
    }
    // Hoisted non-loop stmts slot in here, in their original relative order.
    for (int i = first_for; i < n; ++i) {
      if (must_hoist.count(block->statements[i].get()) != 0) {
        new_stmts.push_back(std::move(block->statements[i]));
      }
    }
    // Remainder (for-loops and non-hoisted non-loops) in original order, with for-loops swapped pairwise inside this
    // suffix.
    std::vector<std::unique_ptr<Stmt>> suffix;
    std::vector<int> suffix_for_positions;
    for (int i = first_for; i < n; ++i) {
      auto &sp = block->statements[i];
      if (!sp) {
        continue;
      }
      bool is_for = sp->is<RangeForStmt>() || sp->is<StructForStmt>();
      if (is_for) {
        suffix_for_positions.push_back((int)suffix.size());
      }
      suffix.push_back(std::move(sp));
    }
    for (int lo = 0, hi = (int)suffix_for_positions.size() - 1; lo < hi; ++lo, --hi) {
      std::swap(suffix[suffix_for_positions[lo]], suffix[suffix_for_positions[hi]]);
    }
    for (auto &s : suffix) {
      new_stmts.push_back(std::move(s));
    }

    QD_ASSERT((int)new_stmts.size() == n);
    block->statements.clear();
    for (auto &s : new_stmts) {
      block->statements.push_back(std::move(s));
    }
  }

  void visit(StructForStmt *stmt) override {
    loop_depth_ += 1;
    if (!is_ib(stmt->body.get())) {
      stmt->body->accept(this);
      reverse_for_loop_order_in_place(stmt->body.get());
    }
    loop_depth_ -= 1;
  }

  void visit(RangeForStmt *stmt) override {
    if (loop_depth_ >= 1) {
      stmt->reversed = !stmt->reversed;
    }
    loop_depth_ += 1;
    if (!is_ib(stmt->body.get())) {
      stmt->body->accept(this);
      reverse_for_loop_order_in_place(stmt->body.get());
    }
    loop_depth_ -= 1;
  }

  // Deliberately no `visit(IfStmt *)` override, although sibling for-loops can live directly inside an if-branch block
  // (`true_statements` / `false_statements`) the same way they live inside a for-body. The default
  // `BasicStmtVisitor::visit(IfStmt *)` recurses into both branches so inner `RangeForStmt::body`s still get the
  // sibling-reorder treatment via the range-for visitor above, but `reverse_for_loop_order_in_place` is never invoked
  // on the branch block itself. That is intentional: `MakeAdjoint::visit(IfStmt *)` below emits the adjoint if-stmt by
  // iterating each branch's statements in reverse order (`for (int i = size - 1; i >= 0; --i)` in its `true_statements`
  // / `false_statements` loops), which achieves the same sibling-for reordering effect that the missing override here
  // would provide. Overriding `visit(IfStmt)` in this class is therefore a no-op on the generated adjoint code. Keep
  // the comment rather than the override so the visitor-coverage gap is documented.

  int loop_depth_;
  std::set<Block *> ib_;

 public:
  static void run(IRNode *root, const std::set<Block *> &IB) {
    ReverseOuterLoops pass(IB);
    root->accept(&pass);
  }
};

}  // namespace

void regulate_tensor_typed_statements(IRNode *root) {
  RegulateTensorTypedStatements::run(root);
}

std::set<Block *> identify_independent_blocks(IRNode *root) {
  return IdentifyIndependentBlocks::run(root);
}

void reverse_outer_loops(IRNode *root, const std::set<Block *> &IB) {
  ReverseOuterLoops::run(root, IB);
}

}  // namespace quadrants::lang
