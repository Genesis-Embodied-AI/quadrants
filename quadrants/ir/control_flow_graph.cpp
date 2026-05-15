#include "quadrants/ir/control_flow_graph.h"

#include <queue>
#include <unordered_set>
#include <fstream>
#include <filesystem>
#include <sstream>

#include "quadrants/common/exceptions.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/system/profiler.h"
#include "quadrants/program/function.h"
#include "quadrants/codegen/ir_dump.h"

namespace quadrants::lang {

namespace {

// Does |store_stmt| store to an address that may alias |var|?
// Matrix-pointer aliasing is handled both ways: a MatrixPtrStmt aliases its
// underlying origin, and vice-versa.
bool may_contain_address(Stmt *store_stmt, Stmt *var) {
  for (auto store_ptr : irpass::analysis::get_store_destination(store_stmt)) {
    if (var->is<MatrixPtrStmt>() && !store_ptr->is<MatrixPtrStmt>()) {
      // check for aliased address with var
      if (irpass::analysis::maybe_same_address(var->as<MatrixPtrStmt>()->origin, store_ptr)) {
        return true;
      }
    }

    if (!var->is<MatrixPtrStmt>() && store_ptr->is<MatrixPtrStmt>()) {
      // check for aliased address with store_ptr
      if (irpass::analysis::maybe_same_address(store_ptr->as<MatrixPtrStmt>()->origin, var)) {
        return true;
      }
    }

    if (irpass::analysis::maybe_same_address(var, store_ptr)) {
      return true;
    }
  }
  return false;
}

// Should |stmt| appear in the synthetic `live_gen` of the final CFG node?
// Locals (allocas, matrix-pointers into allocas) are never live past the
// kernel boundary. Global pointers are live unless SFG has marked their
// SNode as eliminable in |config_opt|.
bool in_final_node_live_gen(const Stmt *stmt,
                            const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &config_opt) {
  if (stmt->is<AllocaStmt>() || stmt->is<AdStackAllocaStmt>()) {
    return false;
  }
  if (stmt->is<MatrixPtrStmt>() && stmt->cast<MatrixPtrStmt>()->origin->is<AllocaStmt>()) {
    return false;
  }
  if (auto *gptr = stmt->cast<GlobalPtrStmt>(); gptr && config_opt.has_value()) {
    return config_opt->eliminable_snodes.count(gptr->snode) == 0;
  }
  // A global pointer that may be loaded after this kernel.
  return true;
}

}  // namespace

CFGNode::CFGNode(Block *block,
                 int begin_location,
                 int end_location,
                 bool is_parallel_executed,
                 CFGNode *prev_node_in_same_block)
    : block(block),
      begin_location(begin_location),
      end_location(end_location),
      is_parallel_executed(is_parallel_executed),
      prev_node_in_same_block(prev_node_in_same_block),
      next_node_in_same_block(nullptr) {
  if (prev_node_in_same_block != nullptr)
    prev_node_in_same_block->next_node_in_same_block = this;
  if (!empty()) {
    // For non-empty nodes, precompute |parent_blocks| to accelerate
    // get_store_forwarding_data().
    QD_ASSERT(begin_location >= 0);
    QD_ASSERT(block);
    auto parent_block = block;
    parent_blocks_.insert(parent_block);
    while (parent_block->parent_block()) {
      parent_block = parent_block->parent_block();
      parent_blocks_.insert(parent_block);
    }
  }
}

CFGNode::CFGNode() : CFGNode(nullptr, -1, -1, false, nullptr) {
}

void CFGNode::add_edge(CFGNode *from, CFGNode *to) {
  from->next.push_back(to);
  to->prev.push_back(from);
}

bool CFGNode::empty() const {
  return begin_location >= end_location;
}

std::size_t CFGNode::size() const {
  return end_location - begin_location;
}

void CFGNode::erase(int location) {
  QD_ASSERT(location >= begin_location && location < end_location);
  block->erase(location);
  end_location--;
  for (auto node = next_node_in_same_block; node != nullptr; node = node->next_node_in_same_block) {
    node->begin_location--;
    node->end_location--;
  }
}

void CFGNode::insert(std::unique_ptr<Stmt> &&new_stmt, int location) {
  QD_ASSERT(location >= begin_location && location <= end_location);
  block->insert(std::move(new_stmt), location);
  end_location++;
  for (auto node = next_node_in_same_block; node != nullptr; node = node->next_node_in_same_block) {
    node->begin_location++;
    node->end_location++;
  }
}

void CFGNode::replace_with(int location, std::unique_ptr<Stmt> &&new_stmt, bool replace_usages) const {
  QD_ASSERT(location >= begin_location && location < end_location);
  block->replace_with(block->statements[location].get(), std::move(new_stmt), replace_usages);
}

bool CFGNode::contain_variable(const std::unordered_set<Stmt *> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    for (Stmt *set_var : var_set) {
      if (irpass::analysis::definitely_same_address(var, set_var)) {
        return true;
      }
    }
    return false;
  }
}

bool CFGNode::contain_variable(const std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    if (var_set.find(var) != var_set.end()) {
      return var_set.at(var) != CFGNode::UseDefineStatus::PARTIAL;
    }
    return false;
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end()) {
      return var_set.at(var) != CFGNode::UseDefineStatus::PARTIAL;
    }
    for (const auto &[key, status] : var_set) {
      if (irpass::analysis::definitely_same_address(var, key)) {
        return status != CFGNode::UseDefineStatus::PARTIAL;
      }
    }
    return false;
  }
}

bool CFGNode::may_contain_variable(const std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    for (const auto &entry : var_set) {
      if (irpass::analysis::maybe_same_address(var, entry.first)) {
        return true;
      }
    }
    return false;
  }
}

bool CFGNode::may_contain_variable(const std::unordered_set<Stmt *> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    for (Stmt *set_var : var_set) {
      if (irpass::analysis::maybe_same_address(var, set_var)) {
        return true;
      }
    }
    return false;
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  return contain_variable(reach_kill, var);
}

bool CFGNode::is_visible_at(Stmt *stmt, int position) const {
  // Check if |stmt| is before |position| here.
  if (stmt->parent == block) {
    return stmt->parent->locate(stmt) < position;
  }
  // |parent_blocks_| is precomputed in the constructor of CFGNode.
  // TODO: What if |stmt| appears in an ancestor of |block| but after
  //  |position|?
  return parent_blocks_.find(stmt->parent) != parent_blocks_.end();
}

bool CFGNode::update_forwarding_result(Stmt *stmt,
                                       int position,
                                       Stmt *&result,
                                       bool &result_visible) const {
  // |stmt| is a definition in the UD-chain of the variable being forwarded.
  // Fold its stored data into |result| / |result_visible|. Return false if
  // forwarding must abort (the caller should propagate nullptr); true to
  // continue scanning.
  auto data = irpass::analysis::get_store_data(stmt);
  if (!data) {  // not forwardable
    return false;
  }
  if (!result) {
    result = data;
    result_visible = is_visible_at(data, position);
    return true;
  }
  if (!irpass::analysis::same_value(result, data)) {
    // check the special case of alloca (initialized to 0)
    if (!(result->is<AllocaStmt>() && data->is<ConstStmt>() && data->as<ConstStmt>()->val.equal_value(0))) {
      return false;
    }
  }
  if (!result_visible && is_visible_at(data, position)) {
    // pick the visible one for store-to-load forwarding
    result = data;
    result_visible = true;
  }
  return true;
}

int CFGNode::find_intra_block_last_def(Stmt *var, int position) const {
  for (int i = position - 1; i >= begin_location; i--) {
    // Find previous store stmt to the same dest_addr, stop at the closest one.
    // store_ptr: prev-store dest_addr
    for (auto *store_ptr : irpass::analysis::get_store_destination(block->statements[i].get())) {
      // Exclude `store_ptr` as a potential store destination due to mixed
      // semantics of store statements for quant types. The store operation
      // involves implicit casting before storing, which may result in a loss of
      // precision. For example:
      //   <i32> $3 = const 233333
      //   <*qi4> $4 = global ptr [S3place<qi4><bit>], index [$1] activate=false
      //   $5 : global store [$4 <- $3]
      //   <i32> $6 = global load $4
      // The store cannot be forwarded because $3 is first casted to a qi4 and
      // then stored into $4. Since 233333 won't fit into a qi4, the store leads
      // to truncation, resulting in a different value stored in $4 compared to
      // $3.
      // TODO: Still forward the store if the value can be statically proven to
      // fit into the quant type.
      if (!is_quant(store_ptr->ret_type.ptr_removed()) && irpass::analysis::definitely_same_address(var, store_ptr)) {
        return i;
      }

      // Special case:
      // $1 = store $0, MatrixInitStmt(...)
      // ...
      // $2 = matrix ptr $0, offset
      // $3 = load $2
      // We can forward MatrixInitStmt->values[offset] to $3
      if (var->is<MatrixPtrStmt>() && var->as<MatrixPtrStmt>()->offset->is<ConstStmt>()) {
        auto *var_origin = var->as<MatrixPtrStmt>()->origin;
        if (irpass::analysis::definitely_same_address(var_origin, store_ptr)) {
          Stmt *store_data = irpass::analysis::get_store_data(block->statements[i].get());
          if (store_data->is<MatrixInitStmt>()) {
            return i;
          }
        }
      }
    }
  }
  return -1;
}

std::optional<int> CFGNode::find_cross_block_def(Stmt *var,
                                                 int position,
                                                 Stmt *&result,
                                                 bool &result_visible) const {
  int last_def_position = -1;
  // [Global Addr only] Stores reaching the entry of this node from previous blocks. `var == stmt`
  // is for the case that a global ptr is never stored; in that case `stmt` comes from
  // nodes[start_node]->reach_gen.
  for (auto *stmt : reach_in) {
    if (var == stmt || may_contain_address(stmt, var)) {
      if (!update_forwarding_result(stmt, position, result, result_visible)) {
        return std::nullopt;
      }
      last_def_position = 0;
    }
  }
  // Stores generated within this node (in reach_gen) that precede |position|.
  for (auto *stmt : reach_gen) {
    if (may_contain_address(stmt, var) && stmt->parent->locate(stmt) < position) {
      if (!update_forwarding_result(stmt, position, result, result_visible)) {
        return std::nullopt;
      }
      last_def_position = stmt->parent->locate(stmt);
    }
  }
  return last_def_position;
}

bool CFGNode::any_aliased_store_breaks_forwarding(Stmt *result, Stmt *var, int from, int to_exclusive) const {
  // Allocas without tensor type cannot be aliased through MatrixPtrStmt, so the check is moot.
  const bool is_tensor_involved = var->ret_type.ptr_removed()->is<TensorType>();
  if (var->is<AllocaStmt>() && !is_tensor_involved) {
    return false;
  }
  for (int i = from; i < to_exclusive; i++) {
    auto *s = block->statements[i].get();
    if (!irpass::analysis::same_value(result, irpass::analysis::get_store_data(s))) {
      if (may_contain_address(s, var)) {
        return true;
      }
    }
  }
  return false;
}

// var: dest_addr. Return the stored data if all definitions in the UD-chain of |var| at this
// position store the same data; otherwise nullptr. Looks intra-block first (cheaper, dominates
// cross-block forwarding when present), then falls back to the cross-block search over reach_in
// and reach_gen. In both cases an intervening aliased store that may write a different value
// breaks the forward.
Stmt *CFGNode::get_store_forwarding_data(Stmt *var, int position) const {
  // [Intra-block search] Walks backwards in this node's block.
  if (int last_def = find_intra_block_last_def(var, position); last_def != -1) {
    Stmt *result = irpass::analysis::get_store_data(block->statements[last_def].get());
    if (any_aliased_store_breaks_forwarding(result, var, last_def + 1, position)) {
      return nullptr;
    }
    return result;
  }

  // [Cross-block search] Walks reach_in / reach_gen, accumulating a single forwardable result.
  Stmt *result = nullptr;
  bool result_visible = false;
  std::optional<int> last_def = find_cross_block_def(var, position, result, result_visible);
  if (!last_def.has_value()) {
    return nullptr;
  }
  if (!result) {
    // The UD-chain is empty.
    auto *offending_load = block->statements[position].get();
    ErrorEmitter(QuadrantsIrWarning(), offending_load,
                 fmt::format("Loading variable {} before anything is stored to it.", var->id));
    return nullptr;
  }
  if (!result_visible) {
    // The data is store-to-load forwardable but not visible at the place we are going to forward.
    return nullptr;
  }
  if (*last_def == -1) {
    return nullptr;
  }
  if (any_aliased_store_breaks_forwarding(result, var, *last_def, position)) {
    return nullptr;
  }
  return result;
}

void CFGNode::reaching_definition_analysis(bool after_lower_access) {
  // Calculate |reach_gen| and |reach_kill|.
  reach_gen.clear();
  reach_kill.clear();
  for (int i = end_location - 1; i >= begin_location; i--) {
    // loop in reversed order
    auto stmt = block->statements[i].get();
    auto data_source_ptrs = irpass::analysis::get_store_destination(stmt);
    for (auto data_source_ptr : data_source_ptrs) {
      // stmt provides a data source
      if (after_lower_access &&
          !((data_source_ptr->is<MatrixPtrStmt>() && data_source_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
            (data_source_ptr->is<MatrixPtrStmt>() &&
             data_source_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
            data_source_ptr->is<AllocaStmt>())) {
        // After lower_access, we only analyze local variables.
        continue;
      }
      if (!reach_kill_variable(data_source_ptr)) {
        reach_gen.insert(stmt);
        reach_kill.insert(data_source_ptr);
      }
    }
  }
}

bool CFGNode::try_forward_load_at(int &i, Stmt *stmt, bool after_lower_access, bool autodiff_enabled, bool &modified) {
  // Find the closest preceding store-to-same-address. For GlobalLoadStmt the forwarder is gated:
  // after lower_access or under autodiff we don't trust cross-block forwarding to be sound.
  Stmt *load_src = nullptr;
  Stmt *result = nullptr;
  if (auto *local_load = stmt->cast<LocalLoadStmt>()) {
    load_src = local_load->src;
    result = get_store_forwarding_data(load_src, i);
  } else if (auto *global_load = stmt->cast<GlobalLoadStmt>()) {
    if (!after_lower_access && !autodiff_enabled) {
      load_src = global_load->src;
      result = get_store_forwarding_data(load_src, i);
    }
  }
  if (!result) {
    return false;
  }
  if (result->is<AllocaStmt>()) {
    // TensorType does not apply to this special case; skip further handling for this stmt.
    if (result->ret_type.ptr_removed()->is<TensorType>()) {
      return true;
    }
    // Alloca initialized to 0: replace the load with a zero const.
    // Note: |modified| is intentionally NOT flipped here (preserved from legacy behavior).
    auto zero = Stmt::make<ConstStmt>(TypedConstant(result->ret_type.ptr_removed(), 0));
    replace_with(i, std::move(zero), true);
    return true;
  }
  // Non-alloca result: forward it. Extract a MatrixInitStmt element when the forwarded data is
  // a TensorType-typed init but the load is for a scalar slot.
  if (result->ret_type.ptr_removed()->is<TensorType>() && !stmt->ret_type->is<TensorType>()) {
    QD_ASSERT(load_src->is<MatrixPtrStmt>() && load_src->as<MatrixPtrStmt>()->offset->is<ConstStmt>());
    QD_ASSERT(result->is<MatrixInitStmt>());
    const int offset = load_src->as<MatrixPtrStmt>()->offset->as<ConstStmt>()->val.val_int32();
    result = result->as<MatrixInitStmt>()->values[offset];
  }
  stmt->replace_usages_with(result);
  erase(i);  // end_location--
  i--;       // cancel the for-loop's i++
  modified = true;
  return true;
}

void CFGNode::try_eliminate_identical_store_at(int &i,
                                               Stmt *stmt,
                                               bool after_lower_access,
                                               bool autodiff_enabled,
                                               bool &modified) {
  // Eliminate a store whose value is provably identical to the closest preceding store to the
  // same address. For local stores under non-autodiff there's also an alloca-zero special case:
  // writing a zero to a freshly-allocated alloca is redundant.
  if (auto *local_store = stmt->cast<LocalStoreStmt>()) {
    Stmt *result = get_store_forwarding_data(local_store->dest, i);
    if (result && result->is<AllocaStmt>() && !autodiff_enabled) {
      // TensorType does not apply to this special case.
      if (result->ret_type.ptr_removed()->is<TensorType>()) {
        return;
      }
      if (auto *stored = local_store->val->cast<ConstStmt>()) {
        if (stored->val.equal_value(0)) {
          erase(i);
          i--;
          modified = true;
        }
      }
      return;
    }
    if (irpass::analysis::same_value(result, local_store->val)) {
      erase(i);
      i--;
      modified = true;
    }
    return;
  }
  if (auto *global_store = stmt->cast<GlobalStoreStmt>()) {
    if (after_lower_access) {
      return;
    }
    Stmt *result = get_store_forwarding_data(global_store->dest, i);
    if (irpass::analysis::same_value(result, global_store->val)) {
      erase(i);
      i--;
      modified = true;
    }
  }
}

bool CFGNode::store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled) {
  // Two passes fused into one forward walk:
  //   1. Store-to-load forwarding: replace a load with the value from the closest preceding store
  //      to the same address.
  //   2. Identical-store elimination: erase a store whose value is identical to the closest
  //      preceding store to the same address.
  // The forwarding pass takes precedence: if we forwarded a load, we don't also try to treat the
  // same stmt as a store. erase() shrinks end_location, so per-elimination we both `erase(i)` and
  // decrement `i` to keep the for-loop pointing at the right next stmt.
  bool modified = false;
  for (int i = begin_location; i < end_location; i++) {
    auto *stmt = block->statements[i].get();
    if (try_forward_load_at(i, stmt, after_lower_access, autodiff_enabled, modified)) {
      continue;
    }
    try_eliminate_identical_store_at(i, stmt, after_lower_access, autodiff_enabled, modified);
  }
  return modified;
}

void CFGNode::gather_loaded_snodes(std::unordered_set<SNode *> &snodes) const {
  // Gather the SNodes which this CFGNode loads.
  // Requires reaching definition analysis.
  std::unordered_set<Stmt *> killed_in_this_node;
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    for (auto &load_ptr : load_ptrs) {
      if (auto global_ptr = load_ptr->cast<GlobalPtrStmt>()) {
        // Avoid computing the UD-chain if every SNode in this global ptr
        // are already loaded because it can be time-consuming.
        auto snode = global_ptr->snode;
        if (snodes.count(snode) > 0) {
          continue;
        }
        if (reach_in.find(global_ptr) != reach_in.end() && !contain_variable(killed_in_this_node, global_ptr)) {
          // The UD-chain contains the value before this offloaded task.
          snodes.insert(snode);
        }
      }
    }
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);
    for (auto &store_ptr : store_ptrs) {
      if (store_ptr->is<GlobalPtrStmt>()) {
        killed_in_this_node.insert(store_ptr);
      }
    }
  }
}

void CFGNode::live_variable_analysis(bool after_lower_access) {
  live_gen.clear();
  live_kill.clear();
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();

    // If stmt is a MatrixPtrStmt, the load only partially uses the original
    // address. Since MatrixPtrStmt relies on the original address, we need to
    // gen the aliased orginal address as well.
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt, true /*get_alias*/);
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        if (!contain_variable(live_kill, load_ptr)) {
          live_gen.insert(load_ptr);
        }
      }
    }

    // If stmt is a MatrixPtrStmt, the store only partially defines the original
    // address. So it's not safe to fully kill the aliased original address
    // here.
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);
    // TODO: Consider AD-stacks in get_store_destination instead of here
    //  for store-to-load forwarding on AD-stacks
    // TODO: SNode deactivation is also a definite store
    if (auto stack_pop = stmt->cast<AdStackPopStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_pop->stack);
    } else if (auto stack_push = stmt->cast<AdStackPushStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_push->stack);
    } else if (auto stack_acc_adj = stmt->cast<AdStackAccAdjointStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_acc_adj->stack);
    }
    for (auto store_ptr : store_ptrs) {
      if (!after_lower_access ||
          (store_ptr->is<MatrixPtrStmt>() && store_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (store_ptr->is<MatrixPtrStmt>() && store_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<AdStackAllocaStmt>())) {
        // After lower_access, we only analyze local variables and stacks.
        live_kill.insert(store_ptr);
      }
    }
  }
}

static void recursive_update_aliased_elements(
    const std::unordered_map<Stmt *, std::vector<Stmt *>> &tensor_to_matrix_ptrs_map,
    std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &container,
    Stmt *key,
    bool to_erase) {
  if (tensor_to_matrix_ptrs_map.find(key) != tensor_to_matrix_ptrs_map.end()) {
    const auto &elements_address = tensor_to_matrix_ptrs_map.at(key);
    // Update aliased MatrixPtrStmt for TensorType<>*
    for (const auto &element_address : elements_address) {
      if (to_erase) {
        if (container.find(element_address) != container.end()) {
          container.erase(element_address);
        }
      } else {
        container[element_address] = CFGNode::UseDefineStatus::NONE;
        if (element_address->ret_type.ptr_removed()->is<TensorType>()) {
          container[element_address] = CFGNode::UseDefineStatus::FULL;
        }
      }

      // Recursively update aliased addresses
      recursive_update_aliased_elements(tensor_to_matrix_ptrs_map, container, element_address, to_erase);
    }
  }
}

static void recursive_update_aliased_parent(const std::unordered_map<Stmt *, Stmt *> &matrix_ptr_to_tensor_map,
                                            std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &container,
                                            Stmt *key,
                                            bool to_erase) {
  if (matrix_ptr_to_tensor_map.find(key) != matrix_ptr_to_tensor_map.end()) {
    const auto &tensor_address = matrix_ptr_to_tensor_map.at(key);
    // no matter to_erase or not, the tensor_address is only partially defined
    // or used
    if (to_erase) {
      if (container.find(tensor_address) != container.end()) {
        container[tensor_address] = CFGNode::UseDefineStatus::PARTIAL;
      }
    } else {
      container[tensor_address] = CFGNode::UseDefineStatus::PARTIAL;
    }

    // Recursively update aliased addresses
    recursive_update_aliased_parent(matrix_ptr_to_tensor_map, container, tensor_address, to_erase);
  }
}

static void update_aliased_stmts(const std::unordered_map<Stmt *, std::vector<Stmt *>> &tensor_to_matrix_ptrs_map,
                                 const std::unordered_map<Stmt *, Stmt *> &matrix_ptr_to_tensor_map,
                                 std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &container,
                                 Stmt *key,
                                 bool to_erase) {
  // Update aliased MatrixPtrStmt for TensorType<>*
  recursive_update_aliased_elements(tensor_to_matrix_ptrs_map, container, key, to_erase);

  // Update aliased TensorType<>* for MatrixPtrStmt
  recursive_update_aliased_parent(matrix_ptr_to_tensor_map, container, key, to_erase);
}

// Insert or erase "key" to "container".
// In case where "key" being MatrixPtrStmt, we also update the aliased original
// address. In case where "key" is involved with TensorType, we also update the
// alised MatrixPtrStmt
//
// CFGNode::UseDefineStatus is used to mark whether a TensorType'd address
// is fully or partially modified.
static void update_container_with_alias(
    const std::unordered_map<Stmt *, std::vector<Stmt *>> &tensor_to_matrix_ptrs_map,
    const std::unordered_map<Stmt *, Stmt *> &matrix_ptr_to_tensor_map,
    std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &container,
    Stmt *key,
    bool to_erase) {
  if (to_erase) {
    container.erase(key);
  } else if (key->ret_type.ptr_removed()->is<TensorType>()) {
    container[key] = CFGNode::UseDefineStatus::FULL;
  } else {
    container[key] = CFGNode::UseDefineStatus::NONE;
  }
  // Recursively update aliased addresses
  update_aliased_stmts(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, container, key, to_erase);
}

namespace {

// === Helpers for CFGNode::dead_store_elimination ===

// Aliasing tables built once per CFGNode pass over the block, then threaded through every
// container update so that touching a `MatrixPtrStmt` propagates to its tensor origin (and back).
struct DseAliasMaps {
  // MatrixPtrStmt->origin -> list of MatrixPtrStmts that share that origin
  std::unordered_map<Stmt *, std::vector<Stmt *>> tensor_to_matrix_ptrs;
  // MatrixPtrStmt -> its origin
  std::unordered_map<Stmt *, Stmt *> matrix_ptr_to_tensor;
};

// Per-pass state mutated in lockstep during the reverse-order walk. `live_in_this_node` and
// `killed_in_this_node` use `UseDefineStatus` to distinguish full vs partial modification of
// tensor-typed addresses; `live_load_in_this_node` maps a variable to its nearest later load
// for identical-load elimination.
struct DseLiveState {
  std::unordered_map<Stmt *, Stmt *> live_load_in_this_node;
  std::unordered_map<Stmt *, CFGNode::UseDefineStatus> live_in_this_node;
  std::unordered_map<Stmt *, CFGNode::UseDefineStatus> killed_in_this_node;
};

DseAliasMaps build_matrix_ptr_alias_maps(Block *block, int begin_location, int end_location) {
  DseAliasMaps alias;
  for (int i = begin_location; i < end_location; i++) {
    auto *stmt = block->statements[i].get();
    if (!stmt->is<MatrixPtrStmt>()) {
      continue;
    }
    auto *origin = stmt->as<MatrixPtrStmt>()->origin;
    alias.tensor_to_matrix_ptrs[origin].push_back(stmt);
    alias.matrix_ptr_to_tensor[stmt] = origin;
  }
  return alias;
}

// Pointer eligibility predicate, applied uniformly to store ptrs, load ptrs, and live-update load
// ptrs. After `lower_access`, only local variables (allocas) and AD-stacks remain analyzable;
// before it, everything is in scope.
bool is_dse_eligible_pointer(Stmt *ptr, bool after_lower_access) {
  if (!after_lower_access) {
    return true;
  }
  if (ptr->is<AllocaStmt>() || ptr->is<AdStackAllocaStmt>()) {
    return true;
  }
  if (ptr->is<MatrixPtrStmt>()) {
    auto *origin = ptr->as<MatrixPtrStmt>()->origin;
    if (origin->is<AllocaStmt>() || origin->is<MatrixPtrStmt>()) {
      return true;
    }
  }
  return false;
}

// Compute the store destinations of |stmt|, including AD-stack stmts whose store semantics
// aren't yet captured by `get_store_destination`. Returns the same `stmt_refs` (= one_or_more<>)
// type the analyzer uses so iteration and size queries are uniform.
// TODO: Consider AD-stacks in get_store_destination instead of here for store-to-load forwarding
// on AD-stacks.
stmt_refs dse_store_destinations(Stmt *stmt) {
  if (auto *pop = stmt->cast<AdStackPopStmt>()) {
    return stmt_refs(pop->stack);
  }
  if (auto *push = stmt->cast<AdStackPushStmt>()) {
    return stmt_refs(push->stack);
  }
  if (auto *acc = stmt->cast<AdStackAccAdjointStmt>()) {
    return stmt_refs(acc->stack);
  }
  if (stmt->is<AdStackAllocaStmt>()) {
    return stmt_refs(stmt);
  }
  return irpass::analysis::get_store_destination(stmt);
}

// Is |store_ptr| guaranteed dead at this point in the reverse-order walk?
//   - !may_contain_variable(live_in_this_node, store_ptr): not loaded after this store in-node
//   - contain_variable(killed_in_this_node, store_ptr): already overwritten in-node, OR
//   - !may_contain_variable(live_out, store_ptr): not used in any successor node
bool is_store_dead(Stmt *store_ptr,
                   const std::unordered_set<Stmt *> &live_out,
                   const DseLiveState &state) {
  bool is_used_in_next_nodes = false;
  for (auto *ptr : irpass::analysis::include_aliased_stmts(store_ptr)) {
    is_used_in_next_nodes |= CFGNode::may_contain_variable(live_out, ptr);
  }
  const bool is_killed_in_current_node = CFGNode::contain_variable(state.killed_in_this_node, store_ptr);
  bool is_dead = is_killed_in_current_node || !is_used_in_next_nodes;
  is_dead &= !CFGNode::may_contain_variable(state.live_in_this_node, store_ptr);
  return is_dead;
}

// On dead-store elimination of an `AtomicOpStmt`, the store part is dropped but the load
// (= the atomic's return value) remains; record the load and mark the dest killed/loaded.
void record_weakened_atomic_as_load(Stmt *dest, Stmt *new_load, const DseAliasMaps &alias, DseLiveState &state) {
  update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.live_in_this_node, dest,
                              false);
  update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.killed_in_this_node, dest,
                              true);
  state.live_load_in_this_node[dest] = new_load;
}

// Try to eliminate a dead store (or weaken a dead-store atomic to a pure load) at position |i|.
// If no elimination applies, fall through to update killed/live state for this non-eliminated
// store and return false; the caller will then move on to load handling for the same stmt.
bool try_eliminate_dead_store_at(CFGNode *node,
                                 int i,
                                 Stmt *stmt,
                                 Stmt *store_ptr,
                                 bool after_lower_access,
                                 const DseAliasMaps &alias,
                                 DseLiveState &state) {
  if (!is_dse_eligible_pointer(store_ptr, after_lower_access)) {
    return false;
  }
  const bool is_dead = is_store_dead(store_ptr, node->live_out, state);
  const bool stmt_eliminable =
      !stmt->is<AllocaStmt>() && !stmt->is<AdStackAllocaStmt>() && !stmt->is<ExternalFuncCallStmt>();
  if (stmt_eliminable && is_dead) {
    // If an address is neither used in this node, nor in the next nodes, eliminate any stores to
    // it. Two scenarios:
    //   1. Direct store stmts (LocalStore, GlobalStore, AdStackPush, ...): erase outright.
    //   2. AtomicOpStmt (load+store): drop the store half, leaving a pure load.
    if (!stmt->is<AtomicOpStmt>()) {
      node->erase(i);
      return true;
    }
    auto *atomic = stmt->cast<AtomicOpStmt>();
    if (atomic->dest->is<AllocaStmt>()) {
      auto local_load = Stmt::make<LocalLoadStmt>(atomic->dest);
      local_load->ret_type = atomic->ret_type;
      record_weakened_atomic_as_load(atomic->dest, local_load.get(), alias, state);
      node->replace_with(i, std::move(local_load), true);
      return true;
    }
    // If this node is parallel executed, we can't weaken a global atomic to a global load.
    // TODO: we can weaken it if it's element-wise (i.e. never accessed by other threads).
    const bool atomic_global_weakable =
        !node->is_parallel_executed ||
        (atomic->dest->is<GlobalPtrStmt>() && atomic->dest->as<GlobalPtrStmt>()->snode->is_scalar());
    if (atomic_global_weakable) {
      auto global_load = Stmt::make<GlobalLoadStmt>(atomic->dest);
      global_load->ret_type = atomic->ret_type;
      record_weakened_atomic_as_load(atomic->dest, global_load.get(), alias, state);
      node->replace_with(i, std::move(global_load), true);
      return true;
    }
    // Atomic was dead but not safely weakable (parallel global, non-scalar). The original code
    // intentionally leaves state alone in this case (the atomic still executes, but state is not
    // updated for it). Preserve that behavior.
    return false;
  }
  // Non-eliminated store (not dead, or stmt not eliminable): update state. Insert into killed,
  // and drop any live-in entry that aliases the same address.
  update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.killed_in_this_node,
                              store_ptr, false);
  auto old_live_in = state.live_in_this_node;
  for (auto &var : old_live_in) {
    if (irpass::analysis::definitely_same_address(store_ptr, var.first)) {
      update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.live_in_this_node,
                                  store_ptr, true);
    }
  }
  return false;
}

// Try to eliminate an identical (redundant) load at |stmt|. We only do this within a single
// CFGNode, and only if no store has intervened since the prior load. The prior load (later in the
// block, since we walk in reverse) is the one erased; |stmt| takes over its usages.
bool try_eliminate_identical_load_at(CFGNode *node,
                                     Stmt *stmt,
                                     Stmt *load_ptr,
                                     bool after_lower_access,
                                     const DseAliasMaps &alias,
                                     DseLiveState &state) {
  if (!is_dse_eligible_pointer(load_ptr, after_lower_access)) {
    return false;
  }
  bool modified = false;
  auto it = state.live_load_in_this_node.find(load_ptr);
  if (it != state.live_load_in_this_node.end() &&
      !CFGNode::may_contain_variable(state.killed_in_this_node, load_ptr)) {
    auto *next_load_stmt = it->second;
    if (irpass::analysis::same_statements(stmt, next_load_stmt)) {
      next_load_stmt->replace_usages_with(stmt);
      node->erase(node->block->locate(next_load_stmt));
      modified = true;
    }
  }
  update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.killed_in_this_node,
                              load_ptr, true);
  state.live_load_in_this_node[load_ptr] = stmt;
  return modified;
}

// Mark all (eligible) loads of |stmt| as live-in-this-node so that earlier-in-reverse-order stores
// to the same address see them and abort their dead-store check. Taken by non-const reference
// because `one_or_more::begin()` is non-const.
void mark_loads_live_in_this_node(stmt_refs &load_ptrs,
                                  bool after_lower_access,
                                  const DseAliasMaps &alias,
                                  DseLiveState &state) {
  for (auto *load_ptr : load_ptrs) {
    if (is_dse_eligible_pointer(load_ptr, after_lower_access)) {
      update_container_with_alias(alias.tensor_to_matrix_ptrs, alias.matrix_ptr_to_tensor, state.live_in_this_node,
                                  load_ptr, false);
    }
  }
}

}  // namespace

bool CFGNode::dead_store_elimination(bool after_lower_access) {
  // Reverse-order walk over this node's statements. At each statement we may (a) eliminate a dead
  // store, (b) eliminate an identical load, or (c) update live/killed state for later iterations.
  // `FuncCallStmt` is treated as a full barrier: it can read/write anything, so we drop the kill
  // and live-load tables and start fresh.
  const DseAliasMaps alias = build_matrix_ptr_alias_maps(block, begin_location, end_location);
  DseLiveState state;
  bool modified = false;
  for (int i = end_location - 1; i >= begin_location; i--) {
    auto *stmt = block->statements[i].get();
    if (stmt->is<FuncCallStmt>()) {
      state.killed_in_this_node.clear();
      state.live_load_in_this_node.clear();
      continue;
    }
    auto store_ptrs = dse_store_destinations(stmt);
    if (store_ptrs.size() == 1) {
      if (try_eliminate_dead_store_at(this, i, stmt, *store_ptrs.begin(), after_lower_access, alias, state)) {
        modified = true;
        continue;
      }
    }
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    if (load_ptrs.size() == 1 && store_ptrs.empty()) {
      if (try_eliminate_identical_load_at(this, stmt, *load_ptrs.begin(), after_lower_access, alias, state)) {
        modified = true;
      }
    }
    mark_loads_live_in_this_node(load_ptrs, after_lower_access, alias, state);
  }
  return modified;
}

void ControlFlowGraph::erase(int node_id) {
  // Erase an empty node.
  QD_ASSERT(node_id >= 0 && node_id < (int)size());
  QD_ASSERT(nodes[node_id] && nodes[node_id]->empty());
  if (nodes[node_id]->prev_node_in_same_block) {
    nodes[node_id]->prev_node_in_same_block->next_node_in_same_block = nodes[node_id]->next_node_in_same_block;
  }
  if (nodes[node_id]->next_node_in_same_block) {
    nodes[node_id]->next_node_in_same_block->prev_node_in_same_block = nodes[node_id]->prev_node_in_same_block;
  }
  for (auto &prev_node : nodes[node_id]->prev) {
    prev_node->next.erase(std::find(prev_node->next.begin(), prev_node->next.end(), nodes[node_id].get()));
  }
  for (auto &next_node : nodes[node_id]->next) {
    next_node->prev.erase(std::find(next_node->prev.begin(), next_node->prev.end(), nodes[node_id].get()));
  }
  for (auto &prev_node : nodes[node_id]->prev) {
    for (auto &next_node : nodes[node_id]->next) {
      CFGNode::add_edge(prev_node, next_node);
    }
  }
  nodes[node_id].reset();
}

std::size_t ControlFlowGraph::size() const {
  return nodes.size();
}

CFGNode *ControlFlowGraph::back() const {
  return nodes.back().get();
}

namespace {

// === Helpers for ControlFlowGraph::dump_graph_to_file ===

// Range label: "empty" or "<first_stmt_name>~<last_stmt_name> (size=N)".
std::string format_cfg_node_range_label(const CFGNode *node) {
  if (node->empty()) {
    return "empty";
  }
  return fmt::format("{}~{} (size={})", node->block->statements[node->begin_location]->name(),
                     node->block->statements[node->end_location - 1]->name(), node->size());
}

// Brace-delimited list of neighbor indices, e.g. "{0, 1, 2}".
std::string format_neighbor_indices(const std::vector<CFGNode *> &neighbors,
                                    const std::unordered_map<CFGNode *, int> &to_index) {
  std::vector<std::string> parts;
  parts.reserve(neighbors.size());
  for (auto *n : neighbors) {
    parts.push_back(std::to_string(to_index.at(n)));
  }
  return fmt::format("{{{}}}", fmt::join(parts, ", "));
}

// Brace-delimited list of stmt names, e.g. "{$3, $5}".
std::string format_live_var_names(const std::unordered_set<Stmt *> &vars) {
  std::vector<std::string> parts;
  parts.reserve(vars.size());
  for (auto *stmt : vars) {
    parts.push_back(stmt->name());
  }
  return fmt::format("{{{}}}", fmt::join(parts, ", "));
}

// Write one node's header line: index, range label, optional prev/next/live_out lists.
void write_cfg_node_header(std::ostream &out,
                           int index,
                           const CFGNode *node,
                           const std::unordered_map<CFGNode *, int> &to_index) {
  out << fmt::format("Node {} : ", index) << format_cfg_node_range_label(node);
  if (!node->prev.empty()) {
    out << "; prev=" << format_neighbor_indices(node->prev, to_index);
  }
  if (!node->next.empty()) {
    out << "; next=" << format_neighbor_indices(node->next, to_index);
  }
  if (!node->live_out.empty()) {
    out << "; live_out=" << format_live_var_names(node->live_out);
  }
  out << "\n";
}

// Write one node's statements, each line indented 4 spaces, blank line after.
void write_cfg_node_statements(std::ostream &out, const CFGNode *node) {
  if (node->empty()) {
    return;
  }
  for (int j = node->begin_location; j < node->end_location; j++) {
    auto *stmt = node->block->statements[j].get();
    std::string stmt_output;
    // print_kernel_wrapper=false to avoid the surrounding "kernel { }" wrapper.
    irpass::print(stmt, &stmt_output, false, false);
    std::istringstream iss(stmt_output);
    std::string line;
    while (std::getline(iss, line)) {
      if (!line.empty()) {
        out << "    " << line << "\n";
      }
    }
  }
  out << "\n";
}

}  // namespace

void ControlFlowGraph::dump_graph_to_file(const CompileConfig &config,
                                          const std::string &kernel_name,
                                          const std::string &suffix) const {
  const std::filesystem::path ir_dump_dir = config.debug_dump_path;
  std::filesystem::create_directories(ir_dump_dir);
  const std::filesystem::path filename = ir_dump_dir / (kernel_name + "_CFG" + suffix + ".txt");

  std::ofstream out_file(filename.string());
  if (!out_file) {
    QD_WARN("Failed to open file for CFG dump: {}", filename.string());
    return;
  }

  const int num_nodes = size();
  std::unordered_map<CFGNode *, int> to_index;
  to_index.reserve(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    to_index[nodes[i].get()] = i;
  }

  for (int i = 0; i < num_nodes; i++) {
    write_cfg_node_header(out_file, i, nodes[i].get(), to_index);
    write_cfg_node_statements(out_file, nodes[i].get());
  }

  out_file.close();
  QD_INFO("CFG dumped to: {}", filename.string());
}

namespace {

// === Helpers for ControlFlowGraph::reaching_definition_analysis ===

// Statements that carry data into this kernel from outside its CFG. Treated as definitions in the
// synthetic entry node's reach_gen so cross-block forwarding correctly sees them. After access
// lowering, only MatrixPtrStmt aliases of allocas/matrix-pointers remain in scope; before, the
// full menagerie of pointer flavors applies.
// TODO: unify them.
bool is_external_input_pointer(const Stmt *stmt, bool after_lower_access) {
  if (stmt->is<MatrixPtrStmt>()) {
    const auto *origin = stmt->as<MatrixPtrStmt>()->origin;
    if (origin->is<AllocaStmt>() || origin->is<MatrixPtrStmt>()) {
      return true;
    }
  }
  if (after_lower_access) {
    return false;
  }
  return stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() || stmt->is<BlockLocalPtrStmt>() ||
         stmt->is<ThreadLocalPtrStmt>() || stmt->is<GlobalTemporaryStmt>() || stmt->is<MatrixPtrStmt>() ||
         stmt->is<GetChStmt>() || stmt->is<MatrixOfGlobalPtrStmt>() || stmt->is<MatrixOfMatrixPtrStmt>();
}

// Seed the synthetic entry node's reach_gen with definitions that "exist" before this kernel
// begins executing: external pointer loads (so a load before the first store still has a
// UD-chain) plus FuncCallStmt store destinations (the callee may have written anything declared
// in its `store_dests`).
void seed_start_node_reach_gen(CFGNode *start_node,
                               const std::vector<std::unique_ptr<CFGNode>> &nodes,
                               bool after_lower_access) {
  start_node->reach_gen.clear();
  start_node->reach_kill.clear();
  for (const auto &node : nodes) {
    for (int j = node->begin_location; j < node->end_location; j++) {
      auto *stmt = node->block->statements[j].get();
      if (is_external_input_pointer(stmt, after_lower_access)) {
        start_node->reach_gen.insert(stmt);
      } else if (auto *func_call = stmt->cast<FuncCallStmt>()) {
        const auto &dests = func_call->func->store_dests;
        start_node->reach_gen.insert(dests.begin(), dests.end());
      }
    }
  }
}

// Is |stmt| killed at |node| (i.e., excluded from reach_out)? For stmts with explicit store
// destinations, killed iff every dest is killed at |node|. For stmts without dests (e.g. raw
// global pointers seeded into start_node's reach_gen), killed iff the stmt itself is killed.
bool is_reach_in_stmt_killed_at(CFGNode *node, Stmt *stmt) {
  // Not const: `one_or_more::begin()` is non-const, so the range-for below would not compile.
  auto store_ptrs = irpass::analysis::get_store_destination(stmt);
  if (store_ptrs.empty()) {
    return node->reach_kill_variable(stmt);
  }
  for (auto *store_ptr : store_ptrs) {
    if (!node->reach_kill_variable(store_ptr)) {
      return false;
    }
  }
  return true;
}

}  // namespace

void ControlFlowGraph::reaching_definition_analysis(bool after_lower_access) {
  // Prerequisite analysis for load-store-forwarding; computes the cross-block use-define chain.
  //
  // Per-node:
  //   - reach_gen:  store stmts that define a variable in this node.
  //   - reach_kill: addresses (GlobalPtrStmt, AllocaStmt, ...) stored to in this node.
  // (reach_gen tracks the defining stmts; reach_kill tracks the killed addresses.)
  //
  // Per-graph (worklist fixpoint):
  //   - reach_in:  union of reach_out of all predecessor nodes.
  //   - reach_out: reach_gen + { stmts from reach_in whose dest is not in reach_kill }.
  QD_AUTO_PROF;
  const int num_nodes = size();
  QD_ASSERT(nodes[start_node]->empty());
  seed_start_node_reach_gen(nodes[start_node].get(), nodes, after_lower_access);

  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  for (int i = 0; i < num_nodes; i++) {
    if (i != start_node) {
      nodes[i]->reaching_definition_analysis(after_lower_access);
    }
    nodes[i]->reach_in.clear();
    nodes[i]->reach_out = nodes[i]->reach_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }

  // [Worklist algorithm] Converge reach_in / reach_out iteratively.
  while (!to_visit.empty()) {
    auto *now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->reach_in.clear();
    for (auto *prev_node : now->prev) {
      now->reach_in.insert(prev_node->reach_out.begin(), prev_node->reach_out.end());
    }
    auto old_out = std::move(now->reach_out);
    now->reach_out = now->reach_gen;
    for (auto *stmt : now->reach_in) {
      if (!is_reach_in_stmt_killed_at(now, stmt)) {
        now->reach_out.insert(stmt);
      }
    }
    if (now->reach_out != old_out) {
      for (auto *next_node : now->next) {
        if (!in_queue[next_node]) {
          to_visit.push(next_node);
          in_queue[next_node] = true;
        }
      }
    }
  }
}

namespace {

// === Helpers for ControlFlowGraph::live_variable_analysis ===

// Seed the synthetic exit node's live_gen with every store destination in the kernel that may
// still be observable after the kernel exits (globals not flagged eliminable by the SFG, plus any
// aliased pointers via `get_alias`). Skipped entirely after access lowering, when only locals
// remain and nothing escapes.
void seed_final_node_live_gen(CFGNode *final_node_ptr,
                              const std::vector<std::unique_ptr<CFGNode>> &nodes,
                              bool after_lower_access,
                              const std::optional<ControlFlowGraph::LiveVarAnalysisConfig> &config_opt) {
  final_node_ptr->live_gen.clear();
  final_node_ptr->live_kill.clear();
  if (after_lower_access) {
    return;
  }
  for (const auto &node : nodes) {
    for (int j = node->begin_location; j < node->end_location; j++) {
      auto *stmt = node->block->statements[j].get();
      for (auto *store_ptr : irpass::analysis::get_store_destination(stmt, /*get_alias=*/true)) {
        if (in_final_node_live_gen(store_ptr, config_opt)) {
          final_node_ptr->live_gen.insert(store_ptr);
        }
      }
    }
  }
}

}  // namespace

void ControlFlowGraph::live_variable_analysis(bool after_lower_access,
                                              const std::optional<LiveVarAnalysisConfig> &config_opt) {
  // Per-node:
  //   - live_gen:  address loaded with no preceding store in this node (must live in from
  //                somewhere -- you can't load before storing).
  //   - live_kill: address stored in this node.
  // Per-graph (worklist fixpoint, propagated backwards):
  //   - live_in:  live_gen + (live_out - live_kill).
  //   - live_out: union of live_in of all successor nodes.
  QD_AUTO_PROF;
  const int num_nodes = size();
  QD_ASSERT(nodes[final_node]->empty());
  seed_final_node_live_gen(nodes[final_node].get(), nodes, after_lower_access, config_opt);

  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  // Push in reversed order: backwards analysis converges slightly faster when the worklist seeds
  // are dequeued from the back of the graph first.
  for (int i = num_nodes - 1; i >= 0; i--) {
    if (i != final_node) {
      nodes[i]->live_variable_analysis(after_lower_access);
    }
    nodes[i]->live_out.clear();
    nodes[i]->live_in = nodes[i]->live_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }

  // [Worklist algorithm] Converge live_in / live_out iteratively (backwards).
  while (!to_visit.empty()) {
    auto *now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->live_out.clear();
    for (auto *next_node : now->next) {
      now->live_out.insert(next_node->live_in.begin(), next_node->live_in.end());
    }
    auto old_in = std::move(now->live_in);
    now->live_in = now->live_gen;
    for (auto *stmt : now->live_out) {
      if (!CFGNode::contain_variable(now->live_kill, stmt)) {
        now->live_in.insert(stmt);
      }
    }
    if (now->live_in != old_in) {
      for (auto *prev_node : now->prev) {
        if (!in_queue[prev_node]) {
          to_visit.push(prev_node);
          in_queue[prev_node] = true;
        }
      }
    }
  }
}

void ControlFlowGraph::simplify_graph() {
  // Simplify the graph structure, do not modify the IR.
  const int num_nodes = size();
  while (true) {
    bool modified = false;
    for (int i = 0; i < num_nodes; i++) {
      // If a node is empty with in-degree or out-degree <= 1, we can eliminate
      // it (except for the start node and the final node).
      if (nodes[i] && nodes[i]->empty() && i != start_node && i != final_node &&
          (nodes[i]->prev.size() <= 1 || nodes[i]->next.size() <= 1)) {
        erase(i);
        modified = true;
      }
    }
    if (!modified)
      break;
  }
  int new_num_nodes = 0;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]) {
      if (i != new_num_nodes) {
        nodes[new_num_nodes] = std::move(nodes[i]);
      }
      if (final_node == i) {
        final_node = new_num_nodes;
      }
      new_num_nodes++;
    }
  }
  nodes.resize(new_num_nodes);
}

bool ControlFlowGraph::unreachable_code_elimination() {
  // Note that container statements are not in the control-flow graph, so
  // this pass cannot eliminate container statements properly for now.
  QD_AUTO_PROF;
  std::unordered_set<CFGNode *> visited;
  std::queue<CFGNode *> to_visit;
  to_visit.push(nodes[start_node].get());
  visited.insert(nodes[start_node].get());
  // Breadth-first search
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    for (auto &next : now->next) {
      if (visited.find(next) == visited.end()) {
        to_visit.push(next);
        visited.insert(next);
      }
    }
  }
  bool modified = false;
  for (auto &node : nodes) {
    if (visited.find(node.get()) == visited.end()) {
      // unreachable
      if (!node->empty()) {
        while (!node->empty())
          node->erase(node->end_location - 1);
        modified = true;
      }
    }
  }
  return modified;
}

bool ControlFlowGraph::store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled) {
  // The key idea of load-store-forwarding is to find a use-define-chain,
  // which is essentially the load-store-chain in CHI IR.
  //
  // Analysis of the load-store-chain can be separated into two parts:
  // 1. cross-node (roughly means cross-blocks) analysis
  //    This is done in reaching_definition_analysis(), generating reach_in and
  //    reach_out
  //
  // 2. analysis within a node (intra-block analysis):
  //   This is done in CFGNode::store_to_load_forwarding() of each node

  QD_AUTO_PROF;
  reaching_definition_analysis(after_lower_access);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->store_to_load_forwarding(after_lower_access, autodiff_enabled))
      modified = true;
  }
  return modified;
}

bool ControlFlowGraph::dead_store_elimination(bool after_lower_access,
                                              const std::optional<LiveVarAnalysisConfig> &lva_config_opt) {
  QD_AUTO_PROF;
  live_variable_analysis(after_lower_access, lva_config_opt);
  const int num_nodes = size();
  bool modified = false;
  for (int i = 0; i < num_nodes; i++) {
    if (nodes[i]->dead_store_elimination(after_lower_access))
      modified = true;
  }
  return modified;
}

std::unordered_set<SNode *> ControlFlowGraph::gather_loaded_snodes() {
  QD_AUTO_PROF;
  reaching_definition_analysis(/*after_lower_access=*/false);
  const int num_nodes = size();
  std::unordered_set<SNode *> snodes;

  // Note: since global store may only partially modify a value state, the
  // result (which contains the modified and unmodified part) actually needs a
  // read from the previous version of the value state.
  //
  // I.e.,
  // output_value_state = merge(input_value_state, written_part)
  //
  // Therefore we include the nodes[final_node]->reach_in in snodes.
  for (auto &stmt : nodes[final_node]->reach_in) {
    if (auto global_ptr = stmt->cast<GlobalPtrStmt>()) {
      snodes.insert(global_ptr->snode);
    }
  }

  for (int i = 0; i < num_nodes; i++) {
    if (i != final_node) {
      nodes[i]->gather_loaded_snodes(snodes);
    }
  }
  return snodes;
}

namespace {

// === Helpers for ControlFlowGraph::determine_ad_stack_size ===

struct AdStackIndex {
  // AdStackAllocaStmt* -> contiguous int id, populated only for adaptive stacks
  // (max_size == 0) that have not yet been resolved by an earlier pass.
  std::unordered_map<AdStackAllocaStmt *, int> stack_id;
  std::vector<AdStackAllocaStmt *> stacks;
};

AdStackIndex collect_adaptive_ad_stacks(const std::vector<std::unique_ptr<CFGNode>> &nodes) {
  AdStackIndex idx;
  for (const auto &node : nodes) {
    for (int j = node->begin_location; j < node->end_location; j++) {
      Stmt *stmt = node->block->statements[j].get();
      auto *stack = stmt->cast<AdStackAllocaStmt>();
      if (!stack || stack->max_size != 0) {
        continue;
      }
      if (idx.stack_id.emplace(stack, static_cast<int>(idx.stacks.size())).second) {
        idx.stacks.push_back(stack);
      }
    }
  }
  return idx;
}

struct AdStackPerNodeSizes {
  // [stack_id][node_id]. `max_increased_size[s][j]` is the maximum (pushes - pops) of stack |s|
  // among all prefixes of CFGNode |j|; `increased_size[s][j]` is the net (pushes - pops) in the
  // whole node. Indexed by contiguous stack id so the per-stack DP reads cheap vectors instead of
  // hashing.
  std::vector<std::vector<int>> increased_size;
  std::vector<std::vector<int>> max_increased_size;
  // True iff the stack actually appears in any push/pop in the CFG. Inactive stacks would settle
  // at `max_size = 0` regardless and are short-circuited below to reproduce the original "Unused
  // autodiff stack" warning.
  std::vector<bool> stack_active;
};

AdStackPerNodeSizes accumulate_per_stack_per_node_size_deltas(
    const std::vector<std::unique_ptr<CFGNode>> &nodes,
    const std::unordered_map<AdStackAllocaStmt *, int> &stack_id,
    int num_stacks) {
  const int num_nodes = static_cast<int>(nodes.size());
  AdStackPerNodeSizes out;
  out.increased_size.assign(num_stacks, std::vector<int>(num_nodes, 0));
  out.max_increased_size.assign(num_stacks, std::vector<int>(num_nodes, 0));
  out.stack_active.assign(num_stacks, false);
  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      Stmt *stmt = nodes[i]->block->statements[j].get();
      AdStackAllocaStmt *stack = nullptr;
      int delta = 0;
      if (auto *push = stmt->cast<AdStackPushStmt>()) {
        stack = push->stack->as<AdStackAllocaStmt>();
        delta = +1;
      } else if (auto *pop = stmt->cast<AdStackPopStmt>()) {
        stack = pop->stack->as<AdStackAllocaStmt>();
        delta = -1;
      } else {
        continue;
      }
      if (stack->max_size != 0 /*non-adaptive*/) {
        continue;
      }
      auto it = stack_id.find(stack);
      QD_ASSERT(it != stack_id.end());
      const int sid = it->second;
      out.stack_active[sid] = true;
      int &cur = out.increased_size[sid][i];
      cur += delta;
      if (cur > out.max_increased_size[sid][i]) {
        out.max_increased_size[sid][i] = cur;
      }
    }
  }
  return out;
}

// Precompute outgoing-edge node ids once per node so the per-stack DP walks an int vector instead
// of hashing `node_ids[next_node]` on every traversal.
std::vector<std::vector<int>> compute_outgoing_node_ids(const std::vector<std::unique_ptr<CFGNode>> &nodes) {
  const int num_nodes = static_cast<int>(nodes.size());
  std::unordered_map<CFGNode *, int> node_ids;
  node_ids.reserve(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    node_ids[nodes[i].get()] = i;
  }
  std::vector<std::vector<int>> next_ids(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    auto &dst = next_ids[i];
    dst.reserve(nodes[i]->next.size());
    for (auto *next_node : nodes[i]->next) {
      dst.push_back(node_ids[next_node]);
    }
  }
  return next_ids;
}

struct TarjanResult {
  std::vector<int> scc_id;                  // node -> SCC index
  std::vector<int> dfs_finish;              // node -> DFS post-order index; ancestors finish AFTER descendants
  std::vector<std::vector<int>> scc_nodes;  // SCC -> list of node ids
};

// Iterative Tarjan SCC. Emits SCCs in reverse topological order (sources end up at the largest
// indices). Also records DFS post-order finish times, used downstream to split each cyclic SCC's
// intra-edges into forward vs back without a second edge-classification pass.
TarjanResult tarjan_scc(const std::vector<std::vector<int>> &next_ids) {
  const int num_nodes = static_cast<int>(next_ids.size());
  TarjanResult out;
  out.scc_id.assign(num_nodes, -1);
  out.dfs_finish.assign(num_nodes, -1);

  std::vector<int> tarjan_index(num_nodes, -1);
  std::vector<int> tarjan_lowlink(num_nodes, 0);
  std::vector<char> on_stack(num_nodes, 0);
  std::vector<int> tarjan_stack;
  tarjan_stack.reserve(num_nodes);
  std::vector<std::pair<int, std::size_t>> dfs_stack;
  dfs_stack.reserve(num_nodes);
  int next_index = 0;
  int next_finish = 0;
  int next_scc = 0;
  for (int origin = 0; origin < num_nodes; origin++) {
    if (tarjan_index[origin] != -1) {
      continue;
    }
    tarjan_index[origin] = next_index;
    tarjan_lowlink[origin] = next_index;
    next_index++;
    tarjan_stack.push_back(origin);
    on_stack[origin] = 1;
    dfs_stack.emplace_back(origin, 0);
    while (!dfs_stack.empty()) {
      auto &frame = dfs_stack.back();
      const int u = frame.first;
      const auto &nb = next_ids[u];
      if (frame.second < nb.size()) {
        const int v = nb[frame.second++];
        if (tarjan_index[v] == -1) {
          tarjan_index[v] = next_index;
          tarjan_lowlink[v] = next_index;
          next_index++;
          tarjan_stack.push_back(v);
          on_stack[v] = 1;
          dfs_stack.emplace_back(v, 0);
        } else if (on_stack[v]) {
          if (tarjan_index[v] < tarjan_lowlink[u]) {
            tarjan_lowlink[u] = tarjan_index[v];
          }
        }
      } else {
        if (tarjan_lowlink[u] == tarjan_index[u]) {
          std::vector<int> component;
          while (true) {
            const int w = tarjan_stack.back();
            tarjan_stack.pop_back();
            on_stack[w] = 0;
            out.scc_id[w] = next_scc;
            component.push_back(w);
            if (w == u) {
              break;
            }
          }
          out.scc_nodes.push_back(std::move(component));
          next_scc++;
        }
        out.dfs_finish[u] = next_finish++;
        dfs_stack.pop_back();
        if (!dfs_stack.empty()) {
          const int parent = dfs_stack.back().first;
          if (tarjan_lowlink[u] < tarjan_lowlink[parent]) {
            tarjan_lowlink[parent] = tarjan_lowlink[u];
          }
        }
      }
    }
  }
  return out;
}

struct SccEdgeSets {
  // Each intra-SCC edge is either "forward" in the DFS spanning sense (target finishes before
  // source, so source can be relaxed before target with a single topological pass) or "back"
  // (target finishes at or after source, closing a cycle). The forward set drives the per-stack
  // DAG dynamic programming; back-edges only need a single post-DP relaxation check to detect
  // positive cycles. Inter-SCC edges always relax forward in topological order.
  std::vector<std::vector<int>> next_ids_intra_fwd;
  std::vector<std::vector<int>> next_ids_intra_back;
  std::vector<std::vector<int>> next_ids_inter;
};

SccEdgeSets classify_scc_edges(const std::vector<std::vector<int>> &next_ids,
                               const std::vector<int> &scc_id,
                               const std::vector<int> &dfs_finish) {
  const int num_nodes = static_cast<int>(next_ids.size());
  SccEdgeSets out;
  out.next_ids_intra_fwd.assign(num_nodes, {});
  out.next_ids_intra_back.assign(num_nodes, {});
  out.next_ids_inter.assign(num_nodes, {});
  for (int u = 0; u < num_nodes; u++) {
    const int su = scc_id[u];
    for (int v : next_ids[u]) {
      if (scc_id[v] == su) {
        if (dfs_finish[v] < dfs_finish[u]) {
          out.next_ids_intra_fwd[u].push_back(v);
        } else {
          out.next_ids_intra_back[u].push_back(v);
        }
      } else {
        out.next_ids_inter[u].push_back(v);
      }
    }
  }
  return out;
}

struct CyclicSccInfo {
  std::vector<char> scc_is_cyclic;          // 1 iff SCC contains a cycle
  std::vector<std::vector<int>> scc_topo;   // for cyclic SCCs: nodes sorted by descending dfs_finish
};

// An SCC is cyclic iff |S| > 1 (any two nodes in a non-trivial SCC lie on a cycle) or |S| == 1
// with a self-loop edge. For cyclic SCCs we precompute the topological ordering of their nodes
// so the per-stack DAG DP visits each node once with all forward predecessors already finalized.
CyclicSccInfo identify_cyclic_sccs_and_topo(const std::vector<std::vector<int>> &scc_nodes,
                                            const std::vector<std::vector<int>> &next_ids_intra_back,
                                            const std::vector<int> &dfs_finish) {
  const int num_sccs = static_cast<int>(scc_nodes.size());
  CyclicSccInfo out;
  out.scc_is_cyclic.assign(num_sccs, 0);
  out.scc_topo.assign(num_sccs, {});
  for (int s = 0; s < num_sccs; s++) {
    const auto &nodes_in_s = scc_nodes[s];
    if (nodes_in_s.size() > 1) {
      out.scc_is_cyclic[s] = 1;
    } else {
      // Self-loops are classified as back-edges above, so a singleton SCC is cyclic iff it has a
      // back-edge pointing at itself.
      const int n = nodes_in_s[0];
      for (int v : next_ids_intra_back[n]) {
        if (v == n) {
          out.scc_is_cyclic[s] = 1;
          break;
        }
      }
    }
    if (out.scc_is_cyclic[s]) {
      struct DfsFinishGreater {
        const std::vector<int> &dfs_finish;
        bool operator()(int a, int b) const {
          return dfs_finish[a] > dfs_finish[b];
        }
      };
      auto topo = nodes_in_s;
      std::sort(topo.begin(), topo.end(), DfsFinishGreater{dfs_finish});
      out.scc_topo[s] = std::move(topo);
    }
  }
  return out;
}

struct FingerprintGroups {
  std::vector<int> stack_to_rep;   // stack id -> representative stack id (whose DP run it shares)
  std::vector<int> rep_stack_ids;  // representative stack ids that actually run the DP
};

// Group AD-stacks whose per-node (increased_size, max_increased_size) rows are bit-identical so
// the DP runs once per equivalence class instead of once per stack. In practice many AD-stacks in
// the same kernel share their push/pop schedule (one alloca per autodiff variable in the same
// loop body), and the DP on a large CFG dwarfs the dedup cost. We hash a sparse fingerprint (only
// nodes where the stack has activity) and group by it. Worst case (no duplicates): one DP run per
// stack, equivalent to running it directly per stack.
FingerprintGroups group_stacks_by_fingerprint(int num_nodes,
                                              int num_stacks,
                                              const std::vector<bool> &stack_active,
                                              const std::vector<std::vector<int>> &increased_size,
                                              const std::vector<std::vector<int>> &max_increased_size) {
  using Fingerprint = std::vector<std::tuple<int, int, int>>;  // (node_id, is, mis), sorted by node_id
  struct FingerprintHash {
    std::size_t operator()(const Fingerprint &f) const noexcept {
      // FNV-1a mix of three components per fingerprint entry.
      constexpr std::size_t fnv_prime = 1099511628211ULL;
      std::size_t h = 1469598103934665603ULL;
      for (auto &[n, i, m] : f) {
        h ^= static_cast<std::size_t>(n);
        h *= fnv_prime;
        h ^= static_cast<std::size_t>(static_cast<unsigned>(i));
        h *= fnv_prime;
        h ^= static_cast<std::size_t>(static_cast<unsigned>(m));
        h *= fnv_prime;
      }
      return h;
    }
  };
  std::unordered_map<Fingerprint, int, FingerprintHash> fp_to_rep;
  FingerprintGroups out;
  out.stack_to_rep.assign(num_stacks, -1);
  for (int sid = 0; sid < num_stacks; sid++) {
    if (!stack_active[sid]) {
      continue;
    }
    Fingerprint fp;
    const auto &is_row = increased_size[sid];
    const auto &mis_row = max_increased_size[sid];
    for (int n = 0; n < num_nodes; n++) {
      if (is_row[n] != 0 || mis_row[n] != 0) {
        fp.emplace_back(n, is_row[n], mis_row[n]);
      }
    }
    auto [it, inserted] = fp_to_rep.emplace(std::move(fp), sid);
    out.stack_to_rep[sid] = it->second;
    if (inserted) {
      out.rep_stack_ids.push_back(sid);
    }
  }
  return out;
}

enum class CyclicSccFastPath {
  kPositiveCycle,  // proven positive cycle exists for this stack inside this SCC
  kZeroSpread,     // no `is` contribution in the SCC; just spread max entry-side begin value
  kFallback,       // mixed-sign, run the full DP
};

// Sign-based fast paths sidestep the cyclic-SCC dynamic programming when the stack's `is`
// contribution inside this SCC is structurally trivial:
//   1. min_is >= 0 with max_is > 0: every node in the SCC lies on some cycle (SCC property), and
//      a cycle through a strictly-positive node with all non-negative `is` along it sums to a
//      positive value, so a positive cycle exists for this stack. This is the autodiff
//      push-only-in-SCC pattern.
//   2. min_is == max_is == 0: no `is` contribution at all in this SCC; the DP would only spread
//      the maximum entry-side `max_size_at_node_begin` value to every node in O(|S|).
CyclicSccFastPath classify_cyclic_scc_fast_path(const std::vector<int> &nodes_in_s,
                                                const std::vector<int> &is_for_stack) {
  int min_is = INT_MAX;
  int max_is = INT_MIN;
  for (int u : nodes_in_s) {
    const int v = is_for_stack[u];
    if (v < min_is) {
      min_is = v;
    }
    if (v > max_is) {
      max_is = v;
    }
  }
  if (min_is >= 0 && max_is > 0) {
    return CyclicSccFastPath::kPositiveCycle;
  }
  if (min_is == 0 && max_is == 0) {
    return CyclicSccFastPath::kZeroSpread;
  }
  return CyclicSccFastPath::kFallback;
}

// Spread the maximum entry-side `max_size_at_node_begin` value across every node in the SCC.
// Equivalent to running the DP with all-zero `is` weights, in O(|S|).
void spread_max_begin_over_zero_scc(const std::vector<int> &nodes_in_s,
                                    std::vector<int> &max_size_at_node_begin) {
  int max_begin = -1;
  for (int u : nodes_in_s) {
    if (max_size_at_node_begin[u] > max_begin) {
      max_begin = max_size_at_node_begin[u];
    }
  }
  if (max_begin < 0) {
    return;
  }
  for (int u : nodes_in_s) {
    if (max_size_at_node_begin[u] < max_begin) {
      max_size_at_node_begin[u] = max_begin;
    }
  }
}

// Mixed-sign case: single-pass dynamic programming on the SCC's forward edges (processed in
// descending DFS finish-time so every forward predecessor is finalized before its successor
// relaxes), followed by one relaxation check on the back-edges. Correctness: every walk inside
// the SCC decomposes into a forward path plus zero or more closed cycles (back-edge + forward
// path). For SCCs with no positive cycle, traversing a cycle adds a non-positive amount to the
// running size and so cannot improve max_size beyond what the forward DP already computed. The
// back-edge relaxation check after the DP detects the only failure mode (some back-edge would
// still improve a forward predecessor's value, which can only happen if a positive cycle exists
// for this stack). Returns true iff a positive cycle was detected.
bool dp_mixed_sign_cyclic_scc(const std::vector<int> &topo,
                              const std::vector<int> &nodes_in_s,
                              const std::vector<int> &is_for_stack,
                              const std::vector<std::vector<int>> &next_ids_intra_fwd,
                              const std::vector<std::vector<int>> &next_ids_intra_back,
                              std::vector<int> &max_size_at_node_begin) {
  for (int u : topo) {
    const int begin = max_size_at_node_begin[u];
    if (begin < 0) {
      continue;
    }
    const int exit_val = begin + is_for_stack[u];
    for (int v : next_ids_intra_fwd[u]) {
      if (exit_val > max_size_at_node_begin[v]) {
        max_size_at_node_begin[v] = exit_val;
      }
    }
  }
  for (int u : nodes_in_s) {
    const int begin = max_size_at_node_begin[u];
    if (begin < 0) {
      continue;
    }
    const int exit_val = begin + is_for_stack[u];
    for (int v : next_ids_intra_back[u]) {
      if (exit_val > max_size_at_node_begin[v]) {
        return true;
      }
    }
  }
  return false;
}

// SCC has converged for this stack. Update global `max_size` from each node's max-prefix
// contribution, then relax inter-SCC outgoing edges into successor SCCs (predecessors are already
// finalized when an SCC is entered, so each inter-SCC edge is touched exactly once).
void update_global_max_and_relax_inter_scc(const std::vector<int> &nodes_in_s,
                                           const std::vector<int> &is_for_stack,
                                           const std::vector<int> &mis_for_stack,
                                           const std::vector<std::vector<int>> &next_ids_inter,
                                           std::vector<int> &max_size_at_node_begin,
                                           int &max_size) {
  for (int u : nodes_in_s) {
    const int begin = max_size_at_node_begin[u];
    if (begin < 0) {
      continue;
    }
    const int prefix = begin + mis_for_stack[u];
    if (prefix > max_size) {
      max_size = prefix;
    }
    const int exit_val = begin + is_for_stack[u];
    for (int v : next_ids_inter[u]) {
      if (exit_val > max_size_at_node_begin[v]) {
        max_size_at_node_begin[v] = exit_val;
      }
    }
  }
}

struct AdStackDPResult {
  int max_size;
  bool has_positive_loop;
};

// Run the per-representative DP. Walks the SCC condensation in topological order (sources first,
// since Tarjan emits SCCs in reverse-topological order so source SCCs end up at the largest
// indices). `max_size_at_node_begin` is taken by reference as a scratch buffer reused across reps
// to avoid reallocating per iteration.
AdStackDPResult run_ad_stack_size_dp_for_representative(int start_node,
                                                        int num_sccs,
                                                        const std::vector<int> &is_for_stack,
                                                        const std::vector<int> &mis_for_stack,
                                                        const std::vector<std::vector<int>> &scc_nodes,
                                                        const std::vector<char> &scc_is_cyclic,
                                                        const std::vector<std::vector<int>> &scc_topo,
                                                        const std::vector<std::vector<int>> &next_ids_intra_fwd,
                                                        const std::vector<std::vector<int>> &next_ids_intra_back,
                                                        const std::vector<std::vector<int>> &next_ids_inter,
                                                        std::vector<int> &max_size_at_node_begin) {
  std::fill(max_size_at_node_begin.begin(), max_size_at_node_begin.end(), -1);
  max_size_at_node_begin[start_node] = 0;
  int max_size = 0;
  for (int s = num_sccs - 1; s >= 0; s--) {
    const auto &nodes_in_s = scc_nodes[s];
    if (scc_is_cyclic[s]) {
      switch (classify_cyclic_scc_fast_path(nodes_in_s, is_for_stack)) {
        case CyclicSccFastPath::kPositiveCycle:
          return {max_size, /*has_positive_loop=*/true};
        case CyclicSccFastPath::kZeroSpread:
          spread_max_begin_over_zero_scc(nodes_in_s, max_size_at_node_begin);
          break;
        case CyclicSccFastPath::kFallback:
          if (dp_mixed_sign_cyclic_scc(scc_topo[s], nodes_in_s, is_for_stack, next_ids_intra_fwd, next_ids_intra_back,
                                       max_size_at_node_begin)) {
            return {max_size, /*has_positive_loop=*/true};
          }
          break;
      }
    }
    update_global_max_and_relax_inter_scc(nodes_in_s, is_for_stack, mis_for_stack, next_ids_inter,
                                          max_size_at_node_begin, max_size);
  }
  return {max_size, /*has_positive_loop=*/false};
}

// Broadcast per-representative DP results to every active stack and apply the resolved
// `max_size`. Stacks with positive cycles are left at `max_size = 0` so the structural
// bounded-loop pre-pass in `irpass::determine_ad_stack_size` gets a chance to derive a symbolic
// bound; if it also cannot, the caller emits a hard compile error (there is no compile-time
// `default_ad_stack_size` fallback).
void apply_ad_stack_dp_results(const std::vector<AdStackAllocaStmt *> &stacks,
                               const std::vector<bool> &stack_active,
                               const std::vector<int> &stack_to_rep,
                               const std::unordered_map<int, AdStackDPResult> &rep_results) {
  const int num_stacks = static_cast<int>(stacks.size());
  for (int sid = 0; sid < num_stacks; sid++) {
    AdStackAllocaStmt *stack = stacks[sid];
    if (!stack_active[sid]) {
      // No push/pop in the CFG: the DP would visit reachable nodes with all-zero edge weights and
      // settle with `max_size = 0`, no positive loop. Reproduce that result directly.
      QD_WARN("Unused autodiff stack {} should have been eliminated.", stack->name());
      continue;
    }
    const AdStackDPResult &res = rep_results.at(stack_to_rep[sid]);
    if (res.has_positive_loop) {
      // Leave `max_size = 0` so the symbolic-bound pre-pass can take over.
      continue;
    }
    // Since we use |max_size| == 0 for adaptive sizes, we do not want stacks with maximum capacity
    // indeed equal to 0.
    QD_WARN_IF(res.max_size == 0, "Unused autodiff stack {} should have been eliminated.", stack->name());
    stack->max_size = res.max_size;
  }
}

}  // namespace

void ControlFlowGraph::determine_ad_stack_size() {
  /**
   * Determine the necessary size of every adaptive AD-stack on the control-flow graph (CFG). For each AD-stack we
   * compute the maximum running net push count along any walk from the kernel entry. AD-stacks whose forward kernel
   * contains a positive cycle (pushes > pops around a loop) are left at `max_size = 0`, and the caller routes them
   * through the structural bounded-loop pre-pass for a symbolic `SizeExpr`, hard-erroring if the grammar still
   * cannot resolve them. There is no compile-time size fallback.
   *
   * Implementation notes for compile-time perf on large reverse-mode kernels:
   *   1. Per-stack per-node pre-aggregates (`max_increased_size`, `increased_size`) are stored in dense
   *      `vector<vector<int>>` indexed by a contiguous int stack id, instead of an
   *      `unordered_map<AdStackAllocaStmt*, vector<int>>` -- this removes hash traffic from the hot inner loop.
   *   2. Stacks whose `(increased_size, max_increased_size)` row pair is bit-identical share a single dynamic
   *      programming run -- typical kernels generate one alloca per autodiff variable in the same loop body, so
   *      most rows collapse to a few representatives.
   *   3. The CFG is condensed via Tarjan into strongly connected components (SCCs). DFS finish times recorded
   *      during the same Tarjan pass split each cyclic SCC's intra-edges into a forward set (target finishes
   *      before source) and a back set (target finishes at or after source). Per representative we run a
   *      single-pass dynamic-programming (DP) sweep over the forward edges in descending finish-time order, then
   *      check the back edges once for positive-cycle relaxation. Correctness: any walk inside an SCC decomposes
   *      into a forward path plus zero or more cycles, and an SCC with no positive cycle has the same max-walk-sum
   *      as the back-edge-removed DAG; a positive cycle is exactly the case where some back-edge would still relax
   *      after the forward DP. This drops the per-cyclic-SCC cost from O(|S| * |E_S|) to O(|S| + |E_S|).
   *   4. Two sign-based fast paths short-circuit the DP for trivial cyclic SCCs: an SCC with `min_is >= 0 && max_is
   *      > 0` for this stack must contain a positive cycle (every node lies on some cycle, and a cycle through a
   *      strictly-positive node with all non-negative `is` along it sums positive); an SCC with `min_is == 0 ==
   *      max_is` has no `is` contribution at all and is handled by spreading the max entry-side
   *      `max_size_at_node_begin` to every node in O(|S|).
   * Per-rep cost becomes O(V + E + sum_{cyclic S} |S| * |E_S|) (with the SCC sum dropping to O(|S| + |E_S|) for
   * the common autodiff push/pop pattern); overall cost is O(V + E + R * (V + E)) with R the number of distinct
   * row-pair representatives.
   */
  AdStackIndex idx = collect_adaptive_ad_stacks(nodes);
  const int num_stacks = static_cast<int>(idx.stacks.size());
  if (num_stacks == 0) {
    return;
  }

  const int num_nodes = size();
  AdStackPerNodeSizes sizes = accumulate_per_stack_per_node_size_deltas(nodes, idx.stack_id, num_stacks);
  std::vector<std::vector<int>> next_ids = compute_outgoing_node_ids(nodes);

  TarjanResult tarjan = tarjan_scc(next_ids);
  const int num_sccs = static_cast<int>(tarjan.scc_nodes.size());
  SccEdgeSets edges = classify_scc_edges(next_ids, tarjan.scc_id, tarjan.dfs_finish);
  CyclicSccInfo cyc = identify_cyclic_sccs_and_topo(tarjan.scc_nodes, edges.next_ids_intra_back, tarjan.dfs_finish);

  FingerprintGroups groups = group_stacks_by_fingerprint(num_nodes, num_stacks, sizes.stack_active, sizes.increased_size,
                                                         sizes.max_increased_size);

  std::vector<int> max_size_at_node_begin(num_nodes);
  std::unordered_map<int, AdStackDPResult> rep_results;
  rep_results.reserve(groups.rep_stack_ids.size());
  for (int rep_sid : groups.rep_stack_ids) {
    rep_results[rep_sid] = run_ad_stack_size_dp_for_representative(
        start_node, num_sccs, sizes.increased_size[rep_sid], sizes.max_increased_size[rep_sid], tarjan.scc_nodes,
        cyc.scc_is_cyclic, cyc.scc_topo, edges.next_ids_intra_fwd, edges.next_ids_intra_back, edges.next_ids_inter,
        max_size_at_node_begin);
  }

  apply_ad_stack_dp_results(idx.stacks, sizes.stack_active, groups.stack_to_rep, rep_results);
}

}  // namespace quadrants::lang
