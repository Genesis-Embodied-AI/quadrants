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
    return std::any_of(var_set.begin(), var_set.end(),
                       [&](Stmt *set_var) { return irpass::analysis::definitely_same_address(var, set_var); });
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
    return std::any_of(var_set.begin(), var_set.end(), [&](const auto &set_var) {
      if (irpass::analysis::definitely_same_address(var, set_var.first)) {
        return set_var.second != CFGNode::UseDefineStatus::PARTIAL;
      }
      return false;
    });
  }
}

bool CFGNode::may_contain_variable(const std::unordered_map<Stmt *, CFGNode::UseDefineStatus> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    return std::any_of(var_set.begin(), var_set.end(),
                       [&](const auto &set_var) { return irpass::analysis::maybe_same_address(var, set_var.first); });
  }
}

bool CFGNode::may_contain_variable(const std::unordered_set<Stmt *> &var_set, Stmt *var) {
  if (var->is<AllocaStmt>() || var->is<AdStackAllocaStmt>()) {
    return var_set.find(var) != var_set.end();
  } else {
    // TODO: How to optimize this?
    if (var_set.find(var) != var_set.end())
      return true;
    return std::any_of(var_set.begin(), var_set.end(),
                       [&](Stmt *set_var) { return irpass::analysis::maybe_same_address(var, set_var); });
  }
}

bool CFGNode::reach_kill_variable(Stmt *var) const {
  // Does this node (definitely) kill a definition of var?
  return contain_variable(reach_kill, var);
}

// var: dest_addr
Stmt *CFGNode::get_store_forwarding_data(Stmt *var, int position) const {
  // Return the stored data if all definitions in the UD-chain of |var| at
  // this position store the same data.
  // [Intra-block Search]
  int last_def_position = -1;
  for (int i = position - 1; i >= begin_location; i--) {
    // Find previous store stmt to the same dest_addr, stop at the closest one.
    // store_ptr: prev-store dest_addr
    for (auto store_ptr : irpass::analysis::get_store_destination(block->statements[i].get())) {
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
        last_def_position = i;
        break;
      }

      // Special case:
      // $1 = store $0, MatrixInitStmt(...)
      // ...
      // $2 = matrix ptr $0, offset
      // $3 = load $2
      // We can forward MatrixInitStmt->values[offset] to $3
      if (var->is<MatrixPtrStmt>() && var->as<MatrixPtrStmt>()->offset->is<ConstStmt>()) {
        auto var_origin = var->as<MatrixPtrStmt>()->origin;
        // Check for same origin address
        if (irpass::analysis::definitely_same_address(var_origin, store_ptr)) {
          // Check for MatrixInitStmt
          Stmt *store_data = irpass::analysis::get_store_data(block->statements[i].get());
          if (store_data->is<MatrixInitStmt>()) {
            last_def_position = i;
            break;
          }
        }
      }
    }
    if (last_def_position != -1) {
      break;
    }
  }

  // Check if store_stmt will ever influence the value of var
  auto may_contain_address = [&](Stmt *store_stmt, Stmt *var) {
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
  };

  // Check for aliased address
  // There's a store to the same dest_addr before this stmt
  if (last_def_position != -1) {
    // result: the value to store
    Stmt *result = irpass::analysis::get_store_data(block->statements[last_def_position].get());
    bool is_tensor_involved = var->ret_type.ptr_removed()->is<TensorType>();
    if (!(var->is<AllocaStmt>() && !is_tensor_involved)) {
      // In between the store stmt and current stmt,
      // if there's a third-stmt that "may" have stored a "different value" to
      // the "same dest_addr", then we can't forward the stored data.
      for (int i = last_def_position + 1; i < position; i++) {
        // A nested child-kernel launch is an opaque barrier: it may overwrite the loaded address, so never forward
        // a store across it.
        if (block->statements[i]->is<ChildLaunchStmt>()) {
          return nullptr;
        }
        if (!irpass::analysis::same_value(result, irpass::analysis::get_store_data(block->statements[i].get()))) {
          if (may_contain_address(block->statements[i].get(), var)) {
            return nullptr;
          }
        }
      }
    }
    return result;
  }

  // [Cross-block search]
  // Search for store to the same dest_addr in reach_in and reach_gen
  Stmt *result = nullptr;
  bool result_visible = false;
  auto visible = [&](Stmt *stmt) {
    // Check if |stmt| is before |position| here.
    if (stmt->parent == block) {
      return stmt->parent->locate(stmt) < position;
    }
    // |parent_blocks| is precomputed in the constructor of CFGNode.
    // TODO: What if |stmt| appears in an ancestor of |block| but after
    //  |position|?
    return parent_blocks_.find(stmt->parent) != parent_blocks_.end();
  };
  /**
   * |stmt| is a definition in the UD-chain of |var|. Update |result| with
   * |stmt|. If either the stored data of |stmt| is not forwardable or the
   * stored data of |stmt| is not definitely the same as other definitions of
   * |var|, return false to show that there is no store-to-load forwardable
   * data.
   */
  auto update_result = [&](Stmt *stmt) {
    auto data = irpass::analysis::get_store_data(stmt);
    if (!data) {     // not forwardable
      return false;  // return nullptr
    }
    if (!result) {
      result = data;
      result_visible = visible(data);
      return true;  // continue the following loops
    }
    if (!irpass::analysis::same_value(result, data)) {
      // check the special case of alloca (initialized to 0)
      if (!(result->is<AllocaStmt>() && data->is<ConstStmt>() && data->as<ConstStmt>()->val.equal_value(0))) {
        return false;  // return nullptr
      }
    }
    if (!result_visible && visible(data)) {
      // pick the visible one for store-to-load forwarding
      result = data;
      result_visible = true;
    }
    return true;  // continue the following loops
  };

  // [Global Addr only]
  // test whether there's a store to the same dest_addr in a previous block.
  // if the store values are the same, then return the value
  last_def_position = -1;
  for (auto stmt : reach_in) {
    // var == stmt is for the case that a global ptr is never stored.
    // In this case, stmt is from nodes[start_node]->reach_gen.
    if (var == stmt || may_contain_address(stmt, var)) {
      if (!update_result(stmt))
        return nullptr;
      else
        last_def_position = 0;
    }
  }

  // test whether there's a store to the same dest_addr before this stmt (in
  // reach_gen)
  //  if the store values are the same, then return the value
  for (auto stmt : reach_gen) {
    if (may_contain_address(stmt, var) && stmt->parent->locate(stmt) < position) {
      if (!update_result(stmt))
        return nullptr;
      else
        last_def_position = stmt->parent->locate(stmt);
    }
  }
  if (!result) {
    // The UD-chain is empty.
    auto offending_load = block->statements[position].get();
    ErrorEmitter(QuadrantsIrWarning(), offending_load,
                 fmt::format("Loading variable {} before anything is stored to it.", var->id));
    return nullptr;
  }
  if (!result_visible) {
    // The data is store-to-load forwardable but not visible at the place we
    // are going to forward. We cannot forward it in this case.
    return nullptr;
  }

  if (last_def_position == -1)
    return nullptr;

  // Check for aliased address
  // There's a store to the same dest_addr before this stmt
  bool is_tensor_involved = var->ret_type.ptr_removed()->is<TensorType>();
  if (!(var->is<AllocaStmt>() && !is_tensor_involved)) {
    // In between the store stmt and current stmt,
    // if there's a third-stmt that "may" have stored a "different value" to
    // the "same dest_addr", then we can't forward the stored data.
    for (int i = last_def_position; i < position; i++) {
      // See the intra-block scan above: never forward across a nested child-kernel launch.
      if (block->statements[i]->is<ChildLaunchStmt>()) {
        return nullptr;
      }
      if (!irpass::analysis::same_value(result, irpass::analysis::get_store_data(block->statements[i].get()))) {
        if (may_contain_address(block->statements[i].get(), var)) {
          return nullptr;
        }
      }
    }
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

bool CFGNode::store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled) {
  // Contains two separate parts:
  // 1. Store-to-load Forwarding: for each load stmt, find the closest previous
  // store stmt
  //        that stores to the same address as the load stmt, then replace
  //        load with the "val".
  // 2. Identical Store Elimination: for each store stmt, find the closest
  // previous store stmt
  //        that stores to the same address as the store stmt. If the "val"s
  //        are the same, then remove the store stmt.
  bool modified = false;
  for (int i = begin_location; i < end_location; i++) {
    // Store-to-load forwarding
    auto stmt = block->statements[i].get();

    // result: the value to be store/load
    Stmt *result = nullptr;

    // [get_store_forwarding_data] find the store stmt that:
    // 1. stores to the same address and as the load stmt
    // 2. (one value at a time) closest to the load stmt but before the load
    // stmt
    Stmt *load_src = nullptr;
    if (auto local_load = stmt->cast<LocalLoadStmt>()) {
      result = get_store_forwarding_data(local_load->src, i);
      load_src = local_load->src;
    } else if (auto global_load = stmt->cast<GlobalLoadStmt>()) {
      if (!after_lower_access && !autodiff_enabled) {
        result = get_store_forwarding_data(global_load->src, i);
        load_src = global_load->src;
      }
    }

    // [Apply Load-Store-Forwarding]
    // replace load stmt with the value-"result"
    if (result) {
      // Forward the stored data |result|.
      if (result->is<AllocaStmt>()) {
        // TensorType does not apply to this special case
        if (result->ret_type.ptr_removed()->is<TensorType>())
          continue;

        // special case of alloca (initialized to 0)
        auto zero = Stmt::make<ConstStmt>(TypedConstant(result->ret_type.ptr_removed(), 0));
        replace_with(i, std::move(zero), true);
      } else {
        if (result->ret_type.ptr_removed()->is<TensorType>() && !stmt->ret_type->is<TensorType>()) {
          QD_ASSERT(load_src->is<MatrixPtrStmt>() && load_src->as<MatrixPtrStmt>()->offset->is<ConstStmt>());
          QD_ASSERT(result->is<MatrixInitStmt>());

          int offset = load_src->as<MatrixPtrStmt>()->offset->as<ConstStmt>()->val.val_int32();

          result = result->as<MatrixInitStmt>()->values[offset];
        }

        stmt->replace_usages_with(result);
        erase(i);  // This causes end_location--
        i--;       // to cancel i++ in the for loop
        modified = true;
      }
      continue;
    }

    // [Identical store elimination]
    // find the store stmt that:
    // 1. stores to the same address as the current store stmt
    // 2. has the same store value as the current store stmt
    // 3. (one value at a time) closest to the current store stmt but before the
    // current store stmt then erase the current store stmt
    if (auto local_store = stmt->cast<LocalStoreStmt>()) {
      result = get_store_forwarding_data(local_store->dest, i);
      if (result && result->is<AllocaStmt>() && !autodiff_enabled) {
        // TensorType does not apply to this special case
        if (result->ret_type.ptr_removed()->is<TensorType>()) {
          continue;
        }

        // special case of alloca (initialized to 0)
        if (auto stored_data = local_store->val->cast<ConstStmt>()) {
          if (stored_data->val.equal_value(0)) {
            erase(i);  // This causes end_location--
            i--;       // to cancel i++ in the for loop
            modified = true;
          }
        }
      } else {
        // not alloca
        if (irpass::analysis::same_value(result, local_store->val)) {
          erase(i);  // This causes end_location--
          i--;       // to cancel i++ in the for loop
          modified = true;
        }
      }
    } else if (auto global_store = stmt->cast<GlobalStoreStmt>()) {
      if (!after_lower_access) {
        result = get_store_forwarding_data(global_store->dest, i);
        if (irpass::analysis::same_value(result, global_store->val)) {
          erase(i);  // This causes end_location--
          i--;       // to cancel i++ in the for loop
          modified = true;
        }
      }
    }
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

bool CFGNode::dead_store_elimination(bool after_lower_access) {
  bool modified = false;
  // Map a variable to its nearest load
  std::unordered_map<Stmt *, Stmt *> live_load_in_this_node;

  // For any stmt with TensorType'd address, the address can be either partially
  // or fully stored/loaded, which will eventually influence the
  // dead-store-elimination strategy
  //
  // Here we use CFGNode::UseDefineStatus to mark whether a TensorType'd address
  // is fully or partially modified.
  std::unordered_map<Stmt *, CFGNode::UseDefineStatus> live_in_this_node;
  std::unordered_map<Stmt *, CFGNode::UseDefineStatus> killed_in_this_node;

  // Search for aliased addresses
  // tensor_to_matrix_ptrs_map: map MatrixPtrStmt->origin to list of
  //   MatrixPtrStmts
  // matrix_ptr_to_tensor_map: map MatrixPtrStmt to
  //   MatrixPtrStmt->origin
  std::unordered_map<Stmt *, std::vector<Stmt *>> tensor_to_matrix_ptrs_map;
  std::unordered_map<Stmt *, Stmt *> matrix_ptr_to_tensor_map;
  for (int i = begin_location; i < end_location; i++) {
    auto stmt = block->statements[i].get();
    if (stmt->is<MatrixPtrStmt>()) {
      auto origin = stmt->as<MatrixPtrStmt>()->origin;
      if (tensor_to_matrix_ptrs_map.count(origin) == 0) {
        tensor_to_matrix_ptrs_map[origin] = {stmt};
      } else {
        tensor_to_matrix_ptrs_map[origin].push_back(stmt);
      }
      matrix_ptr_to_tensor_map[stmt] = origin;
    }
  }

  // Reverse order traversal, starting from the last IR to the first IR
  for (int i = end_location - 1; i >= begin_location; i--) {
    auto stmt = block->statements[i].get();
    if (stmt->is<FuncCallStmt>() || stmt->is<ChildLaunchStmt>()) {
      // Opaque barrier: a function call / nested child-kernel launch may read/write any global memory, so it kills
      // all dead-store and store-to-load reasoning that would otherwise cross it.
      killed_in_this_node.clear();
      live_load_in_this_node.clear();
      continue;
    }
    auto store_ptrs = irpass::analysis::get_store_destination(stmt);

    // TODO: Consider AD-stacks in get_store_destination instead of here
    //  for store-to-load forwarding on AD-stacks
    if (auto stack_pop = stmt->cast<AdStackPopStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_pop->stack);
    } else if (auto stack_push = stmt->cast<AdStackPushStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_push->stack);
    } else if (auto stack_acc_adj = stmt->cast<AdStackAccAdjointStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stack_acc_adj->stack);
    } else if (stmt->is<AdStackAllocaStmt>()) {
      store_ptrs = std::vector<Stmt *>(1, stmt);
    }

    if (store_ptrs.size() == 1) {
      // Dead store elimination
      auto store_ptr = *store_ptrs.begin();

      if (!after_lower_access ||
          (store_ptr->is<MatrixPtrStmt>() && store_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (store_ptr->is<MatrixPtrStmt>() && store_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (store_ptr->is<AllocaStmt>() || store_ptr->is<AdStackAllocaStmt>())) {
        // !may_contain_variable(live_in_this_node, store_ptr): address is not
        //      loaded after this store
        // contain_variable(killed_in_this_node, store_ptr): address is already
        //      stored by another store stmt in this node (thus killed)
        // !may_contain_variable(live_out, store_ptr): address is not used
        //      in the next nodes
        bool is_used_in_next_nodes = false;
        for (auto ptr : irpass::analysis::include_aliased_stmts(store_ptr)) {
          is_used_in_next_nodes |= may_contain_variable(live_out, ptr);
        }

        bool is_killed_in_current_node = contain_variable(killed_in_this_node, store_ptr);
        bool is_dead = is_killed_in_current_node || !is_used_in_next_nodes;
        is_dead &= !may_contain_variable(live_in_this_node, store_ptr);
        if (!stmt->is<AllocaStmt>() && !stmt->is<AdStackAllocaStmt>() && !stmt->is<ExternalFuncCallStmt>() && is_dead) {
          // If an address is neither used in this node, nor used in the next
          // nodes, then we can consider eliminating any stores to this address
          // (it's not used anyway). There's two different scenerios though:
          // 1. Any direct store stmt can be eliminated immediately (LocalStore,
          //    GlobalStore, AdStackPush, ...)
          // 2. AtomicStmt (load + store): remove the store part, thus
          //    converting the AtomicStmt into a LoadStmt
          if (!stmt->is<AtomicOpStmt>()) {
            // Eliminate the dead store.
            erase(i);
            modified = true;
            continue;
          }
          auto atomic = stmt->cast<AtomicOpStmt>();
          // Weaken the atomic operation to a load.
          if (atomic->dest->is<AllocaStmt>()) {
            auto local_load = Stmt::make<LocalLoadStmt>(atomic->dest);
            local_load->ret_type = atomic->ret_type;
            // Notice that we have a load here
            // (the return value of AtomicOpStmt).
            update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, live_in_this_node,
                                        atomic->dest, false);
            update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, killed_in_this_node,
                                        atomic->dest, true);
            live_load_in_this_node[atomic->dest] = local_load.get();

            replace_with(i, std::move(local_load), true);
            modified = true;
            continue;
          } else if (!is_parallel_executed ||
                     (atomic->dest->is<GlobalPtrStmt>() && atomic->dest->as<GlobalPtrStmt>()->snode->is_scalar())) {
            // If this node is parallel executed, we can't weaken a global
            // atomic operation to a global load.
            // TODO: we can weaken it if it's element-wise (i.e. never
            //  accessed by other threads).
            auto global_load = Stmt::make<GlobalLoadStmt>(atomic->dest);
            global_load->ret_type = atomic->ret_type;
            // Notice that we have a load here
            // (the return value of AtomicOpStmt).
            update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, live_in_this_node,
                                        atomic->dest, false);
            update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, killed_in_this_node,
                                        atomic->dest, true);
            live_load_in_this_node[atomic->dest] = global_load.get();

            replace_with(i, std::move(global_load), true);
            modified = true;
            continue;
          }
        } else {
          // A non-eliminated store.
          // Insert to killed_in_this_node if it's stored in this node.
          update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, killed_in_this_node,
                                      store_ptr, false);

          // Remove the address from live_in_this_node if it's stored in this
          // node.
          auto old_live_in_this_node = live_in_this_node;
          for (auto &var : old_live_in_this_node) {
            if (irpass::analysis::definitely_same_address(store_ptr, var.first)) {
              update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, live_in_this_node,
                                          store_ptr, true);
            }
          }
        }
      }
    }
    auto load_ptrs = irpass::analysis::get_load_pointers(stmt);
    if (load_ptrs.size() == 1 && store_ptrs.empty()) {
      // Identical load elimination
      auto load_ptr = load_ptrs.begin()[0];

      if (!after_lower_access ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
        // live_load_in_this_node[addr]: tracks the
        //        next load to the same address
        // "!may_contain_variable(killed_in_this_node, load_ptr)": means it's
        //        not been stored in between the two loads
        if (live_load_in_this_node.find(load_ptr) != live_load_in_this_node.end() &&
            !may_contain_variable(killed_in_this_node, load_ptr)) {
          // Only perform identical load elimination within a CFGNode.
          auto next_load_stmt = live_load_in_this_node[load_ptr];
          if (irpass::analysis::same_statements(stmt, next_load_stmt)) {
            next_load_stmt->replace_usages_with(stmt);
            erase(block->locate(next_load_stmt));
            modified = true;
          }
        }

        update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, killed_in_this_node, load_ptr,
                                    true);
        live_load_in_this_node[load_ptr] = stmt;
      }
    }

    // Update live_in_this_node
    for (auto &load_ptr : load_ptrs) {
      if (!after_lower_access ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (load_ptr->is<MatrixPtrStmt>() && load_ptr->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (load_ptr->is<AllocaStmt>() || load_ptr->is<AdStackAllocaStmt>())) {
        // Addr is used in this node, so it's live in this node
        update_container_with_alias(tensor_to_matrix_ptrs_map, matrix_ptr_to_tensor_map, live_in_this_node, load_ptr,
                                    false);
      }
    }
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

void ControlFlowGraph::dump_graph_to_file(const CompileConfig &config,
                                          const std::string &kernel_name,
                                          const std::string &suffix) const {
  std::filesystem::path ir_dump_dir = config.debug_dump_path;
  std::filesystem::create_directories(ir_dump_dir);
  std::filesystem::path filename = ir_dump_dir / (kernel_name + "_CFG" + suffix + ".txt");

  std::ofstream out_file(filename.string());
  if (!out_file) {
    QD_WARN("Failed to open file for CFG dump: {}", filename.string());
    return;
  }

  // Write directly to the file using fmt::format
  const int num_nodes = size();
  std::unordered_map<CFGNode *, int> to_index;
  for (int i = 0; i < num_nodes; i++) {
    to_index[nodes[i].get()] = i;
  }

  for (int i = 0; i < num_nodes; i++) {
    out_file << fmt::format("Node {} : ", i);
    if (nodes[i]->empty()) {
      out_file << "empty";
    } else {
      out_file << fmt::format("{}~{} (size={})", nodes[i]->block->statements[nodes[i]->begin_location]->name(),
                              nodes[i]->block->statements[nodes[i]->end_location - 1]->name(), nodes[i]->size());
    }
    if (!nodes[i]->prev.empty()) {
      std::vector<std::string> indices;
      for (auto prev_node : nodes[i]->prev) {
        indices.push_back(std::to_string(to_index[prev_node]));
      }
      out_file << fmt::format("; prev={{{}}}", fmt::join(indices, ", "));
    }
    if (!nodes[i]->next.empty()) {
      std::vector<std::string> indices;
      for (auto next_node : nodes[i]->next) {
        indices.push_back(std::to_string(to_index[next_node]));
      }
      out_file << fmt::format("; next={{{}}}", fmt::join(indices, ", "));
    }
    if (!nodes[i]->live_out.empty()) {
      std::vector<std::string> vars;
      for (auto stmt : nodes[i]->live_out) {
        vars.push_back(stmt->name());
      }
      out_file << fmt::format("; live_out={{{}}}", fmt::join(vars, ", "));
    }
    out_file << "\n";

    // Print the actual statements in this node
    if (!nodes[i]->empty()) {
      for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
        auto stmt = nodes[i]->block->statements[j].get();
        std::string stmt_output;
        // Use print_kernel_wrapper=false to avoid the "kernel { }" wrapper
        irpass::print(stmt, &stmt_output, false, false);

        // Add indentation to each line
        std::istringstream iss(stmt_output);
        std::string line;
        while (std::getline(iss, line)) {
          if (!line.empty()) {
            out_file << "    " << line << "\n";
          }
        }
      }
      out_file << "\n";
    }
  }

  out_file.close();
  QD_INFO("CFG dumped to: {}", filename.string());
}

void ControlFlowGraph::reaching_definition_analysis(bool after_lower_access) {
  // Prerequisite analysis for load-store-forwarding to help determine
  // cross-block use-define chain
  //
  // The algorithm is separated into two parts:
  // 1. Determine reach_gen and reach_kill within each node
  // 2. Propagate reach_in and reach_out through the graph
  //
  // - reach_gen: instruction that defines a variable (store stmts) in the
  // current node
  // - reach_kill: address (GlobalPtrStmt, AllocaStmt, ...) that's been defined
  // (stored to) in the current node
  //
  // In general, reach_gen and reach_kill are the same except that reach_gen
  // tracks the store stmts and reach_kill tracks the address
  //
  // - reach_out: reach_gen + { reach_in's dest not in reach_kill }
  // - reach_in: collection of all the reach_out of previous nodes
  //
  // reach_out and reach_in is the ultimate result that helps analyze
  // cross-block use-define chain

  QD_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  QD_ASSERT(nodes[start_node]->empty());
  nodes[start_node]->reach_gen.clear();
  nodes[start_node]->reach_kill.clear();
  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      auto stmt = nodes[i]->block->statements[j].get();
      if ((stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<AllocaStmt>()) ||
          (stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<MatrixPtrStmt>()) ||
          (!after_lower_access &&
           (stmt->is<GlobalPtrStmt>() || stmt->is<ExternalPtrStmt>() || stmt->is<BlockLocalPtrStmt>() ||
            stmt->is<ThreadLocalPtrStmt>() || stmt->is<GlobalTemporaryStmt>() || stmt->is<MatrixPtrStmt>() ||
            stmt->is<GetChStmt>() || stmt->is<MatrixOfGlobalPtrStmt>() || stmt->is<MatrixOfMatrixPtrStmt>()))) {
        // TODO: unify them
        // A global pointer that may contain some data before this kernel.
        nodes[start_node]->reach_gen.insert(stmt);
      } else if (auto func_call = stmt->cast<FuncCallStmt>()) {
        const auto &dests = func_call->func->store_dests;
        nodes[start_node]->reach_gen.insert(dests.begin(), dests.end());
      }
    }
  }

  for (int i = 0; i < num_nodes; i++) {
    if (i != start_node) {
      nodes[i]->reaching_definition_analysis(after_lower_access);
    }
    nodes[i]->reach_in.clear();
    nodes[i]->reach_out = nodes[i]->reach_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }

  // [The worklist algorithm]
  // Determines reach_in and reach_out for each node iteratively.
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->reach_in.clear();
    for (auto prev_node : now->prev) {
      now->reach_in.insert(prev_node->reach_out.begin(), prev_node->reach_out.end());
    }
    auto old_out = std::move(now->reach_out);
    now->reach_out = now->reach_gen;
    for (auto stmt : now->reach_in) {
      auto store_ptrs = irpass::analysis::get_store_destination(stmt);
      bool killed;
      if (store_ptrs.empty()) {  // the case of a global pointer
        killed = now->reach_kill_variable(stmt);
      } else {
        killed = true;
        for (auto store_ptr : store_ptrs) {
          if (!now->reach_kill_variable(store_ptr)) {
            killed = false;
            break;
          }
        }
      }
      if (!killed) {
        now->reach_out.insert(stmt);
      }
    }
    if (now->reach_out != old_out) {
      // changed
      for (auto next_node : now->next) {
        if (!in_queue[next_node]) {
          to_visit.push(next_node);
          in_queue[next_node] = true;
        }
      }
    }
  }
}

void ControlFlowGraph::live_variable_analysis(bool after_lower_access,
                                              const std::optional<LiveVarAnalysisConfig> &config_opt) {
  // [live_variable_analysis]
  // live_gen: address loaded with no previous stored in this node. One cannot
  // load before storing so
  //           addrs in live_gen must come from previous nodes
  // live_kill: address stored in this node
  // live_in: live_gen + (live_out - live_kill)
  // live_out: collection of all the live_in of next nodes
  QD_AUTO_PROF;
  const int num_nodes = size();
  std::queue<CFGNode *> to_visit;
  std::unordered_map<CFGNode *, bool> in_queue;
  QD_ASSERT(nodes[final_node]->empty());
  nodes[final_node]->live_gen.clear();
  nodes[final_node]->live_kill.clear();

  auto in_final_node_live_gen = [&config_opt](const Stmt *stmt) -> bool {
    if (stmt->is<AllocaStmt>() || stmt->is<AdStackAllocaStmt>()) {
      return false;
    }
    if (stmt->is<MatrixPtrStmt>() && stmt->cast<MatrixPtrStmt>()->origin->is<AllocaStmt>()) {
      return false;
    }
    if (auto *gptr = stmt->cast<GlobalPtrStmt>(); gptr && config_opt.has_value()) {
      const bool res = (config_opt->eliminable_snodes.count(gptr->snode) == 0);
      return res;
    }
    // A global pointer that may be loaded after this kernel.
    return true;
  };
  if (!after_lower_access) {
    for (int i = 0; i < num_nodes; i++) {
      for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
        auto stmt = nodes[i]->block->statements[j].get();
        for (auto store_ptr : irpass::analysis::get_store_destination(stmt, true /*get_alias*/)) {
          if (in_final_node_live_gen(store_ptr)) {
            nodes[final_node]->live_gen.insert(store_ptr);
          }
        }
      }
    }
  }

  for (int i = num_nodes - 1; i >= 0; i--) {
    // push into the queue in reversed order to make it slightly faster
    if (i != final_node) {
      nodes[i]->live_variable_analysis(after_lower_access);
    }
    nodes[i]->live_out.clear();
    nodes[i]->live_in = nodes[i]->live_gen;
    to_visit.push(nodes[i].get());
    in_queue[nodes[i].get()] = true;
  }

  // The worklist algorithm.
  while (!to_visit.empty()) {
    auto now = to_visit.front();
    to_visit.pop();
    in_queue[now] = false;

    now->live_out.clear();
    for (auto next_node : now->next) {
      now->live_out.insert(next_node->live_in.begin(), next_node->live_in.end());
    }
    auto old_in = std::move(now->live_in);
    now->live_in = now->live_gen;
    for (auto stmt : now->live_out) {
      if (!CFGNode::contain_variable(now->live_kill, stmt)) {
        now->live_in.insert(stmt);
      }
    }
    if (now->live_in != old_in) {
      // changed
      for (auto prev_node : now->prev) {
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
  const int num_nodes = size();

  // Map AdStackAllocaStmt* to a contiguous int index. Only stacks whose `max_size` is still 0 (i.e. unresolved by an
  // earlier pass) participate in the DP; resolved stacks are skipped so the pass cannot clobber them.
  std::unordered_map<AdStackAllocaStmt *, int> stack_id;
  std::vector<AdStackAllocaStmt *> stacks;

  std::unordered_map<CFGNode *, int> node_ids;
  for (int i = 0; i < num_nodes; i++)
    node_ids[nodes[i].get()] = i;

  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      Stmt *stmt = nodes[i]->block->statements[j].get();
      if (auto *stack = stmt->cast<AdStackAllocaStmt>()) {
        if (stack->max_size != 0) {
          continue;
        }
        if (stack_id.emplace(stack, static_cast<int>(stacks.size())).second) {
          stacks.push_back(stack);
        }
      }
    }
  }

  const int num_stacks = static_cast<int>(stacks.size());
  if (num_stacks == 0) {
    return;
  }

  // max_increased_size[s][j] is the maximum number of (pushes - pops) of stack |s| among all prefixes of the
  // CFGNode |j|. increased_size[s][j] is the net (pushes - pops) of stack |s| in the CFGNode |j|. Both are indexed
  // by contiguous stack id, so the per-stack DP loop reads them with cheap vector access.
  std::vector<std::vector<int>> max_increased_size(num_stacks, std::vector<int>(num_nodes, 0));
  std::vector<std::vector<int>> increased_size(num_stacks, std::vector<int>(num_nodes, 0));

  // Track which stacks actually participate in any push/pop in the CFG. Stacks with no push/pop end with
  // `max_size = 0` regardless of the DP (every node has zero increase), so we skip the DP for them and reproduce
  // the original "Unused autodiff stack" warning here.
  std::vector<bool> stack_active(num_stacks, false);

  for (int i = 0; i < num_nodes; i++) {
    for (int j = nodes[i]->begin_location; j < nodes[i]->end_location; j++) {
      Stmt *stmt = nodes[i]->block->statements[j].get();
      if (auto *stack_push = stmt->cast<AdStackPushStmt>()) {
        auto *stack = stack_push->stack->as<AdStackAllocaStmt>();
        if (stack->max_size == 0 /*adaptive*/) {
          auto it = stack_id.find(stack);
          QD_ASSERT(it != stack_id.end());
          const int sid = it->second;
          stack_active[sid] = true;
          int &cur = increased_size[sid][i];
          cur++;
          if (cur > max_increased_size[sid][i]) {
            max_increased_size[sid][i] = cur;
          }
        }
      } else if (auto *stack_pop = stmt->cast<AdStackPopStmt>()) {
        auto *stack = stack_pop->stack->as<AdStackAllocaStmt>();
        if (stack->max_size == 0 /*adaptive*/) {
          auto it = stack_id.find(stack);
          QD_ASSERT(it != stack_id.end());
          const int sid = it->second;
          stack_active[sid] = true;
          increased_size[sid][i]--;
        }
      }
    }
  }

  // Precompute outgoing-edge node ids once per node so the per-stack DP walks an int vector instead of hashing
  // `node_ids[next_node]` on every traversal.
  std::vector<std::vector<int>> next_ids(num_nodes);
  for (int i = 0; i < num_nodes; i++) {
    auto &dst = next_ids[i];
    dst.reserve(nodes[i]->next.size());
    for (auto *next_node : nodes[i]->next) {
      dst.push_back(node_ids[next_node]);
    }
  }

  // Tarjan strongly-connected-component (SCC) decomposition of the CFG, computed once and shared across all per-stack
  // dynamic-programming runs. Output:
  //   scc_id[n]    : SCC index of each node (lower index = topologically deeper / sink-side, so sources are at index
  //                  num_sccs - 1, matching Tarjan's natural emission order which is reverse topological order).
  //   scc_nodes[s] : list of node ids in SCC s.
  //   dfs_finish[n]: DFS post-order index, used below to split each cyclic SCC's intra-edges into a forward and a
  //                  back set without a separate edge classification pass.
  // We then split each node's outgoing edges into intra-SCC and inter-SCC sets so the per-stack DP iterates each
  // set without filtering inside the hot loop. Cyclic SCCs (|S| > 1 or |S| == 1 with self-loop) are flagged so
  // cycle detection runs only on those, and only at SCC scope.
  std::vector<int> scc_id(num_nodes, -1);
  std::vector<int> dfs_finish(num_nodes, -1);  // DFS post-order index per node; ancestors finish AFTER descendants.
  std::vector<std::vector<int>> scc_nodes;
  {
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
              scc_id[w] = next_scc;
              component.push_back(w);
              if (w == u) {
                break;
              }
            }
            scc_nodes.push_back(std::move(component));
            next_scc++;
          }
          dfs_finish[u] = next_finish++;
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
  }
  const int num_sccs = static_cast<int>(scc_nodes.size());

  // Each intra-SCC edge is either a "forward" edge in the DFS spanning sense (source finishes after target, so source
  // can be processed before target with a single topological pass) or a "back" edge (source finishes before target,
  // closing a cycle). The forward set carries the per-stack DAG dynamic programming; back-edges only need a single
  // post-DP relaxation check to detect positive cycles. Inter-SCC edges always relax forward in topological order.
  std::vector<std::vector<int>> next_ids_intra_fwd(num_nodes);
  std::vector<std::vector<int>> next_ids_intra_back(num_nodes);
  std::vector<std::vector<int>> next_ids_inter(num_nodes);
  for (int u = 0; u < num_nodes; u++) {
    const int su = scc_id[u];
    for (int v : next_ids[u]) {
      if (scc_id[v] == su) {
        if (dfs_finish[v] < dfs_finish[u]) {
          next_ids_intra_fwd[u].push_back(v);
        } else {
          next_ids_intra_back[u].push_back(v);
        }
      } else {
        next_ids_inter[u].push_back(v);
      }
    }
  }

  // An SCC is cyclic iff it contains a cycle. By definition that is |S| > 1 (any two nodes lie on a cycle since the
  // SCC is strongly connected) or |S| == 1 with a self-loop edge. For cyclic SCCs we also precompute a topological
  // ordering of the SCC's nodes (descending DFS finish time) so the per-stack DAG DP visits each node once with all
  // forward predecessors already finalized.
  std::vector<char> scc_is_cyclic(num_sccs, 0);
  std::vector<std::vector<int>> scc_topo(num_sccs);
  for (int s = 0; s < num_sccs; s++) {
    auto &nodes_in_s = scc_nodes[s];
    if (nodes_in_s.size() > 1) {
      scc_is_cyclic[s] = 1;
    } else {
      const int n = nodes_in_s[0];
      // A singleton SCC is cyclic iff it has a self-loop. Self-loops are classified as back-edges above
      // (dfs_finish[v] == dfs_finish[u]), so check there.
      for (int v : next_ids_intra_back[n]) {
        if (v == n) {
          scc_is_cyclic[s] = 1;
          break;
        }
      }
    }
    if (scc_is_cyclic[s]) {
      auto topo = nodes_in_s;
      std::sort(topo.begin(), topo.end(), [&](int a, int b) { return dfs_finish[a] > dfs_finish[b]; });
      scc_topo[s] = std::move(topo);
    }
  }

  // Group AD-stacks whose per-node (increased_size, max_increased_size) rows are bit-identical, so that the DP runs
  // once per equivalence class instead of once per stack. In practice many AD-stacks in the same kernel share their
  // push/pop schedule (one alloca per autodiff variable in the same loop body), and the DP on a large CFG dwarfs
  // the dedup cost. We hash a sparse fingerprint (only nodes where the stack actually has push/pop activity) and
  // group by it. Worst case (no duplicates): one DP run per stack, equivalent to running it directly per stack.
  using Fingerprint = std::vector<std::tuple<int, int, int>>;  // (node_id, is, mis), sorted by node_id
  struct FingerprintHash {
    std::size_t operator()(const Fingerprint &f) const noexcept {
      std::size_t h = 1469598103934665603ULL;
      for (auto &[n, i, m] : f) {
        auto mix = [&](std::size_t x) {
          h ^= x;
          h *= 1099511628211ULL;
        };
        mix(static_cast<std::size_t>(n));
        mix(static_cast<std::size_t>(static_cast<unsigned>(i)));
        mix(static_cast<std::size_t>(static_cast<unsigned>(m)));
      }
      return h;
    }
  };
  std::unordered_map<Fingerprint, int, FingerprintHash> fp_to_rep;  // fingerprint -> representative stack id
  std::vector<int> stack_to_rep(num_stacks, -1);
  std::vector<int> rep_stack_ids;  // stack ids that actually run BF
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
    stack_to_rep[sid] = it->second;
    if (inserted) {
      rep_stack_ids.push_back(sid);
    }
  }

  // Scratch buffer reused across representatives to avoid reallocating per iteration.
  std::vector<int> max_size_at_node_begin(num_nodes);

  // Per-representative results, broadcast below to every stack that hashed to this representative.
  struct DPResult {
    int max_size;
    bool has_positive_loop;
  };
  std::unordered_map<int, DPResult> rep_results;
  rep_results.reserve(rep_stack_ids.size());

  // For each representative stack, walk the SCC condensation in topological order (sources first, since Tarjan emits
  // SCCs in reverse-topological order so source SCCs end up at the largest indices). Inter-SCC edges only relax
  // forward (predecessors are already finalized when an SCC is entered), so each is touched O(1) times per stack.
  for (int rep_sid : rep_stack_ids) {
    const std::vector<int> &mis_for_stack = max_increased_size[rep_sid];
    const std::vector<int> &is_for_stack = increased_size[rep_sid];

    std::fill(max_size_at_node_begin.begin(), max_size_at_node_begin.end(), -1);

    int max_size = 0;
    max_size_at_node_begin[start_node] = 0;

    bool has_positive_loop = false;

    for (int s = num_sccs - 1; s >= 0 && !has_positive_loop; s--) {
      const auto &nodes_in_s = scc_nodes[s];

      if (scc_is_cyclic[s]) {
        // Sign-based fast paths sidestep the cyclic-SCC dynamic programming when the stack's `is` contribution
        // inside this SCC is structurally trivial. Two cases short-circuit:
        //   1. min_is >= 0 with max_is > 0: every node in the SCC lies on some cycle (SCC property), and a cycle
        //      through a strictly-positive node with all non-negative `is` along it sums to a positive value, so a
        //      positive cycle exists for this stack. This is the autodiff push-only-in-SCC pattern.
        //   2. min_is == 0 == max_is (no `is` contribution in this SCC at all): the DP would only spread the maximum
        //      entry-side `max_size_at_node_begin` value to every node in S; we do that directly in O(|S|).
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
          has_positive_loop = true;
          break;
        }
        if (min_is == 0 && max_is == 0) {
          int max_begin = -1;
          for (int u : nodes_in_s) {
            if (max_size_at_node_begin[u] > max_begin) {
              max_begin = max_size_at_node_begin[u];
            }
          }
          if (max_begin >= 0) {
            for (int u : nodes_in_s) {
              if (max_size_at_node_begin[u] < max_begin) {
                max_size_at_node_begin[u] = max_begin;
              }
            }
          }
        } else {
          // Mixed-sign case: single-pass dynamic programming on the SCC's forward edges (processed in descending DFS
          // finish-time so every forward predecessor is finalized before its successor relaxes), followed by one
          // relaxation check on the back-edges. Correctness: every walk inside the SCC decomposes into a forward
          // path (along non-back edges) plus zero or more closed cycles formed by a back-edge plus a forward path.
          // For SCCs with no positive cycle, traversing a cycle adds a non-positive amount to the running size and
          // so cannot improve max_size beyond what the forward DP already computed. The back-edge relaxation check
          // after the DP detects the only failure mode (some back-edge would still improve a forward predecessor's
          // value, which can only happen if a positive cycle exists for this stack).
          for (int u : scc_topo[s]) {
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
                has_positive_loop = true;
                break;
              }
            }
            if (has_positive_loop) {
              break;
            }
          }
          if (has_positive_loop) {
            break;
          }
        }
      }

      // SCC has converged for this stack. Update global max_size from each node's max-prefix contribution and relax
      // inter-SCC outgoing edges into successor SCCs.
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

    rep_results[rep_sid] = {max_size, has_positive_loop};
  }

  // Broadcast representative results to every active stack.
  for (int sid = 0; sid < num_stacks; sid++) {
    AdStackAllocaStmt *stack = stacks[sid];
    if (!stack_active[sid]) {
      // No push/pop in the CFG: the DP would visit reachable nodes with all-zero edge weights and settle with
      // `max_size = 0`, no positive loop. Reproduce that result directly.
      QD_WARN("Unused autodiff stack {} should have been eliminated.", stack->name());
      continue;
    }
    const DPResult &res = rep_results[stack_to_rep[sid]];
    if (res.has_positive_loop) {
      // Leave `max_size = 0` so the structural bounded-loop pre-pass in `irpass::determine_ad_stack_size` gets a
      // chance to derive a symbolic bound (a statically-bounded inner loop whose push-only body defeats the
      // longest-path computation, resolved from the outer ranges). If it also cannot, the caller emits a hard
      // compile error - there is no compile-time `default_ad_stack_size` fallback.
    } else {
      // Since we use |max_size| == 0 for adaptive sizes, we do not want stacks with maximum capacity indeed equal
      // to 0.
      QD_WARN_IF(res.max_size == 0, "Unused autodiff stack {} should have been eliminated.", stack->name());
      stack->max_size = res.max_size;
    }
  }
}

}  // namespace quadrants::lang
