#pragma once

#include <optional>
#include <unordered_set>

#include "quadrants/ir/ir.h"

namespace quadrants::lang {

class Function;
/**
 * A basic block in control-flow graph.
 * A CFGNode contains a reference to a part of the CHI IR, or more precisely,
 * an interval of statements in a Block.
 * The edges in the graph are stored in |prev| and |next|. The control flow is
 * possible to go from any node in |prev| to this node, and is possible to go
 * from this node to any node in |next|.
 */
class CFGNode {
 public:
  // Used for TensorType'd aliasing analysis.
  // Marks whether a TensorType'd address is modified partially or
  // fully in this node
  enum class UseDefineStatus {
    FULL = 0,
    PARTIAL = 1,
    NONE = 2,
  };

 private:
  // For accelerating get_store_forwarding_data()
  std::unordered_set<Block *> parent_blocks_;

 public:
  // This node corresponds to block->statements[i]
  // for i in [begin_location, end_location).
  Block *block;
  int begin_location, end_location;
  // Is this node in an offloaded range_for/struct_for?
  bool is_parallel_executed;

  // For updating begin/end locations when modifying the block.
  CFGNode *prev_node_in_same_block;
  CFGNode *next_node_in_same_block;

  // Edges in the graph
  std::vector<CFGNode *> prev, next;

  // Reaching definition analysis
  // https://en.wikipedia.org/wiki/Reaching_definition
  std::unordered_set<Stmt *> reach_gen, reach_kill, reach_in, reach_out;

  // Live variable analysis
  // https://en.wikipedia.org/wiki/Live_variable_analysis
  std::unordered_set<Stmt *> live_gen, live_kill, live_in, live_out;

  CFGNode(Block *block,
          int begin_location,
          int end_location,
          bool is_parallel_executed,
          CFGNode *prev_node_in_same_block);

  // An empty node
  CFGNode();

  static void add_edge(CFGNode *from, CFGNode *to);

  // Property methods.
  bool empty() const;
  std::size_t size() const;

  // Methods for modifying the underlying CHI IR.
  void erase(int location);
  void insert(std::unique_ptr<Stmt> &&new_stmt, int location);
  void replace_with(int location, std::unique_ptr<Stmt> &&new_stmt, bool replace_usages = true) const;

  // Utility methods.
  static bool contain_variable(const std::unordered_set<Stmt *> &var_set, Stmt *var);
  static bool contain_variable(const std::unordered_map<Stmt *, UseDefineStatus> &var_set, Stmt *var);
  static bool may_contain_variable(const std::unordered_set<Stmt *> &var_set, Stmt *var);
  static bool may_contain_variable(const std::unordered_map<Stmt *, UseDefineStatus> &var_set, Stmt *var);
  bool reach_kill_variable(Stmt *var) const;
  Stmt *get_store_forwarding_data(Stmt *var, int position) const;

  // Analyses and optimizations inside a CFGNode.
  void reaching_definition_analysis(bool after_lower_access);
  bool store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled);
  void gather_loaded_snodes(std::unordered_set<SNode *> &snodes) const;
  void live_variable_analysis(bool after_lower_access);
  bool dead_store_elimination(bool after_lower_access);

 private:
  // Helper for get_store_forwarding_data: is |stmt| visible at |position|
  // inside this node's block? A stmt is visible if it lives in the same block
  // and precedes |position|, or if its parent block is an ancestor of
  // |this->block|.
  bool is_visible_at(Stmt *stmt, int position) const;

  // Helper for get_store_forwarding_data: incorporate |stmt|, a definition in
  // the UD-chain of the variable being forwarded, into the running |result| /
  // |result_visible| state. Returns false to signal that forwarding must
  // abort (the caller should return nullptr), true to continue scanning.
  bool update_forwarding_result(Stmt *stmt,
                                int position,
                                Stmt *&result,
                                bool &result_visible) const;

  // Helper for get_store_forwarding_data: walk this node's block backwards
  // from |position| and return the index of the most recent store to |var|,
  // or -1 if none is in this block. Handles the quant-store exclusion plus the
  // MatrixInitStmt-via-MatrixPtrStmt forwarding special case.
  int find_intra_block_last_def(Stmt *var, int position) const;

  // Helper for get_store_forwarding_data: scan |reach_in| and |reach_gen| for
  // definitions of |var| reaching |position|, folding each into |result| /
  // |result_visible| via update_forwarding_result. Returns nullopt if any
  // visited def is unforwardable (caller must return nullptr); otherwise the
  // last_def_position (0 if only reach_in matched, an in-block index if
  // reach_gen matched, -1 if no eligible def was found).
  std::optional<int> find_cross_block_def(Stmt *var,
                                          int position,
                                          Stmt *&result,
                                          bool &result_visible) const;

  // Helper for get_store_forwarding_data: scan block statements in
  // [from, to_exclusive) for a store that may write a different value to an
  // address aliasing |var|. Returns true iff such a store exists (so
  // forwarding |result| must abort). The check is skipped (returns false) for
  // non-tensor alloca destinations, where aliasing through MatrixPtrStmt
  // cannot apply.
  bool any_aliased_store_breaks_forwarding(Stmt *result, Stmt *var, int from, int to_exclusive) const;
};

class ControlFlowGraph {
 private:
  // Erase an empty node.
  void erase(int node_id);

 public:
  struct LiveVarAnalysisConfig {
    // This is mostly useful for SFG task-level dead store elimination. SFG may
    // detect certain cases where writes to one or more SNodes in a task are
    // eliminable.
    std::unordered_set<const SNode *> eliminable_snodes;
  };
  std::vector<std::unique_ptr<CFGNode>> nodes;
  const int start_node = 0;
  int final_node{0};

  std::unordered_map<Function *, std::unordered_set<Stmt *>> func_store_dests;

  template <typename... Args>
  CFGNode *push_back(Args &&...args) {
    nodes.emplace_back(std::make_unique<CFGNode>(std::forward<Args>(args)...));
    return nodes.back().get();
  }

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] CFGNode *back() const;

  void dump_graph_to_file(const CompileConfig &config,
                          const std::string &kernel_name,
                          const std::string &suffix = "") const;

  /**
   * Perform reaching definition analysis using the worklist algorithm,
   * and store the results in CFGNodes.
   * https://en.wikipedia.org/wiki/Reaching_definition
   *
   * @param after_lower_access
   *   When after_lower_access is true, only consider local variables (allocas).
   */
  void reaching_definition_analysis(bool after_lower_access);

  /**
   * Perform live variable analysis using the worklist algorithm,
   * and store the results in CFGNodes.
   * https://en.wikipedia.org/wiki/Live_variable_analysis
   *
   * @param after_lower_access
   *   When after_lower_access is true, only consider local variables (allocas).
   * @param config_opt
   *   The set of SNodes which is never loaded after this task.
   */
  void live_variable_analysis(bool after_lower_access, const std::optional<LiveVarAnalysisConfig> &config_opt);

  /**
   * Simplify the graph structure to accelerate other analyses and
   * optimizations. The IR is not modified.
   */
  void simplify_graph();

  // This pass cannot eliminate container statements properly for now.
  bool unreachable_code_elimination();

  /**
   * Perform store-to-load forwarding and identical store elimination.
   */
  bool store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled);

  /**
   * Perform dead store elimination and identical load elimination.
   */
  bool dead_store_elimination(bool after_lower_access, const std::optional<LiveVarAnalysisConfig> &lva_config_opt);

  /**
   * Gather the SNodes which is read or partially written in this offloaded
   * task.
   */
  std::unordered_set<SNode *> gather_loaded_snodes();

  /**
   * Determine all adaptive AD-stacks' necessary size. Stacks whose forward kernel contains a
   * positive cycle (pushes - pops > 0 around a loop) are left at `max_size = 0` so the caller can
   * route them to a symbolic `SizeExpr` via the structural bounded-loop pre-pass in
   * `irpass::determine_ad_stack_size`. There is no compile-time size fallback: any alloca the
   * grammar cannot resolve is a hard compile error, surfaced from the caller.
   */
  void determine_ad_stack_size();
};

}  // namespace quadrants::lang
