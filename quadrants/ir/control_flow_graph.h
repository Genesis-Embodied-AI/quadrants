#pragma once

#include <optional>
#include <unordered_set>

#include "quadrants/ir/ir.h"

namespace quadrants::lang {

class Function;
/**
 * A basic block in the control-flow graph (one node).
 *
 * A CFGNode references an interval of statements in a Block:
 * `block->statements[i]` for `i in [begin_location, end_location)`.
 * The graph edges are stored in `prev` and `next`: control may flow from any
 * node in `prev` into this node, and out of this node into any node in `next`.
 *
 * Scope of the methods on this class: each analysis/transform method on
 * CFGNode does only the *intra-block* work for that pass (e.g.
 * `reaching_definition_analysis` computes `reach_gen` / `reach_kill` from this
 * node's statements; `store_to_load_forwarding` rewrites loads/stores within
 * this node's slice of the block). The same-named method on
 * `ControlFlowGraph` is the *whole-graph driver*: it calls this per-node
 * method on every node, then runs the worklist fixpoint or post-processing
 * that needs cross-node state.
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
  // For accelerating find_forwardable_store_value()
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
  bool is_reach_killed(Stmt *var) const;
  Stmt *find_forwardable_store_value(Stmt *var, int position) const;

  // Per-node (intra-block) analyses and transforms. Each is driven across the
  // whole graph by the same-named method on ControlFlowGraph; see below.
  // Per-node `reaching_definition_analysis`: populate this node's reach_gen /
  // reach_kill from its statements.
  void reaching_definition_analysis(bool after_lower_access);
  // Per-node `store_to_load_forwarding`: rewrite loads/stores within this
  // node's statement range. Returns true if any IR change was made.
  bool store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled);
  // Per-node helper for `ControlFlowGraph::gather_loaded_snodes`: append this
  // node's loaded SNodes into `snodes`.
  void gather_loaded_snodes(std::unordered_set<SNode *> &snodes) const;
  // Per-node `live_variable_analysis`: populate this node's live_gen /
  // live_kill from its statements.
  void live_variable_analysis(bool after_lower_access);
  // Per-node `dead_store_elimination`: erase dead stores / weaken dead
  // atomics within this node's statement range. Returns true if any IR change
  // was made.
  bool dead_store_elimination(bool after_lower_access);

 private:
  // Helper for find_forwardable_store_value: is |stmt| visible at |position|
  // inside this node's block? A stmt is visible if it lives in the same block
  // and precedes |position|, or if its parent block is an ancestor of
  // |this->block|.
  bool is_visible_at(Stmt *stmt, int position) const;

  // Helper for find_forwardable_store_value: incorporate |stmt|, a definition in
  // the UD-chain of the variable being forwarded, into the running |result| /
  // |result_visible| state. Returns false to signal that forwarding must
  // abort (the caller should return nullptr), true to continue scanning.
  bool fold_definition_into_result(Stmt *stmt,
                                   int position,
                                   Stmt *&result,
                                   bool &result_visible) const;

  // Helper for find_forwardable_store_value: walk this node's block backwards
  // from |position| and return the index of the most recent store to |var|,
  // or -1 if none is in this block. Handles the quant-store exclusion plus the
  // MatrixInitStmt-via-MatrixPtrStmt forwarding special case.
  int find_intra_block_last_def(Stmt *var, int position) const;

  // Helper for find_forwardable_store_value: scan |reach_in| and |reach_gen| for
  // definitions of |var| reaching |position|, folding each into |result| /
  // |result_visible| via fold_definition_into_result. Returns nullopt if any
  // visited def is unforwardable (caller must return nullptr); otherwise the
  // last_def_position (0 if only reach_in matched, an in-block index if
  // reach_gen matched, -1 if no eligible def was found).
  std::optional<int> find_cross_block_def(Stmt *var,
                                          int position,
                                          Stmt *&result,
                                          bool &result_visible) const;

  // Helper for find_forwardable_store_value: scan block statements in
  // [from, to_exclusive) for a store that may write a different value to an
  // address aliasing |var|. Returns true iff such a store exists (so
  // forwarding |result| must abort). The check is skipped (returns false) for
  // non-tensor alloca destinations, where aliasing through MatrixPtrStmt
  // cannot apply.
  bool any_aliased_store_breaks_forwarding(Stmt *result, Stmt *var, int from, int to_exclusive) const;

  // Helper for store_to_load_forwarding: if |stmt| is a load whose source has
  // a forwardable preceding store, replace it. Returns true iff the load was
  // handled (the caller must skip identical-store elimination for this stmt
  // even if no IR change happened, matching the legacy fall-through). |i| is
  // decremented on erase so the for-loop's natural `i++` lands on the next
  // unread stmt; |modified| is flipped on erase only (preserved from legacy:
  // replace-with-zero does not flip it).
  bool try_forward_load_at(int &i, Stmt *stmt, bool after_lower_access, bool autodiff_enabled, bool &modified);

  // Helper for store_to_load_forwarding: if |stmt| is a store identical to a
  // preceding store to the same address, erase it. Handles the alloca-init-0
  // special case under non-autodiff. Same |i|/|modified| accounting as
  // try_forward_load_at.
  void try_eliminate_identical_store_at(int &i,
                                        Stmt *stmt,
                                        bool after_lower_access,
                                        bool autodiff_enabled,
                                        bool &modified);
};

/**
 * The whole control-flow graph (all nodes plus their edges).
 *
 * Owns the `nodes` vector, the synthetic entry node (`start_node`, always
 * empty), and the synthetic exit node (`final_node`, always empty). Each
 * analysis/transform method on this class is the *whole-graph driver*: it
 * invokes the same-named per-node method on `CFGNode` for every node, then
 * runs the worklist fixpoint (RD / LV) or per-node IR rewrite loop (S2L /
 * DSE). `determine_ad_stack_size` has no per-node counterpart.
 */
class ControlFlowGraph {
 private:
  // Erase an empty node.
  void erase(int node_id);

  // Assert structural invariants that every whole-graph driver assumes: `nodes` non-empty,
  // `start_node` and `final_node` in range, both endpoints actually allocated. Called from each
  // public driver. Cheap (a handful of comparisons); leave on even in release builds since the
  // alternative on violation is silent corruption.
  void assert_structural_invariants() const;

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
   * Whole-graph driver: reaching-definition analysis via worklist fixpoint.
   * Seeds the entry node's reach_gen with external-input pointers, calls
   * `CFGNode::reaching_definition_analysis` per node, then converges
   * reach_in/reach_out. Results are stored on each CFGNode.
   * https://en.wikipedia.org/wiki/Reaching_definition
   *
   * @param after_lower_access
   *   When true, only consider local variables (allocas).
   */
  void reaching_definition_analysis(bool after_lower_access);

  /**
   * Whole-graph driver: live-variable analysis via worklist fixpoint (run
   * backwards). Seeds the exit node's live_gen with kernel-escaping stores,
   * calls `CFGNode::live_variable_analysis` per node, then converges
   * live_in/live_out. Results are stored on each CFGNode.
   * https://en.wikipedia.org/wiki/Live_variable_analysis
   *
   * @param after_lower_access
   *   When true, only consider local variables (allocas).
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
   * Whole-graph driver: store-to-load forwarding + identical-store elimination.
   * Calls `CFGNode::store_to_load_forwarding` on every node and ORs the
   * per-node "did we change the IR" return. Caller is responsible for
   * (re)running `reaching_definition_analysis` first to populate reach_in /
   * reach_out.
   */
  bool store_to_load_forwarding(bool after_lower_access, bool autodiff_enabled);

  /**
   * Whole-graph driver: dead-store elimination + identical-load elimination.
   * Calls `CFGNode::dead_store_elimination` on every node and ORs the
   * per-node return. Caller is responsible for (re)running
   * `live_variable_analysis` first to populate live_in / live_out.
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
