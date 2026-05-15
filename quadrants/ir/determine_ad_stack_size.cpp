#include <algorithm>
#include <climits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "quadrants/common/exceptions.h"
#include "quadrants/ir/control_flow_graph.h"
#include "quadrants/ir/statements.h"

// ===========================================================================
// Adaptive AD-stack sizing.
//
// This file implements ControlFlowGraph::determine_ad_stack_size() and its
// file-local helpers. Pulled out of control_flow_graph.cpp because it is a
// self-contained ~500-line algorithm (Tarjan SCC condensation + per-rep DAG
// DP + fingerprint dedup) that has no per-node CFGNode counterpart and is
// not used by any other pass in this file.
//
// See the docstring on ControlFlowGraph::determine_ad_stack_size() below for
// the "what problem this solves / how" overview.
// ===========================================================================

namespace quadrants::lang {

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
      if (!stack || !stack->is_adaptive()) {
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
      if (!stack->is_adaptive()) {
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
   * What problem this solves
   * ------------------------
   * Reverse-mode autodiff records each primal write to a variable by pushing the value onto a
   * per-variable stack (an "AD-stack"), then pops in reverse order during the backward pass. The
   * stack's *capacity* must be known at allocation time. Some stacks are sized explicitly by the
   * user (`max_size != 0`); the rest are "adaptive" (`max_size == 0`) and we have to figure out
   * how many pushes can stack up on them at runtime by walking the CFG and bookkeeping pushes
   * minus pops along every reachable path from kernel entry.
   *
   * For each adaptive AD-stack we want: the maximum value of (pushes - pops) over any walk from
   * the entry node. Set that as `max_size`. If the kernel has a loop where pushes > pops on every
   * iteration the maximum is unbounded; we leave `max_size = 0` and let the caller's structural
   * bounded-loop pre-pass derive a symbolic `SizeExpr` from the loop ranges (and hard-error if
   * even that fails -- there is no compile-time default fallback).
   *
   * High-level algorithm
   * --------------------
   * 1. Index every adaptive AD-stack (`collect_adaptive_ad_stacks`).
   * 2. For every (stack, CFG node) pair, precompute the net push/pop delta inside that node and
   *    the max running push-count prefix inside that node
   *    (`accumulate_per_stack_per_node_size_deltas`). After this point the actual stmts are
   *    irrelevant; everything below operates on int matrices.
   * 3. Condense the CFG with Tarjan's SCC algorithm (`tarjan_scc`). Inside any cyclic SCC, the
   *    "max walk-sum from entry" question becomes "is there a positive cycle, and if not, what's
   *    the DAG longest path inside this SCC?". The condensation lets us solve those questions
   *    once per SCC, in topological order across SCCs.
   * 4. For each *distinct* push/pop fingerprint across stacks (`group_stacks_by_fingerprint`),
   *    run the DP once: walk SCCs in topological order, per cyclic SCC try two cheap sign-based
   *    fast paths and fall back to a single-pass DAG DP + back-edge relaxation check for cycle
   *    detection. Broadcast the result to every stack that shares this fingerprint
   *    (`apply_ad_stack_dp_results`). In practice many autodiff kernels generate identical
   *    push/pop schedules across stacks (one alloca per AD variable in the same loop body), so
   *    the fingerprint dedup collapses N stacks to a handful of DP runs.
   *
   * The two cheap fast paths for a cyclic SCC:
   *   (a) min_is >= 0 and max_is > 0  =>  positive cycle exists for this stack (every node in an
   *       SCC is on a cycle; a cycle through a strictly-positive node with non-negative weights
   *       sums positive). Bail out.
   *   (b) min_is == 0 == max_is       =>  the SCC adds nothing; just spread the maximum entry-
   *       side `max_size_at_node_begin` to every node in the SCC in O(|S|).
   * The full mixed-sign DP runs only when neither shortcut applies.
   *
   * Performance summary
   * -------------------
   * Per representative cost: O(V + E + sum_{cyclic SCC S} |S| * |E_S|), with the inner sum
   * collapsing to O(|S| + |E_S|) for the common autodiff push/pop pattern (one DP sweep + one
   * back-edge check). Overall: O(V + E + R * (V + E)), with R = number of distinct row-pair
   * fingerprints (typically << number of stacks). Per-stack per-node tables are dense
   * `vector<vector<int>>` indexed by contiguous int stack id, not `unordered_map`, to keep the
   * hot inner loop branch-and-cache friendly.
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
