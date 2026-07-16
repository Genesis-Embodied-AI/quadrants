#include "quadrants/transforms/loop_invariant_detector.h"
#include "quadrants/ir/analysis.h"

#include <cstdlib>
#include <cstdio>
#include <string>

namespace quadrants::lang {

namespace {

// Collect the identities of all AtomicOpStmt destinations in |root|.
// Field destinations are recorded by SNode; ndarray destinations by arg_id.
// These must not be cached by the loop-invariant caching pass because the
// AtomicOpStmt writes directly to global memory and any cached load would
// return a stale value.
void gather_atomic_dests(IRNode *root,
                         std::unordered_set<const SNode *> &field_snodes,
                         std::unordered_set<std::vector<int>, hashing::Hasher<std::vector<int>>> &arr_ids) {
  auto stmts = irpass::analysis::gather_statements(root, [](Stmt *s) { return s->is<AtomicOpStmt>(); });
  for (auto *s : stmts) {
    auto *dest = s->as<AtomicOpStmt>()->dest;
    if (auto *gptr = dest->cast<GlobalPtrStmt>()) {
      field_snodes.insert(gptr->snode);
    } else if (auto *eptr = dest->cast<ExternalPtrStmt>()) {
      if (auto *arg = eptr->base_ptr->cast<ArgLoadStmt>()) {
        arr_ids.insert(arg->arg_id);
      }
    } else if (auto *mptr = dest->cast<MatrixPtrStmt>()) {
      if (auto *gptr = mptr->origin->cast<GlobalPtrStmt>()) {
        field_snodes.insert(gptr->snode);
      } else if (auto *eptr = mptr->origin->cast<ExternalPtrStmt>()) {
        if (auto *arg = eptr->base_ptr->cast<ArgLoadStmt>()) {
          arr_ids.insert(arg->arg_id);
        }
      }
    }
  }
}

}  // namespace

class CacheLoopInvariantGlobalVars : public LoopInvariantDetector {
 public:
  using LoopInvariantDetector::visit;

  enum class CacheStatus {
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
  };

  typedef std::unordered_map<Stmt *, std::pair<CacheStatus, AllocaStmt *>> CacheMap;
  std::vector<CacheMap> cached_maps;

  DelayedIRModifier modifier;
  std::unordered_map<const SNode *, GlobalPtrStmt *> loop_unique_ptr_;
  std::unordered_map<std::vector<int>, ExternalPtrStmt *, hashing::Hasher<std::vector<int>>> loop_unique_arr_ptr_;
  std::unordered_set<MatrixPtrStmt *> loop_unique_matrix_ptr_;

  std::unordered_set<Stmt *> dynamic_indexed_ptrs_;
  std::unordered_set<const SNode *> atomic_dest_snodes_;
  std::unordered_set<std::vector<int>, hashing::Hasher<std::vector<int>>> atomic_dest_arr_ids_;

  OffloadedStmt *current_offloaded;

  // Two-phase soundness state. In phase 1 (analyzing_) we only record which snodes cannot be soundly cached;
  // in phase 2 we transform, skipping those snodes. See visit(OffloadedStmt).
  bool analyzing_ = false;
  std::unordered_set<const SNode *> unsafe_analysis_;  // accumulator during phase 1
  std::unordered_set<const SNode *> unsafe_snodes_;    // frozen result consulted during phase 2

  explicit CacheLoopInvariantGlobalVars(const CompileConfig &config) : LoopInvariantDetector(config) {
  }

  void visit(OffloadedStmt *stmt) override {
    if (stmt->task_type == OffloadedTaskType::range_for || stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedTaskType::struct_for) {
      auto uniquely_accessed_pointers = irpass::analysis::gather_uniquely_accessed_pointers(stmt);
      loop_unique_ptr_ = std::move(std::get<0>(uniquely_accessed_pointers));
      loop_unique_arr_ptr_ = std::move(std::get<1>(uniquely_accessed_pointers));
      loop_unique_matrix_ptr_ = std::move(std::get<2>(uniquely_accessed_pointers));
    }
    current_offloaded = stmt;
    dynamic_indexed_ptrs_ = irpass::analysis::gather_dynamically_indexed_pointers(stmt);
    atomic_dest_snodes_.clear();
    atomic_dest_arr_ids_.clear();
    gather_atomic_dests(stmt, atomic_dest_snodes_, atomic_dest_arr_ids_);

    // We don't need to visit TLS/BLS prologues/epilogues.
    if (!stmt->body) {
      current_offloaded = nullptr;
      return;
    }

    // Phase 1 (analysis): find snodes that cannot be soundly cached. A snode is unsafe if some access to it is
    // eligible for caching (offload-unique, static index, non-atomic) yet not cacheable at its own site -- e.g. a
    // store inside an if-block that LICM did not hoist out (move_loop_invariant_outside_if is off). If we cached
    // this snode's other (cacheable) accesses into a loop-invariant local, that un-hoistable access would still
    // read/write global directly, so the local goes stale. Under whole-kernel CSE all accesses share one hoisted
    // pointer and this never happens; under per-task CSE the read/write pointers stay split, exposing it (this is
    // the solver break-flag / -88% regression). Excluding such snodes keeps every access on global -> correct.
    // The two-phase exclusion is a workaround for the read/write pointer split that per-task CSE used to leave in the
    // IR. With merge_global_ptrs now unifying those pointers in the full_simplify fixpoint (like whole-kernel CSE on
    // main), the split is gone and the exclusion is unnecessary; QD_LICM_NO_EXCLUDE=1 disables it to A/B that.
    unsafe_snodes_.clear();
    if (!no_exclude()) {
      analyzing_ = true;
      unsafe_analysis_.clear();
      run_body(stmt);
      analyzing_ = false;
      unsafe_snodes_ = std::move(unsafe_analysis_);
    }

    // Phase 2 (transform): cache as before, but skip the unsafe snodes.
    run_body(stmt);
    current_offloaded = nullptr;
  }

  static bool no_exclude() {
    static const bool v = []() {
      const char *e = std::getenv("QD_LICM_NO_EXCLUDE");
      return e != nullptr && std::string(e) == "1";
    }();
    return v;
  }

  void run_body(OffloadedStmt *stmt) {
    if (stmt->task_type == OffloadedStmt::TaskType::range_for || stmt->task_type == OffloadedTaskType::mesh_for ||
        stmt->task_type == OffloadedStmt::TaskType::struct_for)
      visit_loop(stmt->body.get());
    else
      stmt->body->accept(this);
  }

  bool is_dynamically_indexed(Stmt *stmt) {
    // Handle GlobalPtrStmt
    Stmt *ptr_stmt = nullptr;
    if (stmt->is<GlobalPtrStmt>()) {
      ptr_stmt = stmt->as<GlobalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>()) {
      ptr_stmt = stmt->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    } else if (stmt->is<ExternalPtrStmt>()) {
      ptr_stmt = stmt->as<ExternalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>()) {
      ptr_stmt = stmt->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
    }

    if (ptr_stmt && dynamic_indexed_ptrs_.count(ptr_stmt)) {
      return true;
    }

    return false;
  }

  bool is_offload_unique(Stmt *stmt) {
    if (current_offloaded->task_type == OffloadedTaskType::serial) {
      return true;
    }

    // Handle GlobalPtrStmt
    bool is_global_ptr_stmt = false;
    GlobalPtrStmt *global_ptr = nullptr;
    if (stmt->is<GlobalPtrStmt>()) {
      is_global_ptr_stmt = true;
      global_ptr = stmt->as<GlobalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>()) {
      if (loop_unique_matrix_ptr_.find(stmt->as<MatrixPtrStmt>()) == loop_unique_matrix_ptr_.end()) {
        return false;
      }

      is_global_ptr_stmt = true;
      global_ptr = stmt->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }

    if (global_ptr) {
      auto snode = global_ptr->snode;
      if (loop_unique_ptr_[snode] == nullptr || loop_unique_ptr_[snode]->indices.empty()) {
        // not uniquely accessed
        return false;
      }
      if (current_offloaded->mem_access_opt.has_flag(snode, SNodeAccessFlag::block_local) ||
          current_offloaded->mem_access_opt.has_flag(snode, SNodeAccessFlag::mesh_local)) {
        // BLS does not support write access yet so we keep atomic_adds.
        return false;
      }
      return true;
    }

    // Handle ExternalPtrStmt
    bool is_external_ptr_stmt = false;
    ExternalPtrStmt *dest_ptr = nullptr;
    if (stmt->is<ExternalPtrStmt>()) {
      is_external_ptr_stmt = true;
      dest_ptr = stmt->as<ExternalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>() && stmt->as<MatrixPtrStmt>()->origin->is<ExternalPtrStmt>()) {
      if (loop_unique_matrix_ptr_.find(stmt->as<MatrixPtrStmt>()) == loop_unique_matrix_ptr_.end()) {
        return false;
      }

      is_external_ptr_stmt = true;
      dest_ptr = stmt->as<MatrixPtrStmt>()->origin->as<ExternalPtrStmt>();
    }

    if (is_external_ptr_stmt) {
      if (dest_ptr->indices.empty()) {
        return false;
      }
      ArgLoadStmt *arg_load_stmt = dest_ptr->base_ptr->as<ArgLoadStmt>();
      std::vector<int> arg_id = arg_load_stmt->arg_id;
      if (loop_unique_arr_ptr_[arg_id] == nullptr) {
        // Not loop unique
        return false;
      }
      return true;
      // TODO: Is BLS / Mem Access Opt a thing for any_arr?
    }
    return false;
  }

  void visit_loop(Block *body) override {
    cached_maps.emplace_back();
    LoopInvariantDetector::visit_loop(body);
    cached_maps.pop_back();
  }

  void add_writeback(AllocaStmt *alloca_stmt, Stmt *global_var, int depth) {
    auto final_value = std::make_unique<LocalLoadStmt>(alloca_stmt);
    auto global_store = std::make_unique<GlobalStoreStmt>(global_var, final_value.get());
    modifier.insert_after(get_loop_stmt(depth), std::move(global_store));
    modifier.insert_after(get_loop_stmt(depth), std::move(final_value));
  }

  void set_init_value(AllocaStmt *alloca_stmt, Stmt *global_var, int depth) {
    auto new_global_load = std::make_unique<GlobalLoadStmt>(global_var);
    auto local_store = std::make_unique<LocalStoreStmt>(alloca_stmt, new_global_load.get());
    modifier.insert_before(get_loop_stmt(depth), std::move(new_global_load));
    modifier.insert_before(get_loop_stmt(depth), std::move(local_store));
  }

  static bool licm_log() {
    static const bool v = []() {
      const char *e = std::getenv("QD_LICM_LOG");
      return e != nullptr && std::string(e) == "1";
    }();
    return v;
  }

  static std::string describe_dest(Stmt *dest) {
    GlobalPtrStmt *g = nullptr;
    if (dest->is<GlobalPtrStmt>()) {
      g = dest->as<GlobalPtrStmt>();
    } else if (dest->is<MatrixPtrStmt>() && dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>()) {
      g = dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>();
    }
    if (g) {
      return "id$" + std::to_string(dest->id) + " snode#" + std::to_string(g->snode->id) + "(" +
             g->snode->get_node_type_name() + ")";
    }
    return "id$" + std::to_string(dest->id) + " non-global";
  }

  // Match an existing cache entry at |depth| by GlobalPtrStmt* identity, or by provable same-address
  // (definitely_same_address). Address matching is required because per-task CSE (post-offload) cannot merge a
  // read GlobalPtrStmt (activate=false, LICM-hoisted) with the in-loop write GlobalPtrStmts to the same address;
  // keying purely by pointer identity would then allocate separate locals -> stale in-loop reads.
  std::pair<CacheStatus, AllocaStmt *> *find_cache_entry(int depth, Stmt *dest) {
    auto &m = cached_maps[depth];
    auto it = m.find(dest);
    if (it != m.end() && it->second.first != CacheStatus::None) {
      return &it->second;
    }
    for (auto &kv : m) {
      if (kv.second.first != CacheStatus::None && kv.first != dest &&
          irpass::analysis::definitely_same_address(kv.first, dest)) {
        return &kv.second;
      }
    }
    return nullptr;
  }

  AllocaStmt *cache_global_to_local(Stmt *dest, CacheStatus status, int depth) {
    auto *entry = find_cache_entry(depth, dest);
    if (licm_log()) {
      std::printf("[LICM] cache dest=%s status=%d depth=%d match=%s mapsz=%zu\n", describe_dest(dest).c_str(),
                  (int)status, depth, entry ? "YES" : "NO", cached_maps[depth].size());
      if (!entry) {
        for (auto &kv : cached_maps[depth]) {
          std::printf("[LICM]   existing=%s status=%d dsa=%d\n", describe_dest(kv.first).c_str(),
                      (int)kv.second.first, (int)irpass::analysis::definitely_same_address(kv.first, dest));
        }
      }
      std::fflush(stdout);
    }
    if (entry) {
      auto &[cached_status, alloca_stmt] = *entry;
      // The global variable has already been cached (same pointer or provably same address).
      if (cached_status == CacheStatus::Read && status == CacheStatus::Write) {
        add_writeback(alloca_stmt, dest, depth);
        cached_status = CacheStatus::ReadWrite;
      }
      return alloca_stmt;
    }
    auto alloca_unique = std::make_unique<AllocaStmt>(dest->ret_type.ptr_removed());
    auto alloca_stmt = alloca_unique.get();
    modifier.insert_before(get_loop_stmt(depth), std::move(alloca_unique));
    set_init_value(alloca_stmt, dest, depth);
    if (status == CacheStatus::Write) {
      add_writeback(alloca_stmt, dest, depth);
    }
    cached_maps[depth][dest] = {status, alloca_stmt};
    return alloca_stmt;
  }

  bool is_atomic_dest(Stmt *stmt) {
    GlobalPtrStmt *gptr = nullptr;
    ExternalPtrStmt *eptr = nullptr;

    if (stmt->is<GlobalPtrStmt>()) {
      gptr = stmt->as<GlobalPtrStmt>();
    } else if (stmt->is<ExternalPtrStmt>()) {
      eptr = stmt->as<ExternalPtrStmt>();
    } else if (stmt->is<MatrixPtrStmt>()) {
      auto *origin = stmt->as<MatrixPtrStmt>()->origin;
      if (origin->is<GlobalPtrStmt>()) {
        gptr = origin->as<GlobalPtrStmt>();
      } else if (origin->is<ExternalPtrStmt>()) {
        eptr = origin->as<ExternalPtrStmt>();
      }
    }

    if (gptr && atomic_dest_snodes_.count(gptr->snode)) {
      return true;
    }
    if (eptr) {
      if (auto *arg = eptr->base_ptr->cast<ArgLoadStmt>()) {
        if (atomic_dest_arr_ids_.count(arg->arg_id)) {
          return true;
        }
      }
    }
    return false;
  }

  std::optional<int> find_cache_depth_if_cacheable(Stmt *operand, Block *current_scope) {
    if (is_dynamically_indexed(operand)) {
      return std::nullopt;
    }
    if (!is_offload_unique(operand)) {
      return std::nullopt;
    }
    if (is_atomic_dest(operand)) {
      return std::nullopt;
    }
    std::optional<int> depth;
    for (int n = loop_blocks.size() - 1; n > 0; n--) {
      if (is_operand_loop_invariant(operand, current_scope, n)) {
        depth = n;
      } else {
        break;
      }
    }
    return depth;
  }

  static const SNode *dest_snode(Stmt *dest) {
    if (dest->is<GlobalPtrStmt>()) {
      return dest->as<GlobalPtrStmt>()->snode;
    }
    if (dest->is<MatrixPtrStmt>() && dest->as<MatrixPtrStmt>()->origin->is<GlobalPtrStmt>()) {
      return dest->as<MatrixPtrStmt>()->origin->as<GlobalPtrStmt>()->snode;
    }
    return nullptr;
  }

  // Would this pass otherwise want to cache accesses to |dest| (offload-unique, statically indexed, non-atomic)?
  // Whether a *particular* access is cacheable additionally depends on its scope (find_cache_depth_if_cacheable).
  bool cache_eligible(Stmt *dest) {
    return !is_dynamically_indexed(dest) && is_offload_unique(dest) && !is_atomic_dest(dest);
  }

  // Phase-1 hook: mark the snode unsafe if this access is cache-eligible but not cacheable at its own site.
  void analyze_access(Stmt *dest, Block *scope) {
    const SNode *sn = dest_snode(dest);
    if (!sn || !cache_eligible(dest)) {
      return;
    }
    if (!find_cache_depth_if_cacheable(dest, scope).has_value()) {
      if (unsafe_analysis_.insert(sn).second && licm_log()) {
        std::printf("[LICM] UNSAFE snode#%d(%s) via %s\n", sn->id, sn->get_node_type_name().c_str(),
                    describe_dest(dest).c_str());
        std::fflush(stdout);
      }
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    // Volatile loads must read from memory on every execution (spin-wait correctness); skip caching.
    if (stmt->is_volatile) {
      return;
    }
    if (analyzing_) {
      analyze_access(stmt->src, stmt->parent);
      return;
    }
    if (const SNode *sn = dest_snode(stmt->src); sn && unsafe_snodes_.count(sn)) {
      return;
    }
    if (auto depth = find_cache_depth_if_cacheable(stmt->src, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(stmt->src, CacheStatus::Read, depth.value());
      auto local_load = std::make_unique<LocalLoadStmt>(alloca_stmt);
      stmt->replace_usages_with(local_load.get());
      modifier.insert_before(stmt, std::move(local_load));
      modifier.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (analyzing_) {
      analyze_access(stmt->dest, stmt->parent);
      return;
    }
    if (const SNode *sn = dest_snode(stmt->dest); sn && unsafe_snodes_.count(sn)) {
      return;
    }
    if (auto depth = find_cache_depth_if_cacheable(stmt->dest, stmt->parent)) {
      auto alloca_stmt = cache_global_to_local(stmt->dest, CacheStatus::Write, depth.value());
      auto local_store = std::make_unique<LocalStoreStmt>(alloca_stmt, stmt->val);
      stmt->replace_usages_with(local_store.get());
      modifier.insert_before(stmt, std::move(local_store));
      modifier.erase(stmt);
    }
  }

  static bool run(IRNode *node, const CompileConfig &config) {
    bool modified = false;

    while (true) {
      CacheLoopInvariantGlobalVars eliminator(config);
      node->accept(&eliminator);
      if (eliminator.modifier.modify_ir())
        modified = true;
      else
        break;
    };

    return modified;
  }
};

namespace irpass {
bool cache_loop_invariant_global_vars(IRNode *root, const CompileConfig &config) {
  QD_AUTO_PROF;
  return CacheLoopInvariantGlobalVars::run(root, config);
}
}  // namespace irpass
}  // namespace quadrants::lang
