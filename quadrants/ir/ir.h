// Intermediate representation system

#pragma once

#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <tuple>
#include <type_traits>
#include <utility>

#include "quadrants/common/core.h"
#include "quadrants/common/exceptions.h"
#include "quadrants/common/one_or_more.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/mesh.h"
#include "quadrants/ir/type_factory.h"
#include "quadrants/util/short_name.h"

#ifdef QD_WITH_LLVM
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/MapVector.h"
#endif

namespace quadrants::lang {

class IRNode;
class Block;
class Stmt;
using pStmt = std::unique_ptr<Stmt>;

class SNode;

class Kernel;
class Callable;
struct CompileConfig;

enum class SNodeAccessFlag : int { block_local, read_only, mesh_local };
std::string snode_access_flag_name(SNodeAccessFlag type);

class MemoryAccessOptions {
 public:
  void add_flag(SNode *snode, SNodeAccessFlag flag) {
    options_[snode].insert(flag);
  }

  bool has_flag(SNode *snode, SNodeAccessFlag flag) const {
    if (auto it = options_.find(snode); it != options_.end())
      return it->second.count(flag) != 0;
    else
      return false;
  }

  std::vector<SNode *> get_snodes_with_flag(SNodeAccessFlag flag) const {
    std::vector<SNode *> snodes;
    for (const auto &opt : options_) {
      if (has_flag(opt.first, flag)) {
        snodes.push_back(opt.first);
      }
    }
    return snodes;
  }

  void clear() {
    options_.clear();
  }

  std::unordered_map<SNode *, std::unordered_set<SNodeAccessFlag>> get_all() const {
    return options_;
  }

 private:
  std::unordered_map<SNode *, std::unordered_set<SNodeAccessFlag>> options_;
};

#define PER_STATEMENT(x) class x;
#include "quadrants/inc/statements.inc.h"
#undef PER_STATEMENT

class Identifier {
 public:
  std::string name_;
  int id{0};

  // Identifier() = default;

  // Multiple identifiers can share the same name but must have different id's
  explicit Identifier(int id, const std::string &name = "") : name_(name), id(id) {
  }

  std::string raw_name() const;

  std::string name() const {
    return "@" + raw_name();
  }

  bool operator<(const Identifier &o) const {
    return id < o.id;
  }

  bool operator==(const Identifier &o) const {
    return id == o.id;
  }
};

#ifdef QD_WITH_LLVM
using stmt_vector = llvm::SmallVector<pStmt, 8>;
using stmt_ref_vector = llvm::SmallVector<Stmt *, 2>;
#else
using stmt_vector = std::vector<pStmt>;
using stmt_ref_vector = std::vector<Stmt *>;
#endif

class VecStatement {
 public:
  stmt_vector stmts;

  VecStatement() {
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  VecStatement(pStmt &&stmt) {
    push_back(std::move(stmt));
  }

  VecStatement(VecStatement &&o) {
    stmts = std::move(o.stmts);
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  VecStatement(stmt_vector &&other_stmts) {
    stmts = std::move(other_stmts);
  }

  Stmt *push_back(pStmt &&stmt);

  template <typename T, typename... Args>
  T *push_back(Args &&...args) {
    auto up = std::make_unique<T>(std::forward<Args>(args)...);
    auto ptr = up.get();
    stmts.push_back(std::move(up));
    return ptr;
  }

  pStmt &back() {
    return stmts.back();
  }

  std::size_t size() const {
    return stmts.size();
  }

  pStmt &operator[](int i) {
    return stmts[i];
  }
};

class IRVisitor {
 public:
  bool allow_undefined_visitor;
  bool invoke_default_visitor;

  IRVisitor() {
    allow_undefined_visitor = false;
    invoke_default_visitor = false;
  }

  virtual ~IRVisitor() = default;

  // default visitor
  virtual void visit(Stmt *stmt) {
    if (!allow_undefined_visitor) {
      QD_ERROR(
          "missing visitor function. Is the statement class registered via "
          "DEFINE_VISIT?");
    }
  }

#define DEFINE_VISIT(T)            \
  virtual void visit(T *stmt) {    \
    if (allow_undefined_visitor) { \
      if (invoke_default_visitor)  \
        visit((Stmt *)stmt);       \
    } else                         \
      QD_NOT_IMPLEMENTED;          \
  }

  DEFINE_VISIT(Block);
#define PER_STATEMENT(x) DEFINE_VISIT(x)
#include "quadrants/inc/statements.inc.h"

#undef PER_STATEMENT
#undef DEFINE_VISIT
};

struct CompileConfig;
class Kernel;

using stmt_refs = one_or_more<Stmt *>;

namespace ir_traits {

// FIXME: Use C++ 20 concepts to replace `dynamic_cast<T>() != nullptr`

class Store {
 public:
  virtual ~Store() = default;

  // Get the list of sinks/destinations of the store operation
  virtual stmt_refs get_store_destination() const = 0;

  // If store_stmt provides one data source, return the data.
  virtual Stmt *get_store_data() const = 0;
};

class Load {
 public:
  virtual ~Load() = default;

  // If load_stmt loads some variables or a stack, return the pointers of them.
  virtual stmt_refs get_load_pointers() const = 0;
};

}  // namespace ir_traits

class IRNode {
 public:
  virtual void accept(IRVisitor *visitor) {
    QD_NOT_IMPLEMENTED
  }

  // * For a Stmt, this returns its enclosing Block
  // * For a Block, this returns its enclosing Stmt
  virtual IRNode *get_parent() const = 0;

  IRNode *get_ir_root();
  const IRNode *get_ir_root() const;

  virtual ~IRNode() = default;

  template <typename T>
  bool is() const {
    return dynamic_cast<const T *>(this) != nullptr;
  }

  template <typename T>
  T *as() {
    QD_ASSERT(is<T>());
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  const T *as() const {
    QD_ASSERT(is<T>());
    return dynamic_cast<const T *>(this);
  }

  template <typename T>
  T *cast() {
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  const T *cast() const {
    return dynamic_cast<const T *>(this);
  }

  std::unique_ptr<IRNode> clone();
};

#define QD_DEFINE_ACCEPT                     \
  void accept(IRVisitor *visitor) override { \
    visitor->visit(this);                    \
  }

#define QD_DEFINE_CLONE                                                         \
  std::unique_ptr<Stmt> clone() const override {                                \
    auto new_stmt = std::make_unique<std::decay<decltype(*this)>::type>(*this); \
    new_stmt->mark_fields_registered();                                         \
    new_stmt->io(new_stmt->field_manager);                                      \
    return new_stmt;                                                            \
  }

#define QD_DEFINE_ACCEPT_AND_CLONE \
  QD_DEFINE_ACCEPT                 \
  QD_DEFINE_CLONE

class StmtField {
 public:
  StmtField() = default;

  virtual bool equal(const StmtField *other) const = 0;

  virtual ~StmtField() = default;
};

template <typename T>
class StmtFieldNumeric final : public StmtField {
 private:
  std::variant<T *, T> value_;

 public:
  explicit StmtFieldNumeric(T *value) : value_(value) {
  }

  explicit StmtFieldNumeric(T value) : value_(value) {
  }

  bool equal(const StmtField *other_generic) const override {
    if (auto other = dynamic_cast<const StmtFieldNumeric *>(other_generic)) {
      if (std::holds_alternative<T *>(other->value_) && std::holds_alternative<T *>(value_)) {
        return *(std::get<T *>(other->value_)) == *(std::get<T *>(value_));
      } else if (std::holds_alternative<T *>(other->value_) || std::holds_alternative<T *>(value_)) {
        QD_ERROR(
            "Inconsistent StmtField value types: a pointer value is compared "
            "to a non-pointer value.");
        return false;
      } else {
        return std::get<T>(other->value_) == std::get<T>(value_);
      }
    } else {
      // Different types
      return false;
    }
  }
};

class StmtFieldSNode final : public StmtField {
 private:
  SNode *const &snode_;

 public:
  explicit StmtFieldSNode(SNode *const &snode) : snode_(snode) {
  }

  static int get_snode_id(SNode *snode);

  bool equal(const StmtField *other_generic) const override;
};

class StmtFieldMemoryAccessOptions final : public StmtField {
 private:
  MemoryAccessOptions const &opt_;

 public:
  explicit StmtFieldMemoryAccessOptions(MemoryAccessOptions const &opt) : opt_(opt) {
  }

  bool equal(const StmtField *other_generic) const override;
};

class StmtFieldManager {
 private:
  Stmt *stmt_;

 public:
  std::vector<std::unique_ptr<StmtField>> fields;

  explicit StmtFieldManager(Stmt *stmt) : stmt_(stmt) {
  }

  template <typename T>
  void operator()(const char *key, T &&value);

  template <typename T, typename... Args>
  void operator()(const char *key_, T &&t, Args &&...rest) {
    std::string key(key_);
    size_t pos = key.find(',');
    std::string first_name = key.substr(0, pos);
    std::string rest_names = key.substr(pos + 2, int(key.size()) - (int)pos - 2);
    this->operator()(first_name.c_str(), std::forward<T>(t));
    this->operator()(rest_names.c_str(), std::forward<Args>(rest)...);
  }

  bool equal(StmtFieldManager &other) const;
};

#define QD_STMT_DEF_FIELDS(...)  \
  template <typename S>          \
  void io(S &serializer) const { \
    QD_IO(__VA_ARGS__);          \
  }
#define QD_STMT_REG_FIELDS  \
  mark_fields_registered(); \
  io(field_manager)

// The graph-region a statement belongs to, for `@qd.kernel(graph=True)` kernels. Captures the
// innermost `qd.graph_do_while()` nesting level and the `qd.stream_parallel()` concurrency group at the
// point in the source where the statement was written. Stamped on every frontend statement by
// `ASTBuilder::insert`, propagated onto lowered statements by `lower_ast`, and read by `offload.cpp` so
// that serial (non-for) work is flushed into an offloaded task tagged with the level it was written at --
// rather than the level of whatever for-loop happens to flush the pending-serial bucket next. The
// defaults below (`-1` / `0`) mean "kernel top level, no concurrency group", which is the correct
// neutral value for ordinary (non-graph) kernels and for statements created by passes outside a graph
// region. For-loop statements additionally carry the same two values in their own dedicated fields (set
// from the `ForLoopConfig`); those are left in place for now.
//
struct GraphRegionTag {
  int graph_do_while_level_id{-1};
  int stream_parallel_group_id{0};
  // Per-kernel `qd.graph_parallel_context()` region id (0 outside any region). Stamped by `ASTBuilder::insert` from
  // `current_graph_parallel_region_id_` onto every frontend statement (so a bare side-effecting store written directly
  // in a `qd.graph_parallel()` section -- which has no for-loop carrying the id via ForLoopConfig -- still names its
  // region), and also set on the PURE-bucket `fallback_tag` in `offload.cpp` (so a section's pure bound-compute serial
  // is grouped into the same region as the for-loop that consumes it). Excluded from operator==/!=: a section's
  // `stream_parallel_group_id` is unique per kernel and a section belongs to exactly one region, so for the fork/join
  // tasks the serial bucket compares, equal group already implies equal region -- region never needs to participate.
  int graph_parallel_region_id{0};
  // `cp_id` (see `quadrants/lang/checkpoint.py`) of the enclosing `qd.checkpoint(...)` block, or `-1` outside any
  // checkpoint. Stamped on every frontend statement by `ASTBuilder::insert` (from `current_checkpoint_id_`) so a
  // SIDE-EFFECTING serial task tagged with a real checkpoint can be gated/skipped on `resume(from_checkpoint=...)`
  // exactly like a range-for in that checkpoint. The PURE-bucket fallback in `offload.cpp` deliberately leaves
  // `checkpoint_id = -1` (always-run) so a hoisted loop bound / shared constant a later task reads is never gated away
  // on a resume launch.
  int checkpoint_id{-1};
  // `is_set` distinguishes a tag that was explicitly stamped at a known program point (frontend
  // build via ASTBuilder::insert, lowering via FlattenContext, or the region-propagating
  // insert_before_me / replace_with helpers) from a tag that's still at its struct default because
  // it was never touched. A bare top-level statement has graph_do_while_level_id=-1 with is_set=true,
  // whereas a stmt manufactured by a later compiler pass that didn't go through any of those paths
  // also reads as graph_do_while_level_id=-1 but with is_set=false; the offloader treats only the
  // former as a trustworthy signal of where the work belongs. Excluded from operator==/!= so the
  // bucket-region comparison in `push_serial_statement` only cares about the actual region.
  bool is_set{false};

  GraphRegionTag() = default;
  // 2-arg ctor: (level, group). Defaults checkpoint_id to -1 ("not inside any checkpoint"). Use at call sites that
  // don't know / don't care about checkpoint_id (e.g., pure-only fallback tag in offload.cpp).
  GraphRegionTag(int level, int group) : graph_do_while_level_id(level), stream_parallel_group_id(group), is_set(true) {
  }
  // 3-arg ctor: (level, group, checkpoint). Used by `ASTBuilder::insert` to stamp the active region on every frontend
  // statement.
  GraphRegionTag(int level, int group, int checkpoint)
      : graph_do_while_level_id(level), stream_parallel_group_id(group), checkpoint_id(checkpoint), is_set(true) {
  }
  // 4-arg ctor: (level, group, region, checkpoint). Builds a full tag from the loose ForLoopConfig ints a
  // `FrontendForStmt` carries, so the offline-cache-key generator can route a for-loop through the same
  // `cache_key_members()` list as a per-`Stmt` `region_tag` (see gen_offline_cache_key.cpp).
  GraphRegionTag(int level, int group, int region, int checkpoint)
      : graph_do_while_level_id(level),
        stream_parallel_group_id(group),
        graph_parallel_region_id(region),
        checkpoint_id(checkpoint),
        is_set(true) {
  }

  // The fields that distinguish a tag for the OFFLINE CACHE KEY: everything *semantic* (anything that changes how the
  // offloader groups or gates a task) except `is_set` (a build-time "was this stamped" provenance flag, not a property
  // of the kernel). This is the SINGLE list gen_offline_cache_key.cpp iterates -- add a field here the moment you add
  // one to the struct, or two kernels that differ only in that field silently share a cached module. The static_assert
  // below ties this list to the struct's size so an >=int-sized field added to the struct but omitted here fails to
  // compile.
  auto cache_key_members() const {
    return std::tie(graph_do_while_level_id, stream_parallel_group_id, graph_parallel_region_id, checkpoint_id);
  }
  // The fields that make two tags EQUAL for the offloader's serial-bucket comparison in `push_serial_statement`. A
  // strict subset of `cache_key_members()`: `graph_parallel_region_id` is deliberately excluded (a section's
  // `stream_parallel_group_id` is unique per kernel and a section belongs to exactly one region, so equal group already
  // implies equal region for the tasks this compares), as is `is_set`. `operator==` derives from this so the exclusion
  // lives in one place next to the field it excludes.
  auto equality_members() const {
    return std::tie(graph_do_while_level_id, stream_parallel_group_id, checkpoint_id);
  }
  bool operator==(const GraphRegionTag &o) const {
    return equality_members() == o.equality_members();
  }
  bool operator!=(const GraphRegionTag &o) const {
    return !(*this == o);
  }
};

namespace graph_region_tag_detail {
constexpr std::size_t round_up(std::size_t n, std::size_t align) {
  return (n + align - 1) / align * align;
}
template <typename Tuple>
struct tuple_bytes;
template <typename... Ts>
struct tuple_bytes<std::tuple<Ts...>> {
  // Sum of the sizes of the *referenced* field types (cache_key_members() ties them as references).
  static constexpr std::size_t value = (std::size_t{0} + ... + sizeof(std::remove_reference_t<Ts>));
};
}  // namespace graph_region_tag_detail

// Layout tripwire for GraphRegionTag, DERIVED from cache_key_members() (no magic size constant). Expected size = the
// packed bytes of every cache-key field + `is_set`, rounded to the struct's alignment. Add an >=int-sized field to the
// struct but forget to add it to cache_key_members() and the two sides disagree: `sizeof` grows while the derived sum
// does not, and this fails the build -- forcing you to decide whether the new field belongs in the cache key (and, if
// it is semantic for equality, in equality_members()). Residual gap: a field that fits in the <4-byte tail padding
// after `is_set` (e.g. another bool) omitted from the list won't change `sizeof`, so keep `is_set` the LAST member and
// put new fields before it.
static_assert(sizeof(GraphRegionTag) ==
                  graph_region_tag_detail::round_up(
                      graph_region_tag_detail::tuple_bytes<
                          decltype(std::declval<const GraphRegionTag &>().cache_key_members())>::value +
                          sizeof(bool),
                      alignof(GraphRegionTag)),
              "GraphRegionTag layout changed: add the new field to cache_key_members() (offline cache key) and, if it "
              "is semantic for equality, to equality_members(); keep is_set the last member.");

class Stmt : public IRNode {
 protected:
  std::vector<Stmt **> operands;
  explicit Stmt(const DebugInfo &dbg_info);

 public:
  StmtFieldManager field_manager;
  static std::atomic<int> instance_id_counter;
  int instance_id;
  int id;
  Block *parent;
  bool erased;
  bool fields_registered;
  DataType ret_type;
  DebugInfo dbg_info;
  // Graph-region this statement belongs to (see GraphRegionTag). Only meaningful for statements
  // that can end up at the offloader's root level as serial work; ignored otherwise.
  GraphRegionTag region_tag;

  Stmt();
  Stmt(const Stmt &stmt);

  virtual bool is_container_statement() const {
    return false;
  }

  DataType &element_type() {
    return ret_type;
  }

  std::string ret_data_type_name() const {
    return ret_type->to_string();
  }

  std::string type_hint() const;

  std::string name() const {
    return fmt::format("${}", id);
  }

  std::string short_name() const {
    return make_short_name_by_id(id);
  }

  std::string raw_name() const {
    return fmt::format("tmp{}", id);
  }

  QD_FORCE_INLINE int num_operands() const {
    return (int)operands.size();
  }

  QD_FORCE_INLINE Stmt *operand(int i) const {
    // QD_ASSERT(0 <= i && i < (int)operands.size());
    return *operands[i];
  }

  std::string get_last_tb() const;

  QD_FORCE_INLINE std::string const &get_tb() const {
    return dbg_info.tb;
  }

  QD_FORCE_INLINE void set_tb(const std::string &tb) {
    dbg_info.tb = tb;
  }

  std::vector<Stmt *> get_operands() const;

  void set_operand(int i, Stmt *stmt);
  void register_operand(Stmt *&stmt);
  int locate_operand(Stmt **stmt);
  void mark_fields_registered();

  bool has_operand(Stmt *stmt) const;

  void replace_usages_with(Stmt *new_stmt);
  void replace_with(VecStatement &&new_statements, bool replace_usages = true);
  virtual void replace_operand_with(Stmt *old_stmt, Stmt *new_stmt);

  IRNode *get_parent() const override;
  virtual Callable *get_callable() const;

  // returns the inserted stmt
  Stmt *insert_before_me(std::unique_ptr<Stmt> &&new_stmt);

  // returns the inserted stmt
  Stmt *insert_after_me(std::unique_ptr<Stmt> &&new_stmt);

  virtual bool has_global_side_effect() const {
    return true;
  }

  virtual bool dead_instruction_eliminable() const {
    return !has_global_side_effect();
  }

  virtual bool common_statement_eliminable() const {
    return !has_global_side_effect();
  }

  template <typename T, typename... Args>
  static std::unique_ptr<T> make_typed(Args &&...args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static pStmt make(Args &&...args) {
    return make_typed<T>(std::forward<Args>(args)...);
  }

  std::string type();

  virtual std::unique_ptr<Stmt> clone() const {
    QD_NOT_IMPLEMENTED
  }

  ~Stmt() override = default;

  static void reset_counter() {
    instance_id_counter = 0;
  }
};

class Block : public IRNode {
 public:
  std::variant<Stmt *, Callable *> parent_;
  stmt_vector statements;
  stmt_vector trash_bin;
  std::vector<SNode *> stop_gradients;

  // Only used in frontend. Stores LoopIndexStmt or BinaryOpStmt for loop
  // variables, and AllocaStmt for other variables.
  std::map<Identifier, Stmt *> local_var_to_stmt;

  explicit Block(Callable *callable = nullptr) {
    parent_ = callable;
  }

  Stmt *parent_stmt() const;

  void set_parent_callable(Callable *callable);

  void set_parent_stmt(Stmt *stmt);

  Callable *parent_callable() const;

  Block *parent_block() const;

  bool has_container_statements();
  int locate(Stmt *stmt);
  stmt_vector::iterator locate(int location);
  stmt_vector::iterator find(Stmt *stmt);
  void erase(int location);
  void erase(Stmt *stmt);
  void erase_range(stmt_vector::iterator begin, stmt_vector::iterator end);
  void erase(std::unordered_set<Stmt *> stmts);
  std::unique_ptr<Stmt> extract(int location);
  std::unique_ptr<Stmt> extract(Stmt *stmt);

  // Returns stmt.get()
  Stmt *insert(std::unique_ptr<Stmt> &&stmt, int location = -1);
  Stmt *insert_at(std::unique_ptr<Stmt> &&stmt, stmt_vector::iterator location);

  // Returns stmt.back().get() or nullptr if stmt is empty
  Stmt *insert(VecStatement &&stmt, int location = -1);
  Stmt *insert_at(VecStatement &&stmt, stmt_vector::iterator location);

  void replace_statements_in_range(int start, int end, VecStatement &&stmts);
  void set_statements(VecStatement &&stmts);
  void replace_with(Stmt *old_statement, std::unique_ptr<Stmt> &&new_statement, bool replace_usages = true);
  void insert_before(Stmt *old_statement, VecStatement &&new_statements);
  void insert_after(Stmt *old_statement, VecStatement &&new_statements);
  void replace_with(Stmt *old_statement, VecStatement &&new_statements, bool replace_usages = true);
  Stmt *lookup_var(const Identifier &ident) const;
  IRNode *get_parent() const override;

  Stmt *back() const {
    return statements.back().get();
  }

  template <typename T, typename... Args>
  Stmt *push_back(Args &&...args) {
    auto stmt = std::make_unique<T>(std::forward<Args>(args)...);
    stmt->parent = this;
    statements.emplace_back(std::move(stmt));
    return back();
  }

  std::size_t size() const {
    return statements.size();
  }

  pStmt &operator[](int i) {
    return statements[i];
  }

  std::unique_ptr<Block> clone() const;

  QD_DEFINE_ACCEPT
};

class DelayedIRModifier {
 private:
  std::vector<std::pair<Stmt *, VecStatement>> to_insert_before_;
  std::vector<std::pair<Stmt *, VecStatement>> to_insert_after_;
  std::vector<std::tuple<Stmt *, VecStatement, bool>> to_replace_with_;
  std::vector<Stmt *> to_erase_;
  std::vector<std::pair<Stmt *, Block *>> to_extract_to_block_front_;
  std::vector<std::pair<IRNode *, CompileConfig>> to_type_check_;
  bool modified_{false};

 public:
  ~DelayedIRModifier();
  void erase(Stmt *stmt);
  void insert_before(Stmt *old_statement, std::unique_ptr<Stmt> new_statement);
  void insert_before(Stmt *old_statement, VecStatement &&new_statements);
  void insert_after(Stmt *old_statement, std::unique_ptr<Stmt> new_statement);
  void insert_after(Stmt *old_statement, VecStatement &&new_statements);
  void replace_with(Stmt *stmt, VecStatement &&new_statements, bool replace_usages = true);
  void extract_to_block_front(Stmt *stmt, Block *blk);
  void type_check(IRNode *node, CompileConfig cfg);
  bool modify_ir();

  // Force the next call of modify_ir() to return true.
  void mark_as_modified();
};

// ImmediateIRModifier aims at replacing Stmt::replace_usages_with, which visits
// the whole tree for a single replacement. ImmediateIRModifier is currently
// associated with a pass, visits the whole tree once at the beginning of that
// pass, and performs a single replacement with amortized constant time.
class ImmediateIRModifier {
 private:
  std::unordered_map<Stmt *, std::vector<std::pair<Stmt *, int>>> stmt_usages_;

 public:
  explicit ImmediateIRModifier(IRNode *root);
  void replace_usages_with(Stmt *old_stmt, Stmt *new_stmt);
};

template <typename T>
inline void StmtFieldManager::operator()(const char *key, T &&value) {
  using decay_T = typename std::decay<T>::type;
  if constexpr (is_specialization<decay_T, std::vector>::value) {
    stmt_->field_manager.fields.emplace_back(std::make_unique<StmtFieldNumeric<std::size_t>>(value.size()));
    for (int i = 0; i < (int)value.size(); i++) {
      (*this)("__element", value[i]);
    }
  } else if constexpr (std::is_same<decay_T, std::variant<Stmt *, std::string>>::value) {
    if (std::holds_alternative<std::string>(value)) {
      stmt_->field_manager.fields.emplace_back(
          std::make_unique<StmtFieldNumeric<std::string>>(std::get<std::string>(value)));
    } else {
      (*this)("__element", std::get<Stmt *>(value));
    }
  } else if constexpr (std::is_same<decay_T, Stmt *>::value) {
    stmt_->register_operand(const_cast<Stmt *&>(value));
  } else if constexpr (std::is_same<decay_T, SNode *>::value) {
    stmt_->field_manager.fields.emplace_back(std::make_unique<StmtFieldSNode>(value));
  } else if constexpr (std::is_same<decay_T, MemoryAccessOptions>::value) {
    stmt_->field_manager.fields.emplace_back(std::make_unique<StmtFieldMemoryAccessOptions>(value));
  } else {
    stmt_->field_manager.fields.emplace_back(std::make_unique<StmtFieldNumeric<std::remove_reference_t<T>>>(&value));
  }
}

}  // namespace quadrants::lang
