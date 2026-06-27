#pragma once

#include <vector>
#include <memory>

namespace quadrants::lang {

class SNode;
class Program;

/**
 * A helper class to keep the root SNodes that aren't materialized yet.
 *
 * This is an awkward workaround given that the bindings do not allow returning a
 * std::unique_ptr instance and then moving it back to C++.
 *
 * Not thread safe.
 */
class SNodeRegistry {
 public:
  SNodeRegistry() = default;
  // Owns unique_ptrs, so it is move-only. nanobind inspects std::is_copy_constructible to decide whether to
  // synthesize a copy function; the vector<unique_ptr> member makes that trait spuriously true (the copy ctor
  // is declared but ill-formed if instantiated), so delete it explicitly to keep nanobind from emitting one.
  SNodeRegistry(const SNodeRegistry &) = delete;
  SNodeRegistry &operator=(const SNodeRegistry &) = delete;

  /**
   * Create a new root SNode.
   *
   * Note that this registry takes the ownership of the created SNode.
   *
   * @return Pointer to the created SNode.
   */
  SNode *create_root(Program *prog);

  /**
   * Transfers the ownership of @param snode to the caller.
   *
   * @param snode Returned from create_root()
   * @return The transferred root SNode.
   */
  std::unique_ptr<SNode> finalize(const SNode *snode);

 private:
  std::vector<std::unique_ptr<SNode>> snodes_;
};

}  // namespace quadrants::lang
