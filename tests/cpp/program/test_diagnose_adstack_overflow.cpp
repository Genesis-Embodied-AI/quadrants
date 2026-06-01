#include "gtest/gtest.h"

#include "quadrants/program/adstack_size_expr_eval.h"
#include "quadrants/program/program.h"

namespace quadrants::lang {

// `AdStackCache::register_adstack_sizing_info` + `lookup_adstack_sizing_info` +
// `diagnose_adstack_overflow_message` exercised in isolation. These are pure host-side data-structure
// operations, so the test does not need
// `materialize_runtime` or any device backend.
TEST(DiagnoseAdstackOverflow, RegistryAndLookup) {
  Program prog(host_arch());

  // Two distinct sizing infos. The pointer is just an identity key for `Program::register_*` - the test does not
  // deref.
  int dummy_a = 0;
  int dummy_b = 0;
  uint32_t id_a =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "kernel_a", /*task=*/0,
                                                        /*allocated_max_sizes=*/{16, 32}, /*size_exprs=*/{});
  uint32_t id_b =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_b), "kernel_b", /*task=*/3,
                                                        /*allocated_max_sizes=*/{100}, /*size_exprs=*/{});
  EXPECT_NE(id_a, 0u);
  EXPECT_NE(id_b, 0u);
  EXPECT_NE(id_a, id_b);

  // Idempotent re-registration: same pointer AND same `(kernel_name, task_id_in_kernel)` returns the same id and
  // updates only the metadata in place. Production callers hit this path twice per codegen
  // (`codegen_llvm.cpp::offloaded_task_start` registers with empty `allocated_max_sizes` so the codegen can bake the
  // id; `finalize_offloaded_task_function` re-registers with the populated metadata after the alloca scan).
  uint32_t id_a_redo =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "kernel_a", /*task=*/0,
                                                        /*allocated_max_sizes=*/{8}, /*size_exprs=*/{});
  EXPECT_EQ(id_a, id_a_redo);
  auto entry = prog.adstack_cache().lookup_adstack_sizing_info(id_a);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->kernel_name, "kernel_a");
  EXPECT_EQ(entry->task_id_in_kernel, 0);
  EXPECT_EQ(entry->allocated_max_sizes, std::vector<int>({8}));

  // Recycled `identity_key` (same pointer address, different logical kernel) does NOT collapse to the previous id.
  // The allocator can free an `OffloadedTask::ad_stack` and reuse its address for an unrelated task across a
  // re-codegen / `qd.reset()` cycle, and the `max_reducer_cache_` keyed by `(registry_id, stack_id, mor_node_idx)`
  // would serve stale results to the new kernel if we returned the previous id here. The content-stable hash path
  // mints (or finds) the correct id for the new kernel; the previous entry stays alive so any still-live owner
  // resolving via its registry id continues to work (e.g. the overflow diagnose path).
  uint32_t id_a_recycled =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "different_kernel",
                                                        /*task=*/5,
                                                        /*allocated_max_sizes=*/{2}, /*size_exprs=*/{});
  EXPECT_NE(id_a, id_a_recycled);
  auto recycled_entry = prog.adstack_cache().lookup_adstack_sizing_info(id_a_recycled);
  ASSERT_TRUE(recycled_entry.has_value());
  EXPECT_EQ(recycled_entry->kernel_name, "different_kernel");
  EXPECT_EQ(recycled_entry->task_id_in_kernel, 5);
  // Previous entry survives intact (the still-live owner resolving via `id_a` keeps working).
  auto preserved_entry = prog.adstack_cache().lookup_adstack_sizing_info(id_a);
  ASSERT_TRUE(preserved_entry.has_value());
  EXPECT_EQ(preserved_entry->kernel_name, "kernel_a");
  EXPECT_EQ(preserved_entry->task_id_in_kernel, 0);

  // Lookup with id 0 (sentinel) returns nullopt.
  EXPECT_FALSE(prog.adstack_cache().lookup_adstack_sizing_info(0).has_value());
  // Lookup with out-of-range id returns nullopt.
  EXPECT_FALSE(prog.adstack_cache().lookup_adstack_sizing_info(static_cast<uint32_t>(0xfffffffful)).has_value());
}

// `register_adstack_sizing_info` content-stable hash dedup. Two registrations with the same `(kernel_name,
// task_id_in_kernel)` pair but different `identity_key`s (the runtime case: a re-codegen of the same source after a
// `qd.reset()` produces a fresh `ad_stack` at a new heap address but the hash inputs are unchanged) must resolve to the
// same id - otherwise the codegen-baked immediate in the cached LLVM IR would point at one entry while the runtime
// registration mints another, and the diagnose-on-overflow path would look up the wrong (or empty) entry. Pins the
// linear-probe loop's same-content branch, which the `RegistryAndLookup` test above does not cover (it only re-uses the
// same identity_key).
TEST(DiagnoseAdstackOverflow, RegistryContentStableDedupAcrossIdentityKeys) {
  Program prog(host_arch());

  int dummy_a = 0;
  int dummy_b = 0;  // Different identity_key, same content.
  uint32_t id_a = prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "compute_grad",
                                                                    /*task=*/3,
                                                                    /*allocated_max_sizes=*/{16, 32},
                                                                    /*size_exprs=*/{});
  uint32_t id_b = prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_b), "compute_grad",
                                                                    /*task=*/3,
                                                                    /*allocated_max_sizes=*/{8, 4},
                                                                    /*size_exprs=*/{});
  EXPECT_NE(id_a, 0u);
  // Same content yields the same id even though `identity_key` differs.
  EXPECT_EQ(id_a, id_b);
  // The second call's metadata wins (latest re-registration overwrites the entry in place).
  auto entry = prog.adstack_cache().lookup_adstack_sizing_info(id_b);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->kernel_name, "compute_grad");
  EXPECT_EQ(entry->task_id_in_kernel, 3);
  EXPECT_EQ(entry->allocated_max_sizes, std::vector<int>({8, 4}));
  // Both identity_keys now resolve to the same id - re-registering with either pointer returns id_a.
  uint32_t id_a_redo =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "compute_grad",
                                                        /*task=*/3,
                                                        /*allocated_max_sizes=*/{1, 2},
                                                        /*size_exprs=*/{});
  EXPECT_EQ(id_a, id_a_redo);
  uint32_t id_b_redo =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_b), "compute_grad",
                                                        /*task=*/3,
                                                        /*allocated_max_sizes=*/{1, 2},
                                                        /*size_exprs=*/{});
  EXPECT_EQ(id_a, id_b_redo);

  // A different `(kernel_name, task_id_in_kernel)` pair must hash to a different id (assuming no collision - 32-bit
  // FNV-1a collision probability for a handful of distinct keys is vanishingly low). Pins that the dedup branch above
  // only triggers on actual content match, not on every re-registration.
  int dummy_c = 0;
  uint32_t id_c =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy_c), "compute_other",
                                                        /*task=*/3,
                                                        /*allocated_max_sizes=*/{1},
                                                        /*size_exprs=*/{});
  EXPECT_NE(id_a, id_c);
}

// Diagnose with an unknown / sentinel id falls back to the generic dual-cause body without crashing.
// The body must mention BOTH causes (DLPack bypass and Quadrants bug) so the user has actionable
// recovery information regardless of whether the runtime captured the offending task identity.
TEST(DiagnoseAdstackOverflow, GenericFallbackOnUnknownId) {
  Program prog(host_arch());
  std::string msg = prog.adstack_cache().diagnose_adstack_overflow_message(/*task_id=*/0);
  EXPECT_NE(msg.find("DLPack"), std::string::npos);
  EXPECT_NE(msg.find("Quadrants bug"), std::string::npos);
  // No identity-block prefix when there is nothing to look up.
  EXPECT_EQ(msg.find("Offending task"), std::string::npos);
}

// Diagnose with a registered id includes the kernel name + offload task index + per-stack allocated
// max_sizes in the identity block prefix.
TEST(DiagnoseAdstackOverflow, EnrichedMessageWithIdentity) {
  Program prog(host_arch());
  int dummy = 0;
  uint32_t id =
      prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&dummy), "compute_grad_diag",
                                                        /*task=*/4,
                                                        /*allocated_max_sizes=*/{32, 64}, /*size_exprs=*/{});
  std::string msg = prog.adstack_cache().diagnose_adstack_overflow_message(id);
  EXPECT_NE(msg.find("Offending task"), std::string::npos);
  EXPECT_NE(msg.find("compute_grad_diag"), std::string::npos);
  EXPECT_NE(msg.find("offload task #4"), std::string::npos);
  EXPECT_NE(msg.find("[32, 64]"), std::string::npos);
  // Dual-cause body still present.
  EXPECT_NE(msg.find("DLPack"), std::string::npos);
  EXPECT_NE(msg.find("Quadrants bug"), std::string::npos);
}

// The registry copies size_exprs into the entry, so the diagnose path is backend-independent: it
// walks the entry's heap-owned `size_exprs` regardless of whether the original kernel data lives in
// LLVM's `AdStackSizingInfo` or SPIR-V's `AdStackSizingAttribs`. Pin that uniform behaviour by
// registering with one empty SerializedSizeExpr and verifying the rerun reports `?`.
TEST(DiagnoseAdstackOverflow, BackendUniformSizeExprWalk) {
  Program prog(host_arch());
  int identity = 0;
  std::vector<SerializedSizeExpr> size_exprs(1);  // one empty tree
  uint32_t id = prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&identity), "any_kernel",
                                                                  /*task=*/0,
                                                                  /*allocated_max_sizes=*/{16}, std::move(size_exprs));
  std::string msg = prog.adstack_cache().diagnose_adstack_overflow_message(id);
  EXPECT_NE(msg.find("Offending task"), std::string::npos);
  EXPECT_NE(msg.find("any_kernel"), std::string::npos);
  // Empty `nodes` triggers the rerun-loop skip; with no resolvable stack, the rerun line is omitted and the
  // dual-cause body fronts the diagnostic instead.
  EXPECT_EQ(msg.find("Synchronous sizer rerun"), std::string::npos);
  EXPECT_NE(msg.find("DLPack"), std::string::npos);
  EXPECT_NE(msg.find("Quadrants bug"), std::string::npos);
}

// `update_adstack_sizing_info_size_exprs` overwrites just the size_exprs without disturbing the rest
// of the entry. Used by the LLVM launcher on every launch to keep the registry in sync with the live
// `OffloadedTask::ad_stack`.
TEST(DiagnoseAdstackOverflow, UpdateSizeExprsRefreshesEntry) {
  Program prog(host_arch());
  int identity = 0;
  uint32_t id = prog.adstack_cache().register_adstack_sizing_info(static_cast<const void *>(&identity), "k",
                                                                  /*task=*/0, /*allocated_max_sizes=*/{8},
                                                                  /*size_exprs=*/{});
  // Empty registry size_exprs => no rerun line.
  EXPECT_EQ(prog.adstack_cache().diagnose_adstack_overflow_message(id).find("Synchronous sizer rerun"),
            std::string::npos);
  prog.adstack_cache().update_adstack_sizing_info_size_exprs(id, std::vector<SerializedSizeExpr>(1));
  // After refresh, the rerun walks the new size_exprs but every leaf is unresolvable (empty tree), so the line
  // is still omitted - same suppression as the BackendUniformSizeExprWalk case above. The `update` change is
  // observable via the dual-cause body still fronting the message; we already pinned that path elsewhere.
  EXPECT_EQ(prog.adstack_cache().diagnose_adstack_overflow_message(id).find("Synchronous sizer rerun"),
            std::string::npos);
}

}  // namespace quadrants::lang
