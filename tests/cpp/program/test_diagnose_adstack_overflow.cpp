#include "gtest/gtest.h"

#include "quadrants/program/program.h"

namespace quadrants::lang {

// `Program::register_adstack_sizing_info` + `lookup_adstack_sizing_info` + `diagnose_adstack_overflow_message`
// exercised in isolation. These are pure host-side data-structure operations, so the test does not need
// `materialize_runtime` or any device backend.
TEST(DiagnoseAdstackOverflow, RegistryAndLookup) {
  Program prog(host_arch());

  // Two distinct sizing infos. The pointer is just an identity key for `Program::register_*` -
  // the test does not deref.
  int dummy_a = 0;
  int dummy_b = 0;
  uint32_t id_a = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "kernel_a", /*task=*/0,
                                                    /*allocated_max_sizes=*/{16, 32}, /*size_exprs=*/{});
  uint32_t id_b = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy_b), "kernel_b", /*task=*/3,
                                                    /*allocated_max_sizes=*/{100}, /*size_exprs=*/{});
  EXPECT_NE(id_a, 0u);
  EXPECT_NE(id_b, 0u);
  EXPECT_NE(id_a, id_b);

  // Idempotent re-registration: same pointer returns the same id (and updates metadata in place).
  uint32_t id_a_redo = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "kernel_a_v2",
                                                         /*task=*/2,
                                                         /*allocated_max_sizes=*/{8}, /*size_exprs=*/{});
  EXPECT_EQ(id_a, id_a_redo);
  auto entry = prog.lookup_adstack_sizing_info(id_a);
  ASSERT_TRUE(entry.has_value());
  EXPECT_EQ(entry->kernel_name, "kernel_a_v2");
  EXPECT_EQ(entry->task_id_in_kernel, 2);
  EXPECT_EQ(entry->allocated_max_sizes, std::vector<int>({8}));

  // Lookup with id 0 (sentinel) returns nullopt.
  EXPECT_FALSE(prog.lookup_adstack_sizing_info(0).has_value());
  // Lookup with out-of-range id returns nullopt.
  EXPECT_FALSE(prog.lookup_adstack_sizing_info(static_cast<uint32_t>(0xfffffffful)).has_value());
}

// Diagnose with an unknown / sentinel id falls back to the generic dual-cause body without crashing.
// The body must mention BOTH causes (DLPack bypass and Quadrants bug) so the user has actionable
// recovery information regardless of whether the runtime captured the offending task identity.
TEST(DiagnoseAdstackOverflow, GenericFallbackOnUnknownId) {
  Program prog(host_arch());
  std::string msg = prog.diagnose_adstack_overflow_message(/*task_id=*/0);
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
  uint32_t id = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy), "compute_grad_diag",
                                                  /*task=*/4,
                                                  /*allocated_max_sizes=*/{32, 64}, /*size_exprs=*/{});
  std::string msg = prog.diagnose_adstack_overflow_message(id);
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
  uint32_t id = prog.register_adstack_sizing_info(static_cast<const void *>(&identity), "any_kernel",
                                                  /*task=*/0,
                                                  /*allocated_max_sizes=*/{16}, std::move(size_exprs));
  std::string msg = prog.diagnose_adstack_overflow_message(id);
  EXPECT_NE(msg.find("Offending task"), std::string::npos);
  EXPECT_NE(msg.find("any_kernel"), std::string::npos);
  // Empty `nodes` triggers the `nodes.empty()` skip in the rerun loop; required size shows as `?`.
  EXPECT_NE(msg.find("Synchronous sizer rerun: required max_size = [?]"), std::string::npos);
  EXPECT_NE(msg.find("DLPack"), std::string::npos);
  EXPECT_NE(msg.find("Quadrants bug"), std::string::npos);
}

// `update_adstack_sizing_info_size_exprs` overwrites just the size_exprs without disturbing the rest
// of the entry. Used by the LLVM launcher on every launch to keep the registry in sync with the live
// `OffloadedTask::ad_stack`.
TEST(DiagnoseAdstackOverflow, UpdateSizeExprsRefreshesEntry) {
  Program prog(host_arch());
  int identity = 0;
  uint32_t id = prog.register_adstack_sizing_info(static_cast<const void *>(&identity), "k",
                                                  /*task=*/0, /*allocated_max_sizes=*/{8},
                                                  /*size_exprs=*/{});
  // Empty registry size_exprs => no rerun line.
  EXPECT_EQ(prog.diagnose_adstack_overflow_message(id).find("Synchronous sizer rerun"), std::string::npos);
  prog.update_adstack_sizing_info_size_exprs(id, std::vector<SerializedSizeExpr>(1));
  // After refresh, the rerun walks the new size_exprs.
  EXPECT_NE(prog.diagnose_adstack_overflow_message(id).find("Synchronous sizer rerun: required max_size = [?]"),
            std::string::npos);
}

}  // namespace quadrants::lang
