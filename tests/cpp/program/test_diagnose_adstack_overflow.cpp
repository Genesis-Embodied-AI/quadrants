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
                                                    /*allocated_max_sizes=*/{16, 32});
  uint32_t id_b = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy_b), "kernel_b", /*task=*/3,
                                                    /*allocated_max_sizes=*/{100});
  EXPECT_NE(id_a, 0u);
  EXPECT_NE(id_b, 0u);
  EXPECT_NE(id_a, id_b);

  // Idempotent re-registration: same pointer returns the same id (and updates metadata in place).
  uint32_t id_a_redo = prog.register_adstack_sizing_info(static_cast<const void *>(&dummy_a), "kernel_a_v2",
                                                         /*task=*/2,
                                                         /*allocated_max_sizes=*/{8});
  EXPECT_EQ(id_a, id_a_redo);
  const auto *entry = prog.lookup_adstack_sizing_info(id_a);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->kernel_name, "kernel_a_v2");
  EXPECT_EQ(entry->task_id_in_kernel, 2);
  EXPECT_EQ(entry->allocated_max_sizes, std::vector<int>({8}));

  // Lookup with id 0 (sentinel) returns nullptr.
  EXPECT_EQ(prog.lookup_adstack_sizing_info(0), nullptr);
  // Lookup with out-of-range id returns nullptr.
  EXPECT_EQ(prog.lookup_adstack_sizing_info(static_cast<uint32_t>(0xfffffffful)), nullptr);
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
                                                  /*allocated_max_sizes=*/{32, 64});
  std::string msg = prog.diagnose_adstack_overflow_message(id);
  EXPECT_NE(msg.find("Offending task"), std::string::npos);
  EXPECT_NE(msg.find("compute_grad_diag"), std::string::npos);
  EXPECT_NE(msg.find("offload task #4"), std::string::npos);
  EXPECT_NE(msg.find("[32, 64]"), std::string::npos);
  // Dual-cause body still present.
  EXPECT_NE(msg.find("DLPack"), std::string::npos);
  EXPECT_NE(msg.find("Quadrants bug"), std::string::npos);
}

}  // namespace quadrants::lang
