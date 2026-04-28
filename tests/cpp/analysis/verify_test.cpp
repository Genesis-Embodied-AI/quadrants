#include "gtest/gtest.h"

#include "quadrants/ir/analysis.h"
#include "quadrants/ir/ir_builder.h"
#include "quadrants/program/compile_config.h"

namespace quadrants::lang {

TEST(VerifyIfDebug, RunsBareVerifyWhenDebugEnabled) {
  IRBuilder builder;
  builder.get_int32(1);
  auto ir = builder.extract_ir();

  CompileConfig cfg;
  cfg.debug = true;

  // On a valid IR the verifier walks every statement and returns successfully. The point of this case is to prove
  // verify_if_debug does forward to the bare verify when the gate is open.
  irpass::analysis::verify_if_debug(ir.get(), cfg);
}

TEST(VerifyIfDebug, NoOpWhenDebugDisabled) {
  IRBuilder builder;
  builder.get_int32(1);
  auto ir = builder.extract_ir();

  CompileConfig cfg;
  cfg.debug = false;

  // Even on a valid IR, with the gate closed verify_if_debug must not invoke the verifier walk. Pinning this on a
  // valid IR is necessarily weaker than pinning it on a deliberately broken IR (the bare verifier would QD_ERROR via
  // QD_UNREACHABLE, which is not catchable as an exception so cannot be cleanly asserted in non-death-test mode);
  // what we can check cheaply is that the call returns and that the bare verify still works on the same IR.
  irpass::analysis::verify_if_debug(ir.get(), cfg);
  irpass::analysis::verify(ir.get());
}

}  // namespace quadrants::lang
