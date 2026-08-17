#include "compile_config.h"

#include <thread>
#include "quadrants/rhi/arch.h"
#include "quadrants/util/offline_cache.h"

namespace quadrants::lang {

CompileConfig::CompileConfig() {
  // Assignments for the computed (non-literal) defaults are generated from
  // tools/config_codegen/schema.py; literal defaults live as in-class member
  // initializers in compile_config.h. Emitted in schema order so a computed
  // default may reference an earlier one (e.g. simd_width uses arch).
#include "quadrants/program/compile_config.ctor.generated.inc"
}

void CompileConfig::fit() {
  if (debug) {
    // TODO: allow users to run in debug mode without out-of-bound checks
    check_out_of_bound = true;
  }
  if (arch_uses_spirv(arch)) {
    demote_dense_struct_fors = true;
  }
}

}  // namespace quadrants::lang
