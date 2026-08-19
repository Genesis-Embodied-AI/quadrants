#pragma once

#include <memory>
#include <functional>

#include "quadrants/runtime/llvm/llvm_fwd.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/jit/jit_module.h"
#include "quadrants/codegen/llvm/llvm_compiled_data.h"  // PerConstructArtifact (per-task module path)

namespace quadrants::lang {

// Backend JIT compiler for all archs

class QuadrantsLLVMContext;
struct CompileConfig;
class ProgramImpl;

class JITSession {
 protected:
  QuadrantsLLVMContext *tlctx_;
  const CompileConfig &config_;

  std::vector<std::unique_ptr<JITModule>> modules;

 public:
  JITSession(QuadrantsLLVMContext *tlctx, const CompileConfig &config);

  virtual JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg = 0) = 0;

  // Per-task path: compile each self-contained artifact to a loadable module and expose them behind one composite
  // JITModule that resolves tasks by name. CUDA-only; the default errors.
  virtual JITModule *add_module_per_task(std::vector<PerConstructArtifact> artifacts, int max_reg = 0) {
    QD_NOT_IMPLEMENTED
  }

  // virtual void remove_module(JITModule *module) = 0;

  virtual void *lookup(const std::string Name) {
    QD_NOT_IMPLEMENTED
  }

  virtual llvm::DataLayout get_data_layout() = 0;

  static std::unique_ptr<JITSession> create(QuadrantsLLVMContext *tlctx,
                                            const CompileConfig &config,
                                            Arch arch,
                                            ProgramImpl *program_impl);

  virtual ~JITSession() = default;
};

}  // namespace quadrants::lang
