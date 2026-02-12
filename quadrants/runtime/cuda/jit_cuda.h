#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Analysis/LoopAnalysisManager.h"

#include "quadrants/rhi/cuda/cuda_context.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/jit/jit_session.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/program/program.h"
#include "quadrants/system/timer.h"
#include "quadrants/util/file_sequence_writer.h"
#include "quadrants/runtime/cuda/ptx_cache.h"
#include "quadrants/program/program_impl.h"

#define TI_RUNTIME_HOST
#include "quadrants/program/context.h"
#undef TI_RUNTIME_HOST

namespace quadrants::lang {

#if defined(TI_WITH_CUDA)
class JITModuleCUDA : public JITModule {
 private:
  void *module_;

 public:
  explicit JITModuleCUDA(void *module);
  void *lookup_function(const std::string &name) override;
  void call(const std::string &name,
            const std::vector<void *> &arg_pointers,
            const std::vector<int> &arg_sizes) override;
  void launch(const std::string &name,
              std::size_t grid_dim,
              std::size_t block_dim,
              std::size_t dynamic_shared_mem_bytes,
              const std::vector<void *> &arg_pointers,
              const std::vector<int> &arg_sizes) override;
  bool direct_dispatch() const override;
};

class JITSessionCUDA : public JITSession {
 public:
  llvm::DataLayout data_layout;

  JITSessionCUDA(QuadrantsLLVMContext *tlctx,
                 const CompileConfig &config,
                 llvm::DataLayout data_layout,
                 ProgramImpl *program_impl);
  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override;
  llvm::DataLayout get_data_layout() override;

 private:
  class Finalizer : public ProgramImpl::NeedsFinalizing {
   public:
    explicit Finalizer(PtxCache *ptx_cache) : ptx_cache_(ptx_cache) {
    }
    void finalize() override {
      ptx_cache_->dump();
    }
    ~Finalizer() override = default;

   private:
    PtxCache *ptx_cache_;
  };

  std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
  std::unique_ptr<PtxCache> ptx_cache_;
  ProgramImpl *program_impl_;
  std::unique_ptr<Finalizer> finalizer_;
  const CompileConfig &config_;
};

#endif

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch,
    ProgramImpl *program_impl);

}  // namespace quadrants::lang
