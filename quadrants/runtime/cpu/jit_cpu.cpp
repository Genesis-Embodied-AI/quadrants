// A LLVM JIT compiler for CPU archs wrapper

#include <memory>

#ifdef QD_WITH_LLVM
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
// From https://github.com/JuliaLang/julia/pull/43664
#if defined(__APPLE__) && defined(__aarch64__)
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#else
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#endif
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Host.h"

#endif

#include "quadrants/jit/jit_module.h"
#include "quadrants/util/lang_util.h"
#include "quadrants/program/program.h"
#include "quadrants/jit/jit_session.h"
#include "quadrants/util/file_sequence_writer.h"
#include "quadrants/runtime/llvm/llvm_context.h"

namespace quadrants::lang {

#ifdef QD_WITH_LLVM
using namespace llvm;
using namespace llvm::orc;
#if defined(__APPLE__) && defined(__aarch64__)
typedef orc::ObjectLinkingLayer ObjLayerT;
#else
typedef orc::RTDyldObjectLinkingLayer ObjLayerT;
#endif
#endif

std::pair<JITTargetMachineBuilder, llvm::DataLayout> get_host_target_info() {
  auto expected_jtmb = JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb)
    QD_ERROR("LLVM TargetMachineBuilder has failed.");
  auto jtmb = *expected_jtmb;
  auto expected_data_layout = jtmb.getDefaultDataLayoutForTarget();
  if (!expected_data_layout) {
    QD_ERROR("LLVM TargetMachineBuilder has failed when getting data layout.");
  }
  auto data_layout = *expected_data_layout;
  return std::make_pair(jtmb, data_layout);
}

class JITSessionCPU;

class JITModuleCPU : public JITModule {
 private:
  JITSessionCPU *session_;
  JITDylib *dylib_;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib)
      : session_(session), dylib_(dylib) {
  }

  void *lookup_function(const std::string &name) override;

  bool direct_dispatch() const override {
    return true;
  }
};

class JITSessionCPU : public JITSession {
 private:
  ExecutionSession es_;
  ObjLayerT object_layer_;
  IRCompileLayer compile_layer_;
  DataLayout dl_;
  MangleAndInterner mangle_;
  std::mutex mut_;
  std::vector<llvm::orc::JITDylib *> all_libs_;
  int module_counter_;
  SectionMemoryManager *memory_manager_;

 public:
  JITSessionCPU(QuadrantsLLVMContext *tlctx,
                std::unique_ptr<ExecutorProcessControl> EPC,
                const CompileConfig &config,
                JITTargetMachineBuilder JTMB,
                DataLayout DL)
      : JITSession(tlctx, config),
        es_(std::move(EPC)),
#if defined(__APPLE__) && defined(__aarch64__)
        object_layer_(es_),
#else
        object_layer_(es_,
                      [&](const MemoryBuffer &) {
                        auto smgr = std::make_unique<SectionMemoryManager>();
                        memory_manager_ = smgr.get();
                        return smgr;
                      }),
#endif
        compile_layer_(es_,
                       object_layer_,
                       std::make_unique<ConcurrentIRCompiler>(JTMB)),
        dl_(DL),
        mangle_(es_, this->dl_),
        module_counter_(0),
        memory_manager_(nullptr) {
#if defined(__linux__) && defined(__aarch64__)
    // On ELF-based targets LLVM's ORC JIT expects the object layer to claim
    // responsibility for every symbol emitted by the module. Without this the
    // resolver may encounter symbols that were not pre-registered in the
    // materialization responsibility set and abort with
    // "Resolving symbol outside this responsibility set" (observed on
    // manylinux aarch64).
    object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
    object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
#else
    if (JTMB.getTargetTriple().isOSBinFormatCOFF()) {
      object_layer_.setOverrideObjectFlagsWithResponsibilityFlags(true);
      object_layer_.setAutoClaimResponsibilityForObjectSymbols(true);
    }
#endif
  }

  ~JITSessionCPU() override {
    std::lock_guard<std::mutex> _(mut_);
    if (memory_manager_)
      memory_manager_->deregisterEHFrames();
    if (auto Err = es_.endSession())
      es_.reportError(std::move(Err));
  }

  DataLayout get_data_layout() override {
    return dl_;
  }

  JITModule *add_module(std::unique_ptr<llvm::Module> M, int max_reg) override {
    QD_ASSERT(max_reg == 0);  // No need to specify max_reg on CPUs
    QD_ASSERT(M);
    std::lock_guard<std::mutex> _(mut_);
    auto dylib_expect = es_.createJITDylib(fmt::format("{}", module_counter_));
    QD_ASSERT(dylib_expect);
    auto &dylib = dylib_expect.get();
    dylib.addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            dl_.getGlobalPrefix())));
    auto *thread_safe_context =
        this->tlctx_->get_this_thread_thread_safe_context();
    cantFail(compile_layer_.add(
        dylib,
        llvm::orc::ThreadSafeModule(std::move(M), *thread_safe_context)));
    all_libs_.push_back(&dylib);
    auto new_module = std::make_unique<JITModuleCPU>(this, &dylib);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    module_counter_++;
    return new_module_raw_ptr;
  }

  void *lookup(const std::string Name) override {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup(all_libs_, mangle_(Name));
#else
    auto symbol = es_.lookup(all_libs_, es_.intern(Name));
#endif
    if (!symbol)
      QD_ERROR("Function \"{}\" not found", Name);
    return symbol->getAddress().toPtr<void *>();
  }

  void *lookup_in_module(JITDylib *lib, const std::string Name) {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup({lib}, mangle_(Name));
#else
    auto symbol = es_.lookup({lib}, es_.intern(Name));
#endif
    if (!symbol)
      QD_ERROR("Function \"{}\" not found", Name);
    return symbol->getAddress().toPtr<void *>();
  }
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session_->lookup_in_module(dylib_, name);
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(
    QuadrantsLLVMContext *tlctx,
    const CompileConfig &config,
    Arch arch) {
  QD_ASSERT(arch_is_cpu(arch));
  auto target_info = get_host_target_info();
  auto EPC = SelfExecutorProcessControl::Create();
  QD_ASSERT(EPC);
  return std::make_unique<JITSessionCPU>(tlctx, std::move(*EPC), config,
                                         target_info.first, target_info.second);
}

}  // namespace quadrants::lang
