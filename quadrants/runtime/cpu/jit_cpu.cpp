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
#include "llvm/Support/MemoryBuffer.h"
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
#include "quadrants/codegen/llvm/per_task_artifact_cache.h"

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
  // The whole-module path resolves in one dylib; the per-task path (add_module_per_task) resolves across N, one per
  // task. A task's entry symbol lives in exactly one of them, so a by-name lookup that searches them all is
  // unambiguous (only the unique task entry names are ever looked up; duplicated helper symbols never are).
  std::vector<JITDylib *> dylibs_;

 public:
  JITModuleCPU(JITSessionCPU *session, JITDylib *dylib) : session_(session), dylibs_{dylib} {
  }
  JITModuleCPU(JITSessionCPU *session, std::vector<JITDylib *> dylibs) : session_(session), dylibs_(std::move(dylibs)) {
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
  // Lazily built host PIC target machine for the per-task object serialize path (add_module_per_task); reused across
  // all misses in a session.
  std::unique_ptr<llvm::TargetMachine> pertask_target_machine_;

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
        compile_layer_(es_, object_layer_, std::make_unique<ConcurrentIRCompiler>(JTMB)),
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
    dylib.addGenerator(cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(dl_.getGlobalPrefix())));
    auto *thread_safe_context = this->tlctx_->get_this_thread_thread_safe_context();
    cantFail(compile_layer_.add(dylib, llvm::orc::ThreadSafeModule(std::move(M), *thread_safe_context)));
    all_libs_.push_back(&dylib);
    auto new_module = std::make_unique<JITModuleCPU>(this, &dylib);
    auto new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
    module_counter_++;
    return new_module_raw_ptr;
  }

  JITModule *add_module_per_task(std::vector<PerConstructArtifact> artifacts, int max_reg) override {
    QD_ASSERT(max_reg == 0);  // No need to specify max_reg on CPUs
    std::lock_guard<std::mutex> _(mut_);

    // Cross-process fill site (mirror of runtime/cuda/jit_cuda.cpp): a hit already carries the cached host object in
    // `code`, used verbatim; a miss compiles the per-task module to a host object and stores it -- object bytes plus
    // the launch metadata that must travel with it -- under the task's IR key so a later process skips this
    // compilation. No-ops when the dir is empty (offline cache off).
    const PerTaskArtifactCache artifact_cache(pertask_artifact_dir_ref());

    // One JITDylib per task (a task's entry symbol lives in exactly one), so per-task objects never collide on a
    // shared helper / global symbol -- the CPU analog of CUDA's one-CUmodule-per-task, and why CPU needs no relink.
    std::vector<llvm::orc::JITDylib *> dylibs;
    dylibs.reserve(artifacts.size());
    for (auto &art : artifacts) {
      // Advance the counter up front: the error path below throws mid-iteration and the created dylib lingers in the
      // session, so reusing this id on a later call would collide on the dylib name (createJITDylib would fail).
      const int mod_id = module_counter_++;
      auto dylib_expect = es_.createJITDylib(fmt::format("pertask_{}", mod_id));
      QD_ASSERT(dylib_expect);
      auto &dylib = dylib_expect.get();
      dylib.addGenerator(
          cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(dl_.getGlobalPrefix())));

      std::unique_ptr<llvm::MemoryBuffer> obj;
      const bool from_cache = !art.code.empty();
      if (from_cache) {
        // Hit: the cached bytes are a relocatable host object; load them straight into the object layer.
        obj = llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(art.code.data(), art.code.size()),
                                                   fmt::format("pertask_{}", mod_id));
      } else {
        QD_ASSERT(art.module);
        obj = compile_module_to_object(*art.module);
        if (!art.key.empty()) {
          PerTaskArtifact rec;
          rec.tasks = art.tasks;
          rec.used_tree_ids = art.used_tree_ids;
          rec.struct_for_tls_sizes = art.struct_for_tls_sizes;
          rec.code.assign(obj->getBufferStart(), obj->getBufferEnd());
          artifact_cache.store(art.key, rec);
        }
      }
      // `object_layer_.add` parses the object eagerly, so malformed bytes surface here -- offline-cache corruption on
      // the hit path, or (defensively) a bad freshly-compiled object we just stored on the miss path. Don't `cantFail`
      // -- that aborts the whole process on every launch through a corrupt cache. Drop the on-disk entry (either path
      // may have written one) so a later process recompiles and refills it, and raise a catchable error instead of
      // terminating (mirrors the CUDA per-task load, which QD_ERRORs on a bad module).
      if (auto err = object_layer_.add(dylib, std::move(obj))) {
        if (!art.key.empty()) {
          artifact_cache.erase(art.key);
        }
        QD_ERROR("Failed to load per-task CPU object into the JIT (offline cache may be corrupt): {}",
                 llvm::toString(std::move(err)));
      }
      // `object_layer_.add` only registers a materialization unit, so it catches parse errors but not a corrupt
      // relocation / undefined reference, which fails later during linking. That failure would otherwise surface at
      // launch in lookup_in_modules -- with no key at hand to invalidate the record, poisoning every future process.
      // Force materialization here (each task's entry symbol resolves in exactly this self-contained dylib) so a
      // deferred link failure is caught while `art.key` is still available, then erased and raised catchably.
      for (const auto &task : art.tasks) {
#ifdef __APPLE__
        auto sym = es_.lookup({&dylib}, mangle_(task.name));
#else
        auto sym = es_.lookup({&dylib}, es_.intern(task.name));
#endif
        if (!sym) {
          if (!art.key.empty()) {
            artifact_cache.erase(art.key);
          }
          QD_ERROR("Failed to materialize per-task CPU object for \"{}\" (offline cache may be corrupt): {}", task.name,
                   llvm::toString(sym.takeError()));
        }
      }
      dylibs.push_back(&dylib);
    }

    auto new_module = std::make_unique<JITModuleCPU>(this, std::move(dylibs));
    auto *new_module_raw_ptr = new_module.get();
    modules.push_back(std::move(new_module));
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

  void *lookup_in_modules(const std::vector<JITDylib *> &libs, const std::string Name) {
    std::lock_guard<std::mutex> _(mut_);
#ifdef __APPLE__
    auto symbol = es_.lookup(libs, mangle_(Name));
#else
    auto symbol = es_.lookup(libs, es_.intern(Name));
#endif
    if (!symbol)
      QD_ERROR("Function \"{}\" not found", Name);
    return symbol->getAddress().toPtr<void *>();
  }

 private:
  // Serialize a self-contained per-task module to a host relocatable object -- the "serialize" half of the per-task
  // disk tier, called on a cache miss. Build the target machine the same way as KernelCodeGenCPU::optimize_module
  // (host CPU, PIC so the object is loadable by the ORC object layer) so the emitted object matches the module that
  // optimize_module already produced.
  std::unique_ptr<llvm::MemoryBuffer> compile_module_to_object(llvm::Module &M) {
    if (!pertask_target_machine_) {
      auto expected_jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
      if (!expected_jtmb) {
        QD_ERROR("LLVM TargetMachineBuilder has failed.");
      }
      // Build the target machine straight from the JTMB so it carries detectHost()'s *explicit* host feature vector,
      // exactly as the whole-kernel path does via ConcurrentIRCompiler(JTMB). Passing the host CPU name with an empty
      // feature string instead selects that CPU model's *default* features, which can be a superset of what the
      // running core actually enables and emits illegal instructions at kernel launch on some hosts. PIC so the
      // emitted object is loadable by the ORC object layer.
      auto jtmb = std::move(*expected_jtmb);
      jtmb.setRelocationModel(llvm::Reloc::PIC_);
      jtmb.setCodeModel(llvm::CodeModel::Small);
      jtmb.setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
      llvm::TargetOptions &options = jtmb.getOptions();
      if (config_.fast_math) {
        options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
        options.NoInfsFPMath = 1;
        options.NoNaNsFPMath = 1;
      } else {
        options.AllowFPOpFusion = llvm::FPOpFusion::Strict;
        options.NoInfsFPMath = 0;
        options.NoNaNsFPMath = 0;
      }
      auto expected_tm = jtmb.createTargetMachine();
      QD_ERROR_UNLESS(expected_tm, "Could not allocate target machine!");
      pertask_target_machine_ = std::move(*expected_tm);
    }
    M.setDataLayout(pertask_target_machine_->createDataLayout());
    llvm::orc::SimpleCompiler compiler(*pertask_target_machine_);
    auto obj = compiler(M);
    if (!obj) {
      QD_ERROR("Per-task CPU object compilation failed");
    }
    return std::move(*obj);
  }
};

void *JITModuleCPU::lookup_function(const std::string &name) {
  return session_->lookup_in_modules(dylibs_, name);
}

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(QuadrantsLLVMContext *tlctx,
                                                        const CompileConfig &config,
                                                        Arch arch) {
  QD_ASSERT(arch_is_cpu(arch));
  auto target_info = get_host_target_info();
  auto EPC = SelfExecutorProcessControl::Create();
  QD_ASSERT(EPC);
  return std::make_unique<JITSessionCPU>(tlctx, std::move(*EPC), config, target_info.first, target_info.second);
}

}  // namespace quadrants::lang
