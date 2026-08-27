#include "quadrants/codegen/llvm/compiled_kernel_data.h"

#include "llvm/IR/Verifier.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"

#include "quadrants/codegen/llvm/per_task_artifact_cache.h"

namespace quadrants::lang {

static std::unique_ptr<CompiledKernelData> new_llvm_compiled_kernel_data() {
  return std::make_unique<LLVM::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::llvm_creator = new_llvm_compiled_kernel_data;

namespace LLVM {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data) : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(arch_, data_);
}

CompiledKernelData::Err CompiledKernelData::check() const {
  const auto &compiled_data = data_.compiled_data;
  const auto &tasks = compiled_data.tasks;
  // Artifact-backed kernel: tasks came from the on-disk per-task cache, so there's no whole-kernel module here (the
  // CUDA JIT assembles the per-task artifacts). The module-based checks below would null-deref; validate what this
  // representation guarantees instead -- every task has a carrying artifact.
  if (!compiled_data.module) {
    if (compiled_data.per_construct_artifacts.empty() || tasks.empty()) {
      return Err::kCompiledKernelDataBroken;
    }
    return Err::kNoError;
  }
  if (llvm::verifyModule(*compiled_data.module, &llvm::errs())) {
    return Err::kCompiledKernelDataBroken;
  }
  for (const auto &t : tasks) {
    if (compiled_data.module->getFunction(t.name) == nullptr) {
      return Err::kCompiledKernelDataBroken;
    }
  }
  return Err::kNoError;
}

std::string CompiledKernelData::debug_dump_to_string() const {
  auto &data = this->get_internal_data().compiled_data;
  auto *module = data.module.get();
  if (module == nullptr) {  // artifact-backed kernel: no whole-kernel module to print
    return "<artifact-backed kernel: " + std::to_string(data.per_task_artifact_keys.size()) + " per-task artifacts>";
  }
  std::string result;
  llvm::raw_string_ostream oss(result);
  module->print(oss, /*AAW=*/nullptr);
  return oss.str();
}

CompiledKernelData::Err CompiledKernelData::load_impl(const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_llvm(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()), data_, true);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  // Counterpart of the artifact-backed dump below: an empty src_code means the device code is per-task artifacts named
  // by `per_task_artifact_keys`, not LLVM IR. Rebuild them from the cache and leave `module` null (the launcher takes
  // the per-task path when `per_construct_artifacts` is non-empty).
  if (file.src_code().empty()) {
    const auto &keys = data_.compiled_data.per_task_artifact_keys;
    if (keys.empty()) {
      return Err::kParseSrcCodeFailed;
    }
    const PerTaskArtifactCache cache(pertask_artifact_dir_ref());
    std::vector<PerConstructArtifact> arts;
    arts.reserve(keys.size());
    for (const auto &k : keys) {
      PerTaskArtifact rec;
      if (!cache.try_load(k, &rec)) {
        // An artifact was evicted or the cache dir moved: fail the load so the manager recompiles rather than
        // producing a kernel with a missing task.
        QD_DEBUG("artifact-backed kernel: missing per-task artifact {}", k);
        return Err::kParseSrcCodeFailed;
      }
      PerConstructArtifact a;
      a.key = k;
      a.code = std::move(rec.code);
      a.tasks = std::move(rec.tasks);
      a.used_tree_ids = std::move(rec.used_tree_ids);
      a.struct_for_tls_sizes = std::move(rec.struct_for_tls_sizes);
      arts.push_back(std::move(a));
    }
    data_.compiled_data.per_construct_artifacts = std::move(arts);
    return Err::kNoError;
  }
  llvm::SMDiagnostic err;
  auto ret = llvm::parseAssemblyString(file.src_code(), err, llvm_ctx_);
  if (!ret) {  // File not found or Parse failed
    QD_DEBUG("Fail to parse llvm::Module from string: {}", err.getMessage().str());
    return Err::kParseSrcCodeFailed;
  }
  data_.compiled_data.module = std::move(ret);
  llvm::Module *mod = data_.compiled_data.module.get();
  mod->setModuleIdentifier("kernel");
  return Err::kNoError;
}

CompiledKernelData::Err CompiledKernelData::dump_impl(CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(liong::json::print(liong::json::serialize(data_)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  // Artifact-backed kernel: no whole-kernel module, device code lives as per-task artifacts. It still needs a `.qdc`
  // entry, or the whole-kernel cache stays empty and every run re-pays the per-construct path. Metadata already
  // carries `tasks` + `per_task_artifact_keys`, so write an empty src_code instead of IR text.
  if (!data_.compiled_data.module) {
    if (data_.compiled_data.per_task_artifact_keys.empty()) {
      return Err::kSerSrcCodeFailed;  // nothing to point at: genuinely unpersistable
    }
    file.set_src_code(std::string());
    return Err::kNoError;
  }
  std::string str;
  llvm::raw_string_ostream oss(str);
  data_.compiled_data.module->print(oss, /*AAW=*/nullptr);
  file.set_src_code(std::move(str));
  return Err::kNoError;
}

}  // namespace LLVM
}  // namespace quadrants::lang
