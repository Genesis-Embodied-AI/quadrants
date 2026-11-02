#pragma once

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/raw_ostream.h"

namespace quadrants::lang {

// LLVM's bitcode reader restores canonical intrinsic attributes, discarding the AlwaysInline added by our runtime
// loader. Carry that preparation flag through metadata so cache hits and misses produce the same runtime IR (and
// consequently the same downstream HSACO key). All carrier metadata is removed before the module is consumed.
inline constexpr const char *kRuntimeInlineIntrinsics = "quadrants.runtime.inline_intrinsics";

// Serialize a prepared LLVM module while preserving its added intrinsic attributes.
inline std::string serialize_runtime_bitcode(llvm::Module &module) {
  llvm::NamedMDNode *metadata = module.getOrInsertNamedMetadata(kRuntimeInlineIntrinsics);
  for (llvm::Function &function : module) {
    if (function.isIntrinsic() && function.hasFnAttribute(llvm::Attribute::AlwaysInline)) {
      metadata->addOperand(llvm::MDNode::get(module.getContext(), llvm::ValueAsMetadata::get(&function)));
    }
  }
  std::string module_serialized;
  llvm::raw_string_ostream module_serialized_writer(module_serialized);
  llvm::WriteBitcodeToFile(module, module_serialized_writer);
  module.eraseNamedMetadata(metadata);
  return module_serialized;
}

// Parse serialized runtime bytes into the supplied LLVM context.
inline llvm::Expected<std::unique_ptr<llvm::Module>> parse_runtime_bitcode(const std::string &module_serialized,
                                                                           llvm::LLVMContext &context) {
  // This is prepared intermediate IR, like clone_module_to_context's input. Device-library CommonLinkage functions
  // are resolved later; raw input files are verified by LlvmModuleBitcodeLoader before reaching this boundary.
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      llvm::parseBitcodeFile(llvm::MemoryBufferRef(module_serialized, "runtime_bitcode"), context);
  if (!module) {
    return module.takeError();
  }
  if (llvm::NamedMDNode *metadata = module.get()->getNamedMetadata(kRuntimeInlineIntrinsics)) {
    for (llvm::MDNode *node : metadata->operands()) {
      llvm::ValueAsMetadata *value = llvm::cast<llvm::ValueAsMetadata>(node->getOperand(0));
      llvm::cast<llvm::Function>(value->getValue())->addFnAttr(llvm::Attribute::AlwaysInline);
    }
    module.get()->eraseNamedMetadata(metadata);
  }
  return module;
}

}  // namespace quadrants::lang
