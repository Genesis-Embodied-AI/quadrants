#include "quadrants/runtime/llvm/runtime_module_utils.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"

namespace quadrants::lang {

void patch_runtime_intrinsic(llvm::Module &module,
                             std::string name,
                             llvm::Intrinsic::ID intrin,
                             bool ret,
                             std::vector<llvm::Type *> types,
                             std::vector<llvm::Value *> extra_args) {
  auto *ctx = &module.getContext();
  auto func = module.getFunction(name);
  if (!func) {
    return;
  }
  func->deleteBody();
  auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
  llvm::IRBuilder<> builder(*ctx);
  builder.SetInsertPoint(bb);
  std::vector<llvm::Value *> args;
  for (auto &arg : func->args())
    args.push_back(&arg);
  args.insert(args.end(), extra_args.begin(), extra_args.end());
  if (ret) {
    builder.CreateRet(builder.CreateIntrinsic(intrin, types, args));
  } else {
    builder.CreateIntrinsic(intrin, types, args);
    builder.CreateRetVoid();
  }
  mark_runtime_function_inline(func);
}

void mark_runtime_function_inline(llvm::Function *f) {
  for (auto &B : *f)
    for (auto &I : B) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (auto func = call->getCalledFunction(); func && func->getName() == "mark_force_no_inline") {
          // Found "mark_force_no_inline". Do not inline.
          return;
        }
      }
    }
  f->removeFnAttr(llvm::Attribute::OptimizeNone);
  f->removeFnAttr(llvm::Attribute::NoInline);
  f->addFnAttr(llvm::Attribute::AlwaysInline);
}

}  // namespace quadrants::lang
