// Driver class for kernel codegen

#include "codegen.h"

#if defined(QD_WITH_LLVM)
#include "quadrants/codegen/cpu/codegen_cpu.h"
#include "quadrants/runtime/program_impls/llvm/llvm_program.h"
#endif
#if defined(QD_WITH_CUDA)
#include "quadrants/codegen/cuda/codegen_cuda.h"
#endif
#if defined(QD_WITH_AMDGPU)
#include "quadrants/codegen/amdgpu/codegen_amdgpu.h"
#endif
#include "quadrants/system/timer.h"
#include "quadrants/ir/analysis.h"
#include "quadrants/ir/statements.h"
#include "quadrants/ir/transforms.h"
#include "quadrants/analysis/offline_cache_util.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

namespace quadrants::lang {

KernelCodeGen::KernelCodeGen(const CompileConfig &compile_config,
                             const Kernel *kernel,
                             IRNode *ir,
                             QuadrantsLLVMContext &tlctx)
    : prog(kernel->program), kernel(kernel), ir(ir), compile_config_(compile_config), tlctx_(tlctx) {
}

std::unique_ptr<KernelCodeGen> KernelCodeGen::create(const CompileConfig &compile_config,
                                                     const Kernel *kernel,
                                                     IRNode *ir,
                                                     QuadrantsLLVMContext &tlctx) {
#ifdef QD_WITH_LLVM
  const auto arch = compile_config.arch;
  if (arch_is_cpu(arch)) {
    return std::make_unique<KernelCodeGenCPU>(compile_config, kernel, ir, tlctx);
  } else if (arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    return std::make_unique<KernelCodeGenCUDA>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    return std::make_unique<KernelCodeGenAMDGPU>(compile_config, kernel, ir, tlctx);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else {
    QD_NOT_IMPLEMENTED
  }
#else
  QD_ERROR("Llvm disabled");
#endif
}
#ifdef QD_WITH_LLVM

LLVMCompiledKernel KernelCodeGen::compile_kernel_to_module() {
  auto block = dynamic_cast<Block *>(ir);
  auto &worker = get_llvm_program(kernel->program)->compilation_workers;
  QD_ASSERT(block);

  auto &offloads = block->statements;
  std::vector<std::unique_ptr<LLVMCompiledTask>> data(offloads.size());
  // DIAGNOSTIC (cse-diag): QD_CSE_STATS=1 logs the final per-task IR composition right before LLVM lowering, so
  // the size + KIND of work each GPU task carries can be compared across QD_CSE_MODE values. This is the per-task
  // kernel that actually runs every sim step, so redundant statements here == redundant work at runtime.
  static const bool cse_stats = []() {
    const char *e = std::getenv("QD_CSE_STATS");
    return e != nullptr && std::string(e) == "1";
  }();
  for (int i = 0; i < offloads.size(); i++) {
    if (cse_stats) {
      auto *task = offloads[i].get();
      auto cnt = [&](const std::function<bool(Stmt *)> &pred) {
        return (int)irpass::analysis::gather_statements(task, pred).size();
      };
      // Execution structure: task type + launch dims + trip count. With identical IR content, a parallel->serial
      // demotion or a shrunken grid would be the mechanism (same body statements, executed by far fewer threads).
      std::string ttype = "?";
      int grid = -1, block = -1, trip = -1;
      OffloadedStmt *off_task = task->is<OffloadedStmt>() ? task->as<OffloadedStmt>() : nullptr;
      if (off_task != nullptr) {
        ttype = OffloadedStmt::task_type_name(off_task->task_type);
        grid = off_task->grid_dim;
        block = off_task->block_dim;
        if (off_task->const_begin && off_task->const_end)
          trip = off_task->end_value - off_task->begin_value;
      }
      // Statements bucketed by INNER-loop nesting depth (loops below the offloaded task itself). Statements at
      // depth>=1 execute once PER inner-loop iteration PER thread, so redundant work there is the dynamic-cost
      // mechanism that a static total can't see: identical totals but more in_loop => more work at runtime.
      int in_loop = 0, deep = 0;
      for (auto *s : irpass::analysis::gather_statements(task, [](Stmt *) { return true; })) {
        int d = 0;
        for (Block *b = s->parent; b != nullptr;) {
          Stmt *ps = b->parent_stmt();
          if (ps == nullptr || ps == off_task)
            break;
          if (ps->is<RangeForStmt>() || ps->is<StructForStmt>() || ps->is<WhileStmt>())
            d++;
          b = ps->parent;
        }
        if (d >= 1)
          in_loop++;
        if (d >= 2)
          deep++;
      }
      std::cerr << "[CSE_STATS] mode=" << irpass::cse_mode() << " kernel=" << kernel->get_name() << " task=" << i
                << " type=" << ttype << " grid=" << grid << " block=" << block << " trip=" << trip
                << " in_loop=" << in_loop << " deep=" << deep
                << " total=" << irpass::analysis::count_statements(task)
                << " gload=" << cnt([](Stmt *s) { return s->is<GlobalLoadStmt>(); })
                << " gstore=" << cnt([](Stmt *s) { return s->is<GlobalStoreStmt>(); })
                << " gptr=" << cnt([](Stmt *s) { return s->is<GlobalPtrStmt>(); })
                << " binop=" << cnt([](Stmt *s) { return s->is<BinaryOpStmt>(); })
                << " unop=" << cnt([](Stmt *s) { return s->is<UnaryOpStmt>(); })
                << " const=" << cnt([](Stmt *s) { return s->is<ConstStmt>(); }) << std::endl;
    }
    auto compile_func = [&, i] {
      tlctx_.fetch_this_thread_struct_module();
      auto offload = irpass::analysis::clone(offloads[i].get());
      irpass::re_id(offload.get());

      Block blk;
      blk.insert(std::move(offload));
      // DIAGNOSTIC (cse-diag): QD_CSE_MODE=experiment reproduces exp2 (per-task CSE here, none in full_simplify).
      if (irpass::cse_mode() == "experiment") {
        irpass::whole_kernel_cse(&blk);
      }
      auto new_data = this->compile_task(i, compile_config_, nullptr, &blk);
      data[i] = std::make_unique<LLVMCompiledTask>(std::move(new_data));
    };
    worker.enqueue(compile_func);
  }
  worker.flush();

  auto llvm_compiled_kernel = tlctx_.link_compiled_tasks(std::move(data));
  optimize_module(llvm_compiled_kernel.module.get());
  return llvm_compiled_kernel;
}

#endif
}  // namespace quadrants::lang
