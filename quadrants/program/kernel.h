#pragma once

#include "quadrants/util/lang_util.h"
#include "quadrants/ir/snode.h"
#include "quadrants/ir/ir.h"
#include "quadrants/rhi/arch.h"
#include "quadrants/program/callable.h"
#include "quadrants/program/ndarray.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

class Program;

class TI_DLL_EXPORT Kernel : public Callable {
 public:
  std::vector<SNode *> no_activate;

  bool is_accessor{false};

  Kernel(Program &program,
         const std::function<void()> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         const std::function<void(Kernel *)> &func,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  Kernel(Program &program,
         std::unique_ptr<IRNode> &&ir,
         const std::string &name = "",
         AutodiffMode autodiff_mode = AutodiffMode::kNone);

  bool ir_is_ast() const {
    return ir_is_ast_;
  }

  std::string to_string() const;

  LaunchContextBuilder make_launch_context();

  template <typename T>
  T fetch_ret(DataType dt, int i);

  [[nodiscard]] std::string get_name() const override;

  void set_kernel_key_for_cache(const std::string &kernel_key) const {
    kernel_key_ = kernel_key;
  }

  const std::string &get_cached_kernel_key() const {
    return kernel_key_;
  }

 private:
  void init(Program &program,
            const std::function<void()> &func,
            const std::string &name = "",
            AutodiffMode autodiff_mode = AutodiffMode::kNone);

  // True if |ir| is a frontend AST. False if it's already offloaded to CHI IR.
  bool ir_is_ast_{false};
  mutable std::string kernel_key_;
};

}  // namespace quadrants::lang
