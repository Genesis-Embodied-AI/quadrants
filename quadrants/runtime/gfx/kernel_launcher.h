#pragma once

#include "quadrants/program/kernel_launcher.h"
#include "quadrants/runtime/gfx/runtime.h"

namespace quadrants::lang {
namespace gfx {

class KernelLauncher : public lang::KernelLauncher {
 public:
  struct Config {
    GfxRuntime *gfx_runtime_{nullptr};
  };

  explicit KernelLauncher(Config config);

  void launch_kernel(const lang::CompiledKernelData &compiled_kernel_data, LaunchContextBuilder &ctx) override;

  // Slice 4 (Vulkan / Metal): route through to `GfxRuntime::last_yield_cp_id_on_last_call()`. Without
  // this override the default in `program/kernel_launcher.h` returns -1 unconditionally and Python's
  // `GraphStatus.yielded` always reports False on GFX backends. Matches the AMDGPU / CUDA overrides.
  int get_graph_last_yield_cp_id_on_last_call() const override {
    return config_.gfx_runtime_->last_yield_cp_id_on_last_call();
  }

 private:
  void launch_offloaded_tasks_with_do_while(Handle handle, LaunchContextBuilder &ctx);
  Handle register_kernel(const lang::CompiledKernelData &compiled_kernel_data);

  Config config_;
};

}  // namespace gfx
}  // namespace quadrants::lang
