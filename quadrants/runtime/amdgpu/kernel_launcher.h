#pragma once

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/amdgpu/device_scratch_buffer.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace amdgpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  // Per-handle launcher state. Each registered kernel handle owns one
  // Context. Move-only because of the DeviceScratchBuffer members; the
  // contexts_ vector relies on move semantics during resize.
  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters{
        nullptr};
    std::vector<OffloadedTask> offloaded_tasks;

    // Cached device scratch buffers, lazily sized on first launch and
    // reused across subsequent launches.
    //
    // Why cache instead of calling hipMallocAsync/hipFreeAsync per launch?
    // Because on ROCm those calls have non-trivial per-call overhead even
    // when the memory pool is fully primed (mutex + stream-ordering
    // bookkeeping in the CLR layer that no pool tuning can eliminate).
    // See DeviceScratchBuffer's header comment for the full motivation.
    //
    // Stream affinity: default-constructed, i.e. bound to the default
    // (NULL) HIP stream, which is what the launcher uses today. When a
    // non-default stream is introduced (e.g. a per-handle stream set
    // during register_llvm_kernel), construct these with that stream --
    // either via the constructor argument or via set_stream() before the
    // first launch -- and update the memcpy_*_async / kernel-launch
    // sites in launch_llvm_kernel to use the same stream. The class
    // itself needs no further changes.
    //
    // RAII-managed: freed automatically when this Context (and thus the
    // launcher) is destroyed.
    DeviceScratchBuffer device_arg_buffer;
    DeviceScratchBuffer device_result_buffer;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  void launch_offloaded_tasks(
      LaunchContextBuilder &ctx,
      JITModule *amdgpu_module,
      const std::vector<OffloadedTask> &offloaded_tasks);
  void launch_offloaded_tasks_with_do_while(
      LaunchContextBuilder &ctx,
      JITModule *amdgpu_module,
      const std::vector<OffloadedTask> &offloaded_tasks);
  bool on_amdgpu_device(void *ptr);
  std::vector<Context> contexts_;
};

}  // namespace amdgpu
}  // namespace quadrants::lang

