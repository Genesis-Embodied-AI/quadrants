#pragma once

#include <unordered_map>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace cuda {

struct CudaKernelNodeParams {
  void *func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
};

struct CachedCudaGraph {
  void *graph_exec{nullptr};
  char *persistent_device_arg_buffer{nullptr};
  char *persistent_device_result_buffer{nullptr};
  RuntimeContext persistent_ctx{};
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};

  CachedCudaGraph() = default;
  ~CachedCudaGraph();
  CachedCudaGraph(const CachedCudaGraph &) = delete;
  CachedCudaGraph &operator=(const CachedCudaGraph &) = delete;
  CachedCudaGraph(CachedCudaGraph &&other) noexcept;
  CachedCudaGraph &operator=(CachedCudaGraph &&other) noexcept;
};

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(
      const LLVM::CompiledKernelData &compiled) override;

 private:
  bool on_cuda_device(void *ptr);
  bool launch_llvm_kernel_graph(Handle handle, LaunchContextBuilder &ctx);
  std::vector<Context> contexts_;
  std::unordered_map<int, CachedCudaGraph> cuda_graph_cache_;
  bool use_cuda_graph_{false};
  bool use_cuda_graph_checked_{false};
};

}  // namespace cuda
}  // namespace quadrants::lang
