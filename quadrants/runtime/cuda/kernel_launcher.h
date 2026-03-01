#pragma once

#include <string>
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

// Mirrors CUgraphNodeParams layout for conditional while nodes.
// See CUDA driver API: CUgraphNodeParams / CUDA_CONDITIONAL_NODE_PARAMS.
struct CudaGraphNodeParams {
  unsigned int type;  // CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
  int reserved0[3];
  // Union starts at offset 16 (232 bytes total)
  unsigned long long handle;   // CUgraphConditionalHandle
  unsigned int condType;       // CU_GRAPH_COND_TYPE_WHILE = 1
  unsigned int size;           // 1 for while
  void *phGraph_out;           // CUgraph* output array
  void *ctx;                   // CUcontext
  char _pad[232 - 8 - 4 - 4 - 8 - 8];
  long long reserved2;
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
  bool resolve_ctx_ndarray_ptrs(
      LaunchContextBuilder &ctx,
      const std::vector<std::pair<int, Callable::Parameter>> &parameters);
  bool launch_llvm_kernel_graph(Handle handle, LaunchContextBuilder &ctx);
  void ensure_condition_kernel_loaded();
  std::vector<Context> contexts_;
  std::unordered_map<int, CachedCudaGraph> cuda_graph_cache_;

  // JIT-compiled condition kernel for graph_while conditional nodes
  void *cond_kernel_module_{nullptr};   // CUmodule
  void *cond_kernel_func_{nullptr};     // CUfunction
};

}  // namespace cuda
}  // namespace quadrants::lang
