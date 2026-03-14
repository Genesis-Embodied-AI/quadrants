#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

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

// Mirrors CUDA driver API CUgraphNodeParams / CUDA_CONDITIONAL_NODE_PARAMS.
// Field order verified against cuda-python bindings (handle, type, size,
// phGraph_out, ctx). Introduced in CUDA 12.4; layout stable through 13.2+.
struct CudaGraphNodeParams {
  unsigned int type;  // CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
  int reserved0[3];
  // Union starts at offset 16 (232 bytes total)
  unsigned long long handle;  // CUgraphConditionalHandle
  unsigned int condType;      // CU_GRAPH_COND_TYPE_WHILE = 1
  unsigned int size;          // 1 for while
  void *phGraph_out;          // CUgraph* output array
  void *ctx;                  // CUcontext
  char _pad[232 - 8 - 4 - 4 - 8 - 8];
  long long reserved2;
};
static_assert(
    sizeof(CudaGraphNodeParams) == 256,
    "CudaGraphNodeParams layout must match CUgraphNodeParams (256 bytes)");

struct CachedCudaGraph {
  // CUgraphExec handle (typed as void* since driver API is loaded dynamically).
  // This is the instantiated, launchable form of the captured CUDA graph.
  void *graph_exec{nullptr};
  char *persistent_device_arg_buffer{nullptr};
  char *persistent_device_result_buffer{nullptr};
  RuntimeContext persistent_ctx{};
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  void *graph_do_while_flag_dev_ptr{nullptr};

  CachedCudaGraph() = default;
  ~CachedCudaGraph();
  CachedCudaGraph(const CachedCudaGraph &) = delete;
  CachedCudaGraph &operator=(const CachedCudaGraph &) = delete;
  CachedCudaGraph(CachedCudaGraph &&other) noexcept;
  CachedCudaGraph &operator=(CachedCudaGraph &&other) noexcept;
};

class CudaGraphManager {
 public:
  // Attempts to launch the kernel via a cached or newly built CUDA graph.
  // Returns true on success; false if the graph path can't be used (e.g.
  // host-resident ndarrays) and the caller should fall back to normal launch.
  // Internally tracks whether the graph was used, queryable via used_on_last_call().
  bool try_launch(int launch_id, LaunchContextBuilder &ctx,
                  JITModule *cuda_module,
                  const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                  const std::vector<OffloadedTask> &offloaded_tasks,
                  LlvmRuntimeExecutor *executor);

  void mark_not_used() { used_on_last_call_ = false; }
  std::size_t cache_size() const { return cache_.size(); }
  bool used_on_last_call() const { return used_on_last_call_; }

 private:
  bool on_cuda_device(void *ptr);
  bool resolve_ctx_ndarray_ptrs(
      LaunchContextBuilder &ctx,
      const std::vector<std::pair<int, Callable::Parameter>> &parameters,
      LlvmRuntimeExecutor *executor);
  void ensure_condition_kernel_loaded();
  void *add_conditional_while_node(void *graph,
                                   unsigned long long *cond_handle_out);
  void *add_kernel_node(void *graph, void *prev_node, void *func,
                        unsigned int grid_dim, unsigned int block_dim,
                        unsigned int shared_mem, void **kernel_params);

  // Keyed by launch_id, which uniquely identifies a compiled kernel variant
  // (each template specialization gets its own launch_id).
  std::unordered_map<int, CachedCudaGraph> cache_;
  bool used_on_last_call_{false};

  // JIT-compiled condition kernel for graph_do_while conditional nodes
  void *cond_kernel_module_{nullptr};  // CUmodule
  void *cond_kernel_func_{nullptr};    // CUfunction
};

}  // namespace cuda
}  // namespace quadrants::lang
