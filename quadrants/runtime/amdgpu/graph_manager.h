#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/llvm/llvm_runtime_executor.h"

namespace quadrants::lang {
namespace amdgpu {

// Mirrors `hipKernelNodeParams` from /opt/rocm/include/hip/hip_runtime_api.h. We define our own copy because Quadrants
// loads the HIP runtime dynamically rather than linking against it, so we don't pull in the HIP headers here. Field
// order matters: HIP's struct is NOT the same shape as CUDA's `CUDA_KERNEL_NODE_PARAMS` (see
// `runtime/cuda/graph_manager.h`).
struct HipKernelNodeParams {
  // HIP's `dim3` is `{uint32 x, uint32 y, uint32 z}`. We flatten to three explicit fields rather than depending on the
  // HIP type to keep this file header-only.
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  void **extra;
  void *func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  void **kernelParams;
  unsigned int sharedMemBytes;
};

// Kernel node arg-packing wrapper. HIP's hipModuleLaunchKernel on AMD requires args via the `extra` byte-buffer
// convention (see `rhi/amdgpu/amdgpu_context.cpp::AMDGPUContext::launch`), not via the per-arg `kernelParams` pointer
// array. Graph kernel nodes follow the same calling convention, so we mirror the non-graph launcher's setup: a packed
// byte buffer + a 5-element extra config with HIP_LAUNCH_PARAM_* markers.
//
// One instance per graph (not per node): all kernel nodes in a cached graph share a single `device_runtime_ctx` pointer
// arg, and the size never changes.
struct CachedKernelArgs {
  // Byte-packed copy of the single kernel arg: the device-side RuntimeContext pointer. Sized for `sizeof(void *)`.
  void *packed_runtime_ctx_ptr{nullptr};
  std::size_t pack_size{sizeof(void *)};
  // {HIP_LAUNCH_PARAM_BUFFER_POINTER=0x01, &packed_runtime_ctx_ptr, HIP_LAUNCH_PARAM_BUFFER_SIZE=0x02, &pack_size,
  // HIP_LAUNCH_PARAM_END=0x03}. Held by value so its address is stable for the graph's lifetime.
  void *extra_config[5]{};
};

// Per-(launch_id) graph cache entry. Construction allocates the persistent device-side buffers that the cached graph
// reads through; destruction frees them and the instantiated `hipGraphExec_t`.
struct CachedGraph {
  // hipGraphExec_t. The instantiated, launchable form of the captured HIP graph. Typed as void * since the driver is
  // loaded dynamically.
  void *graph_exec{nullptr};
  // Persistent device buffer that the host arg buffer is copied into before every graph launch. The graph kernel nodes
  // read from this address (baked in via the persistent `RuntimeContext`'s `arg_buffer` field below).
  char *persistent_device_arg_buffer{nullptr};
  // Persistent device buffer for struct return values. Unused at the moment (graph mode rejects kernels with struct
  // returns up-front) but kept for parity with the CUDA implementation and future expansion.
  char *persistent_device_result_buffer{nullptr};
  // Host-side shadow of the `RuntimeContext` fields. The pointers it carries (`runtime`, `arg_buffer`,
  // `result_buffer`) reference the persistent device buffers above. Filled once in the constructor; copied into
  // `device_runtime_ctx` exactly once at graph build time.
  RuntimeContext persistent_ctx{};
  // Device-resident copy of `persistent_ctx`. The graph's kernel-node args bake the *value* of this pointer; the
  // kernels dereference it on the GPU to reach the persistent arg / result buffers. Unlike the CUDA path which can pass
  // a host pointer and rely on UVA / HMM, the AMDGPU runtime device-stages the `RuntimeContext` (see
  // `runtime/amdgpu/kernel_launcher.cpp` for the per-launch equivalent).
  void *device_runtime_ctx{nullptr};
  // Per-graph kernel-arg packing (see CachedKernelArgs). Held by value so the `extra_config` array's address is stable
  // for the cached graph's lifetime. All kernel nodes in this graph share the same packed args (the device
  // RuntimeContext pointer).
  CachedKernelArgs kernel_args;
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  std::size_t num_nodes{0};

  CachedGraph(std::size_t arg_buffer_size, std::size_t result_buffer_size, LlvmRuntimeExecutor *executor);
  ~CachedGraph();
  CachedGraph(const CachedGraph &) = delete;
  CachedGraph &operator=(const CachedGraph &) = delete;
  CachedGraph(CachedGraph &&other) noexcept;
  CachedGraph &operator=(CachedGraph &&other) noexcept;
};

class GraphManager {
 public:
  // Attempts to launch the kernel via a cached or newly built HIP graph. Returns true on success; false if the graph
  // path can't be used (e.g. host-resident ndarrays) and the caller should fall back to the regular streaming launch.
  // Tracks whether the graph was used, queryable via `used_on_last_call()`.
  bool try_launch(int launch_id,
                  LaunchContextBuilder &ctx,
                  JITModule *amdgpu_module,
                  const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                  const std::vector<OffloadedTask> &offloaded_tasks,
                  LlvmRuntimeExecutor *executor);

  // Reset the per-call flags. Called by the launcher when the graph path is skipped, so the test-facing accessors
  // below report the absence of a cache hit on the most recent launch.
  void mark_not_used() {
    used_on_last_call_ = false;
    num_nodes_on_last_call_ = 0;
  }
  std::size_t cache_size() const {
    return cache_.size();
  }
  bool used_on_last_call() const {
    return used_on_last_call_;
  }
  std::size_t num_nodes_on_last_call() const {
    return num_nodes_on_last_call_;
  }
  std::size_t total_builds() const {
    return total_builds_;
  }

 private:
  bool launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx);
  void resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                LlvmRuntimeExecutor *executor);
  void *add_kernel_node(void *graph,
                        void *prev_node,
                        void *func,
                        unsigned int grid_dim,
                        unsigned int block_dim,
                        unsigned int shared_mem,
                        CachedKernelArgs &kernel_args);

  // Keyed by `launch_id`, which uniquely identifies a compiled kernel variant (each template specialization gets its
  // own launch_id).
  std::unordered_map<int, CachedGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
  std::size_t total_builds_{0};
};

}  // namespace amdgpu
}  // namespace quadrants::lang
