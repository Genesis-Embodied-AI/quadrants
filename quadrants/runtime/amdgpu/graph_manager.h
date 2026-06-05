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
// reads through; destruction frees them and the instantiated `hipGraphExec_t`(s).
//
// Two layouts share one struct:
//   - Plain (no `qd.checkpoint` in the source kernel): `graph_exec` holds the single instantiated graph; the
//     `sub_*` fields below are empty. This is the path slice 4 leaves untouched.
//   - Checkpoint (slice 4): `graph_exec` is `nullptr`; `sub_graph_execs[i]` holds the i-th batch (a contiguous
//     run of offloaded tasks sharing the same `cp_id`), with `batch_cp_ids[i]` recording the cp_id (-1 for
//     unconditional batches that run outside any `qd.checkpoint`). The launcher iterates batches in order and
//     decides per batch whether to launch, mirroring the CPU slice 6 host-branch gating but with a HIP graph
//     launch instead of a per-task kernel launch.
struct CachedGraph {
  // hipGraphExec_t. The instantiated, launchable form of the captured HIP graph. Typed as void * since the driver is
  // loaded dynamically. Populated only on the "plain" layout (see struct comment); `nullptr` on the checkpoint
  // layout, where `sub_graph_execs` carries one exec per batch.
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

  // Slice 4 checkpoint layout. Empty on the plain layout.
  //
  // sub_graph_execs[i]: hipGraphExec_t for the i-th contiguous batch of same-cp_id offloaded tasks (in source
  //   declaration order). Released alongside `graph_exec` in the destructor.
  // batch_cp_ids[i]: cp_id this batch carries, or -1 if the batch is outside every `qd.checkpoint` (always runs).
  // Host-side reads `batch_cp_ids[i]` to decide whether to skip this sub-graph on the current launch.
  std::vector<void *> sub_graph_execs;
  std::vector<int32_t> batch_cp_ids;

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
  // Slice 4: cp_id of the first checkpoint whose `yield_on=` flag was non-zero on the most recent launch, or `-1`
  // if no yield was observed (or the launch was not a checkpoint kernel). Mirrors the CUDA GraphManager surface
  // (`cuda::GraphManager::last_yield_cp_id_on_last_call`) so `Program::get_graph_last_yield_cp_id_on_last_call`
  // can call straight through.
  int last_yield_cp_id_on_last_call() const {
    return last_yield_cp_id_on_last_call_;
  }

 private:
  bool launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx);
  // Slice 4: orchestrates the per-batch launch loop for checkpoint kernels. Mirrors the CPU launcher's
  // `launch_offloaded_tasks` host-branch gating, but with HIP sub-graph launches and a D2H of the yield flag
  // after each yielding checkpoint instead of single-task launches.
  bool launch_cached_checkpoint_graph(CachedGraph &cached, LaunchContextBuilder &ctx);
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
  // Slice 4: builds one HIP graph + hipGraphExec_t over a contiguous run of offloaded tasks. Used during the
  // sub-graph build path; returns the instantiated graph exec (graph object is destroyed before return).
  void *build_subgraph_exec_for_tasks(const std::vector<OffloadedTask> &tasks,
                                      std::size_t begin,
                                      std::size_t end,
                                      JITModule *amdgpu_module,
                                      CachedKernelArgs &kernel_args);

  // Keyed by `launch_id`, which uniquely identifies a compiled kernel variant (each template specialization gets its
  // own launch_id).
  std::unordered_map<int, CachedGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
  std::size_t total_builds_{0};
  // Slice 4: persistent across launches. Reset to `-1` at the start of every checkpoint-graph launch, then set by
  // `launch_cached_checkpoint_graph` when a yield_on=` flag reads non-zero. Mirrors the CUDA GraphManager field
  // of the same name.
  int last_yield_cp_id_on_last_call_{-1};
};

}  // namespace amdgpu
}  // namespace quadrants::lang
