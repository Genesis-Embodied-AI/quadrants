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
// We define our own copy because Quadrants loads the CUDA driver dynamically
// rather than linking against it, so we don't have access to those headers.
// Field order verified against cuda-python bindings (handle, type, size,
// phGraph_out, ctx). Introduced in CUDA 12.4; layout stable through 13.2+.
//
// Used to add the conditional while node via cuGraphAddNode. Normal kernel
// nodes have a dedicated cuGraphAddKernelNode API with CudaKernelNodeParams,
// but conditional nodes use the generic cuGraphAddNode which takes this
// catch-all 256-byte union. The type field selects the variant; we only use
// the conditional node variant, so most of the bytes are padding.
struct GraphNodeParams {
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
static_assert(sizeof(GraphNodeParams) == 256, "GraphNodeParams layout must match CUgraphNodeParams (256 bytes)");

// Everything try_launch needs to embed a child qd.kernel as a subgraph: the child's own (already-populated) launch
// context, plus the child's compiled artifacts looked up from the launcher's per-handle Context by child launch_id.
// Assembled in the CUDA KernelLauncher and passed into try_launch, indexed by OffloadedTask::child_call_index.
struct ChildLaunchInfo {
  LaunchContextBuilder *child_ctx{nullptr};
  JITModule *child_module{nullptr};
  const std::vector<std::pair<int, Callable::Parameter>> *child_parameters{nullptr};
  const std::vector<OffloadedTask> *child_tasks{nullptr};
};

struct CachedGraph {
  // Per-child persistent device state for an embedded child subgraph (the D1 data model: each child gets its own
  // arg buffer + RuntimeContext). Lives in `children` below, whose heap buffer is preserved across the CachedGraph
  // move into `cache_`, so `&persistent_ctx` stays stable for the lifetime of the instantiated graph_exec.
  struct ChildGraphState {
    char *persistent_device_arg_buffer{nullptr};
    RuntimeContext persistent_ctx{};
    std::size_t arg_buffer_size{0};
  };

  // CUgraphExec handle (typed as void* since driver API is loaded dynamically).
  // This is the instantiated, launchable form of the captured CUDA graph.
  void *graph_exec{nullptr};
  char *persistent_device_arg_buffer{nullptr};
  char *persistent_device_result_buffer{nullptr};
  RuntimeContext persistent_ctx{};
  std::vector<ChildGraphState> children;
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  // Device-side pointer slot for graph_do_while indirection. Holds the address
  // of the user's counter ndarray. The condition kernel reads through this
  // slot, allowing the counter ndarray to change between calls without
  // rebuilding.
  void *counter_ptr_slot{nullptr};
  std::size_t num_nodes{0};

  CachedGraph(std::size_t arg_buffer_size,
              std::size_t result_buffer_size,
              bool needs_counter_ptr_slot,
              LlvmRuntimeExecutor *executor);
  ~CachedGraph();
  CachedGraph(const CachedGraph &) = delete;
  CachedGraph &operator=(const CachedGraph &) = delete;
  CachedGraph(CachedGraph &&other) noexcept;
  CachedGraph &operator=(CachedGraph &&other) noexcept;
};

class GraphManager {
 public:
  // Attempts to launch the kernel via a cached or newly built CUDA graph.
  // Returns true on success; false if the graph path can't be used (e.g.
  // host-resident ndarrays) and the caller should fall back to normal launch.
  // Internally tracks whether the graph was used, queryable via
  // used_on_last_call().
  bool try_launch(int launch_id,
                  LaunchContextBuilder &ctx,
                  JITModule *cuda_module,
                  const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                  const std::vector<OffloadedTask> &offloaded_tasks,
                  LlvmRuntimeExecutor *executor,
                  const std::vector<ChildLaunchInfo> &child_infos);

  // cache_size and used_on_last_call used for tests
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
  bool launch_cached_graph(CachedGraph &cached,
                           LaunchContextBuilder &ctx,
                           bool use_graph_do_while,
                           const std::vector<ChildLaunchInfo> &child_infos,
                           LlvmRuntimeExecutor *executor);
  void resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                LlvmRuntimeExecutor *executor);
  // Refresh a child's persistent device arg buffer from its launch context (resolve ndarray handles -> device
  // pointers, then upload). Run on every launch (build and cached replay) so a child sees the parent call's current
  // ndarray data pointers, mirroring how the parent arg buffer is re-uploaded in launch_cached_graph.
  void refresh_child_arg_buffer(const ChildLaunchInfo &info,
                                CachedGraph::ChildGraphState &state,
                                LlvmRuntimeExecutor *executor);
  // Build a standalone CUgraph for an embedded child kernel (one kernel node per child task, chained), using the
  // child's persistent RuntimeContext. The returned graph is owned by the caller (embedded via add_child_graph_node
  // then destroyed after the parent graph is instantiated).
  void *build_child_subgraph(const ChildLaunchInfo &info, CachedGraph::ChildGraphState &state);
  void ensure_condition_kernel_loaded();
  void *add_conditional_while_node(void *graph, unsigned long long *cond_handle_out);
  void *add_kernel_node(void *graph,
                        void *prev_node,
                        void *func,
                        unsigned int grid_dim,
                        unsigned int block_dim,
                        unsigned int shared_mem,
                        void **kernel_params);
  // Embeds an already-built child graph as a single child-graph node, optionally chained after prev_node. This is
  // the C2 ("child-graph node") composition primitive used for nested qd.kernel-as-subgraph calls.
  void *add_child_graph_node(void *graph, void *prev_node, void *child_graph);

  // Keyed by launch_id, which uniquely identifies a compiled kernel variant
  // (each template specialization gets its own launch_id).
  std::unordered_map<int, CachedGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
  std::size_t total_builds_{0};

  // JIT-compiled condition kernel for graph_do_while conditional nodes
  void *cond_kernel_module_{nullptr};  // CUmodule
  void *cond_kernel_func_{nullptr};    // CUfunction
};

}  // namespace cuda
}  // namespace quadrants::lang
