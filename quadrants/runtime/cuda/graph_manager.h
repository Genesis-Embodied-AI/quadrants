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

struct CachedGraph {
  // CUgraphExec handle (typed as void* since driver API is loaded dynamically).
  // This is the instantiated, launchable form of the captured CUDA graph.
  void *graph_exec{nullptr};
  char *persistent_device_arg_buffer{nullptr};
  char *persistent_device_result_buffer{nullptr};
  RuntimeContext persistent_ctx{};
  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  // Device-side pointer slots for graph_do_while indirection, one per nested level (indexed by level
  // id). Each holds the address of that level's condition ndarray; the condition kernel reads through
  // its slot, so the ndarray can change between launches without rebuilding the graph. Empty when the
  // kernel has no graph_do_while loop.
  std::vector<void *> counter_ptr_slots;
  // Persistent device int holding the constant 1, plus a slot pointing at it. Used to re-arm a nested
  // conditional handle at the start of each parent iteration: the condition kernel invoked with this
  // slot unconditionally sets the handle to 1 (cudaGraphCondAssignDefault only re-arms at top-level
  // launch, not per child-body re-execution -- see graph_nested_design.md R1). Only allocated for
  // kernels that have at least one nested graph_do_while level.
  void *const_one_dev{nullptr};
  void *const_one_slot{nullptr};
  std::size_t num_nodes{0};

  CachedGraph(std::size_t arg_buffer_size,
              std::size_t result_buffer_size,
              int num_graph_do_while_levels,
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
                  LlvmRuntimeExecutor *executor);

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
  bool launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx, bool use_graph_do_while);
  void resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                LlvmRuntimeExecutor *executor);
  void ensure_condition_kernel_loaded();
  // Create a conditional handle on `graph` with default launch value 1 (CU_GRAPH_COND_ASSIGN_DEFAULT).
  // Must be called before any re-arm init kernel that references the handle, so the handle value is
  // baked into that kernel's params.
  unsigned long long create_cond_handle(void *graph);
  // Create a conditional WHILE node in `graph` using the pre-created `handle`, depending on `prev_node`
  // if non-null. Returns the conditional node (for chaining siblings); outputs the body graph to fill.
  void *add_conditional_while_node(void *graph, void *prev_node, unsigned long long handle, void **body_graph_out);
  void *add_kernel_node(void *graph,
                        void *prev_node,
                        void *func,
                        unsigned int grid_dim,
                        unsigned int block_dim,
                        unsigned int shared_mem,
                        void **kernel_params);
  // Recursively build the nodes for graph_do_while level `parent_id` (-1 = kernel top level) over the
  // task range [begin, end) into `target_graph` (the body graph of `parent_id`, or the root graph for
  // -1). Direct tasks become kernel nodes; each contiguous run of a child level becomes a conditional
  // WHILE node (preceded by a re-arm init kernel when nested, i.e. parent_id != -1) whose body is
  // filled recursively. For a real loop level (parent_id >= 0) the level's condition kernel is appended
  // last. `cond_handles` is indexed by level id and filled as conditional nodes are created.
  void build_level(int parent_id,
                   void *target_graph,
                   int begin,
                   int end,
                   const std::vector<OffloadedTask> &tasks,
                   const std::vector<GraphDoWhileLevel> &levels,
                   std::vector<unsigned long long> &cond_handles,
                   JITModule *cuda_module,
                   CachedGraph &cached);

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
