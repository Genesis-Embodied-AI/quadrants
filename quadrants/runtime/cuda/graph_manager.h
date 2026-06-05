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
  // Device-side pointer slot for graph_do_while indirection. Holds the address
  // of the user's counter ndarray. The condition kernel reads through this
  // slot, allowing the counter ndarray to change between calls without
  // rebuilding.
  void *counter_ptr_slot{nullptr};
  // Framework-internal `resume_point` scalar (one int32 on device) read by every checkpoint
  // gate kernel at launch time. `nullptr` when the kernel has no `qd.checkpoint()` blocks.
  // Initialised to `0` so all checkpoints run on the first launch. Slice 1d's yield-check
  // kernel bumps this to INT_MAX when a checkpoint yields, so every later checkpoint's gate
  // sees `cp_id >= INT_MAX == false` and skips its body for the rest of the launch.
  // Slice 2's host-side `step.resume(from_checkpoint=cp)` will memcpy the resume cp_id into
  // this slot before relaunching the same cached graph. Lives for the lifetime of the
  // cached graph.
  void *resume_point_dev_ptr{nullptr};
  // Framework-internal `yield_signal` scalar (one int32 on device). `-1` (read as "no yield
  // this launch") on launch; the yield-check kernel atomically CASes the first yielding
  // checkpoint's cp_id into this slot. The cond-with-yield kernel reads this slot inside
  // `graph_do_while` bodies to exit the WHILE early on yield. After each launch the host
  // reads this back (synchronously via the launch's cudaStreamSynchronize) so the
  // `GraphStatus` host API in slice 2 can tell the user which checkpoint yielded.
  // `nullptr` when the kernel has no `qd.checkpoint(yield_on=...)` blocks.
  void *yield_signal_dev_ptr{nullptr};
  // Per-checkpoint indirection slots for the user's `yield_on=` ndarray pointer (one entry
  // per checkpoint, indexed by cp_id; `nullptr` for checkpoints without `yield_on=`). Same
  // indirection trick as `counter_ptr_slot`: the slot's device address is baked into the
  // graph (the yield-check kernel reads `*(int32_t**)slot`), but the pointer it holds is
  // re-memcpy'd from the host each launch to follow the current user ndarray. Lives for the
  // lifetime of the cached graph.
  std::vector<void *> checkpoint_yield_on_ptr_slots;
  // Per-checkpoint count (number of distinct cp_ids in this kernel's offloaded_tasks).
  // Stored here so test introspection can see whether the graph build actually emitted
  // IF nodes -- slice 1c covers the build-time wiring before the user-visible host API
  // arrives in slice 2. `0` for kernels without checkpoints.
  std::size_t num_checkpoints{0};
  std::size_t num_nodes{0};

  CachedGraph(std::size_t arg_buffer_size,
              std::size_t result_buffer_size,
              bool needs_counter_ptr_slot,
              bool needs_resume_point_slot,
              bool needs_yield_signal_slot,
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
  // Number of `qd.checkpoint(...)` blocks (== IF conditional nodes emitted) for the most
  // recent successful graph build / cached launch. `0` when the kernel has no checkpoints.
  // Used by tests to confirm the GraphManager actually wired IF nodes instead of silently
  // falling back to the flat top-level layout.
  std::size_t num_checkpoints_on_last_call() const {
    return num_checkpoints_on_last_call_;
  }
  // cp_id of the checkpoint that fired its `yield_on` flag on the most recent successful
  // graph launch, or `-1` if no checkpoint yielded. Read back from the device's
  // `yield_signal` scalar at the end of `launch_cached_graph` (a host sync precedes the
  // copy so the value is valid). Returns `-1` when the most recent launch wasn't a graph
  // launch or the kernel had no `yield_on` checkpoints.
  int last_yield_cp_id_on_last_call() const {
    return last_yield_cp_id_on_last_call_;
  }

 private:
  bool launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx, bool use_graph_do_while);
  void resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                LlvmRuntimeExecutor *executor);
  void ensure_condition_kernel_loaded();
  void ensure_cond_with_yield_kernel_loaded();
  void ensure_checkpoint_gate_kernel_loaded();
  void ensure_checkpoint_yield_check_kernel_loaded();
  void *add_conditional_while_node(void *graph, unsigned long long *cond_handle_out);
  void *add_kernel_node(void *graph,
                        void *prev_node,
                        void *func,
                        unsigned int grid_dim,
                        unsigned int block_dim,
                        unsigned int shared_mem,
                        void **kernel_params);

  // Keyed by launch_id, which uniquely identifies a compiled kernel variant
  // (each template specialization gets its own launch_id).
  std::unordered_map<int, CachedGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
  std::size_t num_checkpoints_on_last_call_{0};
  // -1 means "no checkpoint yielded on the most recent graph launch" (or the most recent
  // call didn't take the graph path). Slice 1d uses this for test-side introspection;
  // slice 2 will route it through `GraphStatus` on the Python API.
  int last_yield_cp_id_on_last_call_{-1};
  std::size_t total_builds_{0};

  // JIT-compiled condition kernel for graph_do_while conditional nodes
  void *cond_kernel_module_{nullptr};  // CUmodule
  void *cond_kernel_func_{nullptr};    // CUfunction
  // Variant of the graph_do_while condition kernel that also takes the framework's
  // `yield_signal` pointer and exits the WHILE early on yield. Lives in the same fatbin
  // (regenerated alongside `_qd_graph_do_while_cond` by `build_condition_kernel_fatbin.py`)
  // so we don't need a second module load. Only loaded lazily when the kernel actually has
  // `qd.checkpoint(yield_on=...)` inside a `qd.graph_do_while` body.
  void *cond_with_yield_kernel_func_{nullptr};  // CUfunction
  // JIT-compiled gate kernel for qd.checkpoint() IF conditional nodes. Loaded lazily from the
  // pre-built `checkpoint_gate_fatbin.h`. One shared kernel handles every checkpoint -- the
  // per-checkpoint `cp_id` is passed as a literal argument (slice 1c). Pre-CUDA-12.4 / non-CUDA
  // backends will use a separate per-checkpoint specialised gate kernel for indirect dispatch
  // in slice 4/5.
  void *gate_kernel_module_{nullptr};  // CUmodule
  void *gate_kernel_func_{nullptr};    // CUfunction
  // JIT-compiled yield-check kernel for `qd.checkpoint(yield_on=...)`. Loaded lazily from the
  // pre-built `checkpoint_yield_check_fatbin.h`. Inserted at the end of each IF body whose
  // checkpoint declared a `yield_on=` parameter. One shared kernel; cp_id is passed by value.
  void *yield_check_kernel_module_{nullptr};  // CUmodule
  void *yield_check_kernel_func_{nullptr};    // CUfunction
};

}  // namespace cuda
}  // namespace quadrants::lang
