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
// Two distinct call sites use this:
//   - Body kernels: 1 arg (RuntimeContext *device_runtime_ctx), 8 bytes packed. One instance shared by every body
//     kernel node in a CachedGraph (the device pointer is the same for every node).
//   - Yield-check kernels: 4 args (int32_t **yield_on_ptr_slot, int32_t cp_id, int32_t *yield_signal,
//     int32_t *resume_point), 32 bytes packed (C-struct alignment, 4-byte cp_id then 4 bytes padding before the
//     two 8-byte pointers). One instance PER yielding checkpoint, because the cp_id literal differs per checkpoint.
//
// The packed buffer is held by value (32-byte array) so its address is stable for the graph's lifetime, regardless
// of which arg shape is in use. `pack_size` records the actually-occupied prefix.
struct CachedKernelArgs {
  // C-struct-packed args bytes. 32 bytes covers both shapes above; unused tail bytes are ignored by HIP because
  // `pack_size` declares the actually-used prefix.
  unsigned char packed_args[32]{};
  std::size_t pack_size{sizeof(void *)};
  // {HIP_LAUNCH_PARAM_BUFFER_POINTER=0x01, &packed_args[0], HIP_LAUNCH_PARAM_BUFFER_SIZE=0x02, &pack_size,
  // HIP_LAUNCH_PARAM_END=0x03}. Held by value so its address is stable for the graph's lifetime.
  void *extra_config[5]{};
};

// Per-(launch_id) graph cache entry. Construction allocates the persistent device-side buffers that the cached graph
// reads through; destruction frees them and the instantiated `hipGraphExec_t`.
//
// One layout for all kernels: a single flat HIP graph holds every offloaded task. When the kernel carries no
// `qd.checkpoint(...)` blocks, the graph is just the task chain (the body kernels' codegen prologue is dead code
// because RuntimeContext::checkpoint_*_ptr is nullptr). When the kernel carries checkpoints, every cp_id >= 0 body
// kernel reads its own prologue and self-gates against `*resume_point_dev_ptr` / `*yield_signal_dev_ptr`, and the
// yield-check kernel (loaded once via `GraphManager::ensure_checkpoint_yield_check_kernel_loaded`) is inserted inline
// after each yielding checkpoint's last body kernel to maintain the yield-signal / resume-point state.
//
// All gating is on the GPU. The host writes resume_point once per launch (4-byte HtoD), reads yield_signal once after
// the graph completes (4-byte DtoH), and otherwise stays out of the way. Same shape as pre-Hopper CUDA's `CachedGraph`
// in `runtime/cuda/graph_manager.h`; HIP 7.2 lacks both conditional graph nodes and indirect dispatch, so the
// codegen-prologue + flat-graph pattern is the only mechanism that keeps gating on the GPU.
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
  // Host-side shadow of the `RuntimeContext` fields. The pointers it carries (`runtime`, `arg_buffer`, `result_buffer`,
  // `checkpoint_resume_point_ptr`, `checkpoint_yield_signal_ptr`) reference the persistent device buffers above. Filled
  // once in the constructor; copied into `device_runtime_ctx` exactly once at graph build time.
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
  // Per-yield-check-kernel-node arg packing. The yield-check kernel takes 4 args (yield_on_ptr_slot, cp_id,
  // yield_signal_ptr, resume_point_ptr); each yield-check node needs its own packed-args block because the cp_id is
  // per-node. One entry per yielding checkpoint inserted into the graph.
  std::vector<CachedKernelArgs> yield_check_kernel_args;
  // One cp_id literal per yielding checkpoint, held by value so its address stays stable for the graph's lifetime (the
  // yield-check kernel reads cp_id by pointer through HIP_LAUNCH_PARAM_BUFFER_POINTER, see CachedKernelArgs).
  std::vector<int32_t> yield_check_cp_id_storage;
  // One per-cp persistent slot holding the device address of the user's `yield_on=` ndarray. The slot's address is
  // baked into the graph; the pointer it contains is host-updated each launch via memcpy (mirrors CUDA's
  // `checkpoint_yield_on_ptr_slots`). Sized by max_cp_id + 1; non-yielding checkpoints have a nullptr slot.
  std::vector<void *> checkpoint_yield_on_ptr_slots;

  // Device-side int32 scalars read by the codegen prologue and the yield-check kernel. Allocated only when the kernel
  // has at least one `qd.checkpoint(...)`; nullptr otherwise. `persistent_ctx.checkpoint_*_ptr` carry their device
  // addresses so the body kernels can dereference them.
  void *resume_point_dev_ptr{nullptr};
  void *yield_signal_dev_ptr{nullptr};

  std::size_t arg_buffer_size{0};
  std::size_t result_buffer_size{0};
  std::size_t num_nodes{0};
  // Number of distinct cp_id >= 0 checkpoints this kernel carries. Reported via `cache_size` / for diagnostics; not
  // load-bearing for the graph build path.
  std::size_t num_checkpoints{0};

  CachedGraph(std::size_t arg_buffer_size,
              std::size_t result_buffer_size,
              bool needs_checkpoint_scalars,
              LlvmRuntimeExecutor *executor);
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
  // Slice 4: cp_id of the first checkpoint whose `yield_on=` flag was non-zero on the most recent launch, or `-1` if no
  // yield was observed (or the launch was not a checkpoint kernel). Mirrors the CUDA GraphManager surface
  // (`cuda::GraphManager::last_yield_cp_id_on_last_call`) so `Program::get_graph_last_yield_cp_id_on_last_call` can
  // call straight through.
  int last_yield_cp_id_on_last_call() const {
    return last_yield_cp_id_on_last_call_;
  }
  // Slice 4: writable setter so the streaming `KernelLauncher` (which handles `graph_do_while + checkpoint` kernels
  // that fall through the graph-path eligibility check above) can record yields into the same field the graph path
  // uses. The override in `kernel_launcher.h` reads it through `last_yield_cp_id_on_last_call()` either way, so the
  // Python `GraphStatus` surface stays uniform across both code paths.
  void set_last_yield_cp_id_on_last_call(int cp_id) {
    last_yield_cp_id_on_last_call_ = cp_id;
  }

  // Lazy-load + return the pre-built AMDGPU yield-check kernel function. Public so the streaming launcher
  // (`runtime/amdgpu/kernel_launcher.cpp`) can launch the same kernel inline for graph_do_while + checkpoint kernels
  // that fall through the graph-fast-path. Returns nullptr if the bundle doesn't cover the current arch.
  void *ensure_and_get_checkpoint_yield_check_kernel() {
    ensure_checkpoint_yield_check_kernel_loaded();
    return yield_check_kernel_func_;
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
  // Lazily loads the AMDGPU yield-check kernel from the pre-built bundled HSACO. Mirrors the CUDA equivalent
  // `cuda::GraphManager::ensure_checkpoint_yield_check_kernel_loaded`; called from `try_launch` when the kernel carries
  // at least one `qd.checkpoint(yield_on=...)`. Sets `yield_check_kernel_func_` to nullptr if the bundle can't be
  // loaded (unsupported arch); callers must treat nullptr as "yield-check unavailable".
  void ensure_checkpoint_yield_check_kernel_loaded();
  // Populates a CachedKernelArgs to pack 4 args (yield_on_ptr_slot **, cp_id literal *, yield_signal *, resume_point *)
  // in the layout the AMDGPU yield-check kernel expects via HIP_LAUNCH_PARAM_BUFFER_POINTER. The packed buffer is
  // stored inside the CachedKernelArgs (its `packed_runtime_ctx_ptr` slot is repurposed as a void * to an owned heap
  // allocation); extra_config[1] points at that heap slot.
  void initialize_yield_check_kernel_args(CachedKernelArgs &kernel_args,
                                          void *yield_on_ptr_slot_addr,
                                          int32_t *cp_id_storage,
                                          void *yield_signal_dev_ptr,
                                          void *resume_point_dev_ptr);

  // Keyed by `launch_id`, which uniquely identifies a compiled kernel variant (each template specialization gets its
  // own launch_id).
  std::unordered_map<int, CachedGraph> cache_;
  bool used_on_last_call_{false};
  std::size_t num_nodes_on_last_call_{0};
  std::size_t total_builds_{0};
  // Persistent across launches. Reset to `-1` at the start of every checkpoint-graph launch, then set by
  // `launch_cached_graph` when the post-graph DtoH of `yield_signal` returns a cp_id >= 0. Mirrors the CUDA
  // GraphManager field of the same name.
  int last_yield_cp_id_on_last_call_{-1};

  // Pre-built yield-check kernel state. Loaded once (lazily) per process; the loaded HIP module / function pointers are
  // shared across every cached graph. Same lifecycle as the CUDA `gate_kernel_*` / `yield_check_*` pair in
  // `cuda::GraphManager`.
  void *yield_check_kernel_module_{nullptr};
  void *yield_check_kernel_func_{nullptr};
};

}  // namespace amdgpu
}  // namespace quadrants::lang
