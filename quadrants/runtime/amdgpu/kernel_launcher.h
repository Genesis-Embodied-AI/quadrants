#pragma once

#include <deque>

#include "quadrants/codegen/llvm/compiled_kernel_data.h"
#include "quadrants/runtime/amdgpu/graph_manager.h"
#include "quadrants/runtime/llvm/kernel_launcher.h"

namespace quadrants::lang {
namespace amdgpu {

class KernelLauncher : public LLVM::KernelLauncher {
  using Base = LLVM::KernelLauncher;

  struct Context {
    JITModule *jit_module{nullptr};
    const std::vector<std::pair<int, Callable::Parameter>> *parameters;
    std::vector<OffloadedTask> offloaded_tasks;
    // Per-kernel-handle persistent scratch. Allocated lazily on first use, grown amortised-doubling on demand,
    // freed only on launcher destruction. Stored per-handle (not per-launcher) because `publish_adstack_metadata`'s
    // host-eval branch evaluates `FieldLoad` SizeExpr leaves through `SNodeRwAccessorsBank`, which recursively
    // launches `snode_reader_*` kernels via the same launcher singleton; sharing one device-side `arg_buffer` /
    // `RuntimeContext` across kernels would let the recursive launch overwrite the parent's args before the parent
    // kernel reads them. Per-handle buffers give each kernel a private device address that recursive launches
    // cannot touch.
    void *arg_buffer_dev_ptr{nullptr};
    std::size_t arg_buffer_capacity{0};
    void *runtime_context_dev_ptr{nullptr};

    // GPU-side `qd.checkpoint` gating state, allocated lazily on the first launch of a checkpoint-bearing kernel that
    // lands on this Handle (i.e. graph_do_while + checkpoint kernels that fall through the graph fast path). Mirrors
    // the per-CachedGraph state in `GraphManager`: `checkpoint_resume_point_dev_ptr` and
    // `checkpoint_yield_signal_dev_ptr` hold device addresses of int32 scalars the codegen prologue and the yield-check
    // kernel dereference. `checkpoint_yield_on_slots[cp]` holds the device address of a slot that carries the user's
    // `yield_on=` ndarray address for cp `cp` (or nullptr if cp `cp` is non-yielding); slot contents are host-updated
    // each launch via memcpy. All three are freed once in the launcher destructor.
    void *checkpoint_resume_point_dev_ptr{nullptr};
    void *checkpoint_yield_signal_dev_ptr{nullptr};
    std::vector<void *> checkpoint_yield_on_slots;
    bool checkpoint_state_initialized{false};
  };

 public:
  using Base::Base;

  void launch_llvm_kernel(Handle handle, LaunchContextBuilder &ctx) override;
  Handle register_llvm_kernel(const LLVM::CompiledKernelData &compiled) override;
  std::size_t get_graph_cache_size() const override {
    return graph_manager_.cache_size();
  }
  bool get_graph_cache_used_on_last_call() const override {
    return graph_manager_.used_on_last_call();
  }
  std::size_t get_graph_num_nodes_on_last_call() const override {
    return graph_manager_.num_nodes_on_last_call();
  }
  std::size_t get_graph_total_builds() const override {
    return graph_manager_.total_builds();
  }
  // Slice 4: bridge to `GraphManager::last_yield_cp_id_on_last_call`. Without this override, the default in
  // `program/kernel_launcher.h` returns -1 unconditionally and the Python `GraphStatus` always reports `yielded=False`
  // on AMDGPU. Matches the equivalent override in `runtime/cuda/kernel_launcher.h`.
  int get_graph_last_yield_cp_id_on_last_call() const override {
    return graph_manager_.last_yield_cp_id_on_last_call();
  }

 private:
  void launch_offloaded_tasks(LaunchContextBuilder &ctx,
                              Context &launcher_ctx,
                              JITModule *amdgpu_module,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              void *context_pointer,
                              int arg_size);
  void launch_offloaded_tasks_with_do_while(LaunchContextBuilder &ctx,
                                            Context &launcher_ctx,
                                            JITModule *amdgpu_module,
                                            const std::vector<OffloadedTask> &offloaded_tasks,
                                            void *context_pointer,
                                            int arg_size);
  // GPU-side `qd.checkpoint` setup for the streaming path. Lazily allocates the per-handle resume_point / yield_signal
  // / yield_on slots, updates per-launch state (slot contents, resume_point HtoD, yield_signal init), and wires the
  // device pointers into `ctx.get_context()`. Must be called BEFORE the launch's `memcpy_host_to_device_async` of the
  // RuntimeContext, since that's what publishes `ctx.get_context().checkpoint_*_ptr` to the GPU. No-op if the kernel
  // carries no checkpoints.
  void prepare_streaming_checkpoint_state(LaunchContextBuilder &ctx,
                                          Context &launcher_ctx,
                                          const std::vector<OffloadedTask> &offloaded_tasks);
  // Launch the pre-built AMDGPU yield-check kernel directly via hipModuleLaunchKernel. Used by the streaming path to
  // keep `yield_signal` / `resume_point` consistent with the codegen prologue's view, mirroring how the graph fast path
  // inlines the yield-check kernel into the flat HIP graph.
  void launch_streaming_yield_check_kernel(Context &launcher_ctx, int32_t cp_id, void *stream);
  // Read the device-side `yield_signal` int32 and publish to the GraphManager surface. Mirrors the post-launch D2H the
  // graph fast path does (in `GraphManager::launch_cached_graph`). Returns the read value (`-1` if no yield this
  // launch).
  int32_t fetch_streaming_yield_signal(Context &launcher_ctx, void *stream);
  bool on_amdgpu_device(void *ptr);
  // `result_buffer` is the only scratch that stays launcher-global: kernels write to it then the host reads it
  // back via `hipMemcpyDtoH` before the next kernel runs, so recursive snode-reader launches that reuse it cannot
  // smuggle stale bytes into the parent's read (the parent's host-side `fetch_result_uint64` happens after every
  // child completes, before the parent kernel that would be the next reader). Grown amortised-doubling.
  void *persistent_result_buffer_dev_ptr_{nullptr};
  std::size_t persistent_result_buffer_capacity_{0};
  // std::deque (not std::vector): `publish_adstack_metadata`'s host-eval branch recursively registers snode-reader
  // kernels via this same launcher, calling `contexts_.resize()` while a parent `launch_llvm_kernel` frame still
  // holds a reference into the container.  std::deque never invalidates references on push_back / resize, so the
  // parent's `launcher_ctx` reference survives the child's registration.
  std::deque<Context> contexts_;
  GraphManager graph_manager_;

 public:
  ~KernelLauncher() override;
};

}  // namespace amdgpu
}  // namespace quadrants::lang
