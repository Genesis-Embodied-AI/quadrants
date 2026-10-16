#pragma once
#include "quadrants/util/lang_util.h"

#include <unordered_map>
#include <vector>
#include <chrono>

#include "quadrants/rhi/device.h"
#include "quadrants/codegen/spirv/snode_struct_compiler.h"
#include "quadrants/codegen/spirv/kernel_utils.h"
#include "quadrants/program/compile_config.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/program/snode_expr_utils.h"
#include "quadrants/program/program_impl.h"
#include "quadrants/program/kernel_launcher.h"

namespace quadrants::lang {
namespace gfx {

using namespace quadrants::lang::spirv;

using BufferType = TaskAttributes::BufferType;
using BufferInfo = TaskAttributes::BufferInfo;
using BufferBind = TaskAttributes::BufferBind;
using BufferInfoHasher = TaskAttributes::BufferInfoHasher;

using high_res_clock = std::chrono::high_resolution_clock;

// TODO: In the future this isn't necessarily a pointer, since DeviceAllocation
// is already a pretty cheap handle>
using InputBuffersMap = std::unordered_map<BufferInfo, DeviceAllocation *, BufferInfoHasher>;

class SNodeTreeManager;

class CompiledQuadrantsKernel {
 public:
  struct Params {
    const QuadrantsKernelAttributes *qd_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::size_t num_snode_trees{0};

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
    DeviceAllocation *listgen_buffer{nullptr};

    PipelineCache *backend_cache{nullptr};
  };

  explicit CompiledQuadrantsKernel(const Params &qd_params);

  const QuadrantsKernelAttributes &qd_kernel_attribs() const;

  size_t num_pipelines() const;

  size_t get_args_buffer_size() const;
  size_t get_ret_buffer_size() const;

  Pipeline *get_pipeline(int i);

  DeviceAllocation *get_buffer_bind(const BufferInfo &bind) {
    return input_buffers_[bind];
  }

 private:
  QuadrantsKernelAttributes qd_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  [[maybe_unused]] Device *device_;

  InputBuffersMap input_buffers_;

  size_t args_buffer_size_{0};
  size_t ret_buffer_size_{0};
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

struct PerTaskAdStackRuntime {
  std::vector<uint32_t> metadata;
  uint32_t stride_float{0};
  uint32_t stride_int{0};
};

class QD_DLL_EXPORT GfxRuntime {
 public:
  struct Params {
    Device *device{nullptr};
    KernelProfilerBase *profiler{nullptr};
    ProgramImpl *program_impl{nullptr};
  };

  explicit GfxRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~GfxRuntime();

  using KernelHandle = KernelLauncher::Handle;

  struct RegisterParams {
    QuadrantsKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
    std::size_t num_snode_trees{0};
  };

  KernelHandle register_quadrants_kernel(RegisterParams params);

  // Records (a contiguous range of) the kernel's offloaded tasks into the current command list. `task_begin` /
  // `task_end` select the half-open task range `[task_begin, task_end)`; the default `task_end == -1` means "all
  // tasks", i.e. the normal whole-kernel launch. The range form is used by the GFX host-side graph_do_while driver to
  // replay individual loop-body task sub-ranges (see `runtime/gfx/kernel_launcher.cpp`). The caller is responsible for
  // `synchronize()`ing before reading back any device-side loop condition.
  void launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx, int task_begin = 0, int task_end = -1);

  // Number of offloaded tasks in a registered kernel. Used by the graph_do_while driver to size its per-task level
  // table.
  int get_num_tasks(KernelHandle handle) const;

  // Innermost graph_do_while loop level for offloaded task `task_idx` (-1 = top-level / runs once). See
  // `spirv::TaskAttributes::graph_do_while_level_id`.
  int get_task_graph_do_while_level_id(KernelHandle handle, int task_idx) const;

  // Slice 4 (Vulkan / Metal): cp_id of the first checkpoint whose `yield_on=` flag was non-zero on the most recent
  // `launch_kernel` call, or `-1` if no yield was observed / the launched kernel has no yielding checkpoints. Mirrors
  // the AMDGPU `GraphManager::last_yield_cp_id_on_last_call()` / CUDA-side surface so
  // `Program::get_graph_last_yield_cp_id_on_last_call` can route through the GFX launcher uniformly.
  int last_yield_cp_id_on_last_call() const {
    return last_yield_cp_id_on_last_call_;
  }

  void buffer_copy(DevicePtr dst, DevicePtr src, size_t size);

  void synchronize();

  StreamSemaphore flush();

  Device *get_ti_device() const;

  void add_root_buffer(size_t root_buffer_size);

  DeviceAllocation *get_root_buffer(int id) const;

  size_t get_root_buffer_size(int id) const;

  void enqueue_compute_op_lambda(std::function<void(Device *device, CommandList *cmdlist)> op,
                                 const std::vector<ComputeOpImageRef> &image_refs);

  bool used_in_kernel(DeviceAllocationId id) {
    return ndarrays_in_use_.count(id) > 0;
  }

  static std::pair<const lang::StructType *, size_t> get_struct_type_with_data_layout(const lang::StructType *old_ty,
                                                                                      const std::string &layout);

  static std::tuple<const lang::StructType *, size_t, size_t> get_struct_type_with_data_layout_impl(
      const lang::StructType *old_ty,
      const std::string &layout);

 private:
  friend class quadrants::lang::gfx::SNodeTreeManager;

  void ensure_current_cmdlist();
  void submit_current_cmdlist_if_timeout();

  std::vector<PerTaskAdStackRuntime> publish_adstack_metadata_spirv(
      LaunchContextBuilder &host_ctx,
      const std::vector<spirv::TaskAttributes> &task_attribs);

  void init_nonroot_buffers();

  Device *device_{nullptr};
  KernelProfilerBase *profiler_;

  std::unique_ptr<PipelineCache> backend_cache_{nullptr};

  std::vector<std::unique_ptr<DeviceAllocationGuard>> root_buffers_;
  std::unique_ptr<DeviceAllocationGuard> global_tmps_buffer_;
  // FIXME: Support proper multiple lists
  std::unique_ptr<DeviceAllocationGuard> listgen_buffer_;

  // Deferred-free buffers associated with queued-but-not-yet-completed cmdlists. `flush()` leaves this
  // untouched after submit (any buffer here may still be referenced by an in-flight command); only
  // `synchronize()` clears it, after `wait_idle()` drains the stream. Across repeated `flush()` calls without
  // an intervening sync this can accumulate in principle - one batch per flush - but every workload in
  // Quadrants touches a Python-side observable (result fetch, `to_numpy()`, field readback, etc.) between
  // kernel launches and those paths trigger an implicit `synchronize()` that drains the queue.
  //
  // A bounded-FIFO / semaphore-keyed retirement scheme was considered (see the `is_signaled()` discussion in
  // the closed PR #538) and rejected as net-negative: the FIFO variant trades a theoretical growth path for
  // a real, measurable blocking stall every N flushes, at an arbitrary threshold. The non-blocking polling
  // variant is the one worth doing, but only if a real workload motivates it - that requires
  // `bool is_signaled() const` on `StreamSemaphoreObject` and per-backend implementations (`vkGetFenceStatus`
  // on Vulkan, `MTLSharedEvent` on Metal, trivial on CPU), an RHI public-surface change that should stand
  // alone when it lands.
  std::vector<std::unique_ptr<DeviceAllocationGuard>> ctx_buffers_;

  // Per-dispatch heaps for SPIR-V adstack primal/adjoint storage. The float heap backs f32-valued adstacks; the int
  // heap backs i32 and u1 adstacks (u1 stored as i32 to match the Function-scope path's bool->int remap). Other
  // primitive types (f64, i64, ...) are hard-errored in the shader codegen (no fallback). Each heap is sized at `stride
  // * (group_x * block_dim) * sizeof(element)` and grown lazily; reused across launches whenever the current allocation
  // is already big enough. On grow, the previous buffer is moved into `ctx_buffers_` rather than freed synchronously,
  // so any in-flight cmdlist still referencing it stays valid until the stream drains.
  std::unique_ptr<DeviceAllocationGuard> adstack_heap_buffer_float_;
  size_t adstack_heap_buffer_float_size_{0};
  std::unique_ptr<DeviceAllocationGuard> adstack_heap_buffer_int_;
  size_t adstack_heap_buffer_int_size_{0};
  // Owning `ProgramImpl` back-reference; propagated from `Params::program_impl`. See the comment on
  // `Params::program_impl` for the contract.
  ProgramImpl *program_impl_{nullptr};

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};
  high_res_clock::time_point current_cmdlist_pending_since_;

  // Counts kernel launches since the last `synchronize()`. `submit_current_cmdlist_if_timeout` forces a drain once this
  // crosses a threshold, bounding the growth of `VulkanStream::submitted_cmdbuffers_` (and the fences, semaphores and
  // descriptor sets those entries keep alive) on tight kernel-launch loops that never touch a Python-side observable
  // -workloads like MPM88 where every substep is a pure GPU update and the host only reads state once at the end. See
  // the assignment site for the MoltenVK SIGSEGV this guards against.
  size_t pending_launches_since_sync_{0};

  std::vector<std::unique_ptr<CompiledQuadrantsKernel>> qd_kernels_;

  std::unordered_map<DeviceAllocation *, size_t> root_buffers_size_map_;
  std::unordered_map<DeviceAllocationId, ImageLayout> last_image_layouts_;
  // [Note] Why do we need to track ndarrays that are in use?
  // Since we separate cmdlist is async, quadrants needs a way to know whether
  // ndarrays are still used by pending kernels to be executed. So we use
  // ndarray_in_use_ to track this so that we can free memory allocated for
  // ndarray whenever it's safe to do so.
  std::unordered_set<DeviceAllocationId> ndarrays_in_use_;

  // Slice 4: see the public `last_yield_cp_id_on_last_call()` accessor. Reset to -1 at the start of every
  // `launch_kernel` call that has any `checkpoint_yield_on_arg_ids[cp] >= 0`; updated when the post-checkpoint readback
  // observes a non-zero `yield_on=` flag.
  int last_yield_cp_id_on_last_call_{-1};

  // GPU-side checkpoint gating (Vulkan / Metal). Lazily-built generic gate + yield-check pipelines (one per
  // `GfxRuntime`, shared across every yielding-capable kernel). The gate shader is
  // `quadrants/codegen/spirv/checkpoint_gate_shader.{h,cpp}`; the yield-check shader is
  // `quadrants/codegen/spirv/checkpoint_yield_check_shader.{h,cpp}`. Both are vanilla compute shaders (no capability
  // requirements beyond `OpAtomicCompareExchange`, universal across the Quadrants-supported Vulkan / Metal feature
  // set). See `runtime/gfx/checkpoint_launch.cpp` for the orchestration that uses them.
  std::unique_ptr<Pipeline> checkpoint_gate_pipeline_{nullptr};
  std::unique_ptr<Pipeline> checkpoint_yield_check_pipeline_{nullptr};

  // Per-kernel-handle cached gating state. Indexed by `KernelHandle::get_launch_id()` (same indexing as `qd_kernels_`),
  // allocated lazily on the first checkpoint-bearing launch of each handle and reused across every subsequent launch of
  // that handle. Entries for non-checkpoint kernels (or pre-allocation slots for not-yet-launched handles) are left
  // default-constructed (`per_cp.empty()`); the launcher tests `state.per_cp.empty()` as the fast-path guard.
  //
  // Lifecycle: never freed during a `GfxRuntime`'s lifetime - the device allocations live in the
  // `DeviceAllocationGuard`s here, which match the lifetime of the `GfxRuntime` (the kernel handles themselves never
  // get reissued, so the per-handle entry stays valid as long as the handle is callable). `qd.reset()` tears the whole
  // runtime down and reclaims everything.
  struct CheckpointPerCpState {
    // cp_id this entry describes. Matches the index into `CheckpointHandleState::per_cp` for every yielding-capable
    // handle (per-cp entries are dense, indexed 0..max_cp_id).
    int32_t cp_id{-1};
    // Per-checkpoint params SSBO. Layout matches `spirv::CheckpointGateParams` followed by the active-dim u32 triples
    // (one per body kernel in the checkpoint, plus one extra triple for the yield-check shader's grid dim when this
    // checkpoint has `yield_on=`). Written once at first launch; the gate shader reads from it and the contents never
    // change across launches.
    std::unique_ptr<DeviceAllocationGuard> gate_params;
    // Per-checkpoint out-dims SSBO. Holds N+(yielding?1:0) u32 triples; written by the gate shader each launch with
    // either the active dim or `(0, 0, 0)`; consumed by each body kernel's (and the yield-check shader's)
    // `CommandList::dispatch_indirect` at offset `12 * slot_idx`. Allocated with `AllocUsage::Storage |
    // AllocUsage::Indirect` so the same buffer can be both written as an SSBO and read as an indirect-dispatch source.
    std::unique_ptr<DeviceAllocationGuard> out_dims;
    // Per-yielding-checkpoint yield-check params SSBO (4 bytes, holds cp_id). Allocated only when `yield_on=` was
    // supplied for this checkpoint; null otherwise.
    std::unique_ptr<DeviceAllocationGuard> yield_check_params;
    // Body-kernel task indices that belong to this checkpoint, in original task order. The launcher walks these to bind
    // + dispatch_indirect each body kernel; offset `12 * i` in `out_dims` gives the i-th body kernel's dim3 slot. The
    // yield-check shader, when present, reads from the trailing slot at offset `12 * body_tasks.size()`.
    std::vector<int> body_task_indices;
  };

  struct CheckpointHandleState {
    // Per-launch control buffer: `[resume_point: i32, yield_signal: i32]`. Allocated with `host_write=true,
    // host_read=true` so the launcher can `upload_data` the initial `(resume_point, -1)` at launch start and
    // `readback_data` the final `yield_signal` at launch end without going through a separate staging buffer. Shared
    // across all per-cp states; the gate and yield-check shaders both read / write it.
    std::unique_ptr<DeviceAllocationGuard> control;
    // Dense per-cp entries: `per_cp[i]` describes checkpoint id `i`. Sparse cp_ids (a kernel with checkpoints 0 and 2
    // but no 1) are not produced by the AST transformer (cp_ids are dense by construction in
    // `quadrants/python/quadrants/lang/transformer.py`), so the dense layout is correct; `per_cp[i].cp_id == i` always
    // when the entry is populated.
    std::vector<CheckpointPerCpState> per_cp;
  };

  // Per-handle cached gating state. Resized to `qd_kernels_.size()` lazily; entries for handles that have never
  // launched a checkpoint-bearing kernel have `per_cp.empty()` and skip the GPU- side gating path entirely (no buffers
  // allocated).
  std::vector<CheckpointHandleState> checkpoint_handle_states_;

  // First-launch setup for the per-handle gating state. Builds the generic gate + yield-check pipelines (idempotent
  // across handles) and populates `state.per_cp` based on the kernel's task list. Body-kernel active dims are baked
  // into `state.per_cp[cp].gate_params` on this call; subsequent launches reuse the same buffers without re-uploading.
  //
  // Returns true if the kernel has any cp_id >= 0 tasks (i.e. the launcher should use the GPU- side gating path); false
  // otherwise (no checkpoints, run the standard direct-dispatch path).
  bool ensure_checkpoint_state_for_handle(KernelHandle handle,
                                          const std::vector<quadrants::lang::spirv::TaskAttributes> &task_attribs,
                                          const std::vector<int> &checkpoint_yield_on_arg_ids,
                                          const std::vector<int> &per_task_group_x);

  // Record a gate-shader dispatch into `cmdlist` for one checkpoint. Implementation in `checkpoint_launch.cpp`. Called
  // from `launch_kernel` immediately before the first body task of each checkpoint. Caller must `memory_barrier()`
  // afterwards before the body tasks dispatch indirect off the gate's out_dims output.
  void dispatch_checkpoint_gate(CommandList *cmdlist, const CheckpointHandleState &state, int cp_id);

  // Record a yield-check-shader dispatch into `cmdlist` for one yielding checkpoint. Implementation in
  // `checkpoint_launch.cpp`. Called from `launch_kernel` immediately after the last body task of each yielding
  // checkpoint. Indirect-dispatched off the trailing out_dims slot so a skipped checkpoint also skips its yield-check.
  void dispatch_checkpoint_yield_check(CommandList *cmdlist,
                                       const CheckpointHandleState &state,
                                       int cp_id,
                                       DeviceAllocation yield_on_devalloc);

  // Per-launch outputs of `prepare_checkpoint_launch_state`. `slot_in_cp[i]` is the dense slot index into
  // `state.per_cp[cp].out_dims` for task `i` (12 bytes per slot, gate-written, body- kernel-indirect-read).
  // `is_first_in_cp[i]` / `is_last_in_cp[i]` mark task-list boundaries used by the per-task dispatch loop to decide
  // when to inject the gate / yield-check shaders. `yield_on_devallocs[cp]` is the user's resolved `yield_on=` ndarray
  // DeviceAllocation for cp (or `kDeviceNullAllocation` for non-yielding checkpoints / cps whose ndarray arg wasn't
  // found in `any_arrays`).
  struct CheckpointLaunchPlan {
    std::vector<int> slot_in_cp;
    std::vector<bool> is_first_in_cp;
    std::vector<bool> is_last_in_cp;
    std::vector<DeviceAllocation> yield_on_devallocs;
  };

  // Per-launch preparation of the checkpoint subsystem on the GPU side. Builds the task-index -> slot map, resolves
  // `yield_on=` DeviceAllocations, conditionally resets `last_yield_cp_id_on_last_call_`, and uploads the initial
  // `(resume_point, -1)` words into the control buffer. Caller passes the freshly resolved `any_arrays` and the
  // pre-launch `host_ctx.checkpoint_yield_on_arg_ids`. Implementation in `checkpoint_launch.cpp`. Returns an empty /
  // default-constructed plan when the kernel has no checkpoints; runtime.cpp's main loop checks
  // `kernel_has_checkpoints` before consuming any of the plan's vectors.
  CheckpointLaunchPlan prepare_checkpoint_launch_state(
      KernelHandle handle,
      const LaunchContextBuilder &host_ctx,
      const std::vector<quadrants::lang::spirv::TaskAttributes> &task_attribs,
      const std::unordered_map<int, DeviceAllocation> &any_arrays,
      bool kernel_has_checkpoints);

  // Post-launch readback: flushes the cmdlist, blocks until the GPU drains, reads the 8-byte `(resume_point,
  // yield_signal)` tuple back from the control buffer, and updates `last_yield_cp_id_on_last_call_` if a yield fired.
  // Re-opens the cmdlist on return so any downstream work (e.g. the D2H blit of return values) can keep recording.
  // Implementation in `checkpoint_launch.cpp`. Skips work entirely (still re-opens the cmdlist via the same
  // `ensure_current_cmdlist()` call the caller would have made) when `kernel_has_checkpoints` is false or no checkpoint
  // in this kernel has `yield_on=`.
  void finalize_checkpoint_readback(KernelHandle handle,
                                    const LaunchContextBuilder &host_ctx,
                                    bool kernel_has_checkpoints);
};

GfxRuntime::RegisterParams run_codegen(Kernel *kernel,
                                       Arch arch,
                                       const DeviceCapabilityConfig &caps,
                                       const std::vector<CompiledSNodeStructs> &compiled_structs,
                                       const CompileConfig &compile_config);

}  // namespace gfx
}  // namespace quadrants::lang
