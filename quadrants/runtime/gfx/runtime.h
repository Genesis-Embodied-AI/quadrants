#pragma once
#include "quadrants/util/lang_util.h"

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
    const QuadrantsKernelAttributes *ti_kernel_attribs{nullptr};
    std::vector<std::vector<uint32_t>> spirv_bins;
    std::size_t num_snode_trees{0};

    Device *device{nullptr};
    std::vector<DeviceAllocation *> root_buffers;
    DeviceAllocation *global_tmps_buffer{nullptr};
    DeviceAllocation *listgen_buffer{nullptr};

    PipelineCache *backend_cache{nullptr};
  };

  explicit CompiledQuadrantsKernel(const Params &ti_params);

  const QuadrantsKernelAttributes &ti_kernel_attribs() const;

  size_t num_pipelines() const;

  size_t get_args_buffer_size() const;
  size_t get_ret_buffer_size() const;

  Pipeline *get_pipeline(int i);

  DeviceAllocation *get_buffer_bind(const BufferInfo &bind) {
    return input_buffers_[bind];
  }

 private:
  QuadrantsKernelAttributes ti_kernel_attribs_;
  std::vector<TaskAttributes> tasks_attribs_;

  [[maybe_unused]] Device *device_;

  InputBuffersMap input_buffers_;

  size_t args_buffer_size_{0};
  size_t ret_buffer_size_{0};
  std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

class QD_DLL_EXPORT GfxRuntime {
 public:
  struct Params {
    Device *device{nullptr};
    KernelProfilerBase *profiler{nullptr};
    // Back-reference to the owning `GfxProgramImpl` so `launch_kernel` can reach `ProgramImpl::program` and
    // evaluate per-task `SerializedSizeExpr` trees captured in `TaskAttributes::ad_stack.allocas[i].size_expr`
    // against the live field state (via `SNodeRwAccessorsBank`) + the per-launch `LaunchContextBuilder` args.
    // Null is tolerated for pre-`materialize_runtime` construction paths; only the adstack-metadata publish
    // path uses it, and kernels without any adstack do not trigger that path.
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

  void launch_kernel(KernelHandle handle, LaunchContextBuilder &host_ctx);

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

  // Single u32 SSBO written by kernels that overflow an adstack. Allocated lazily on the first launch that binds
  // BufferType::AdStackOverflow and then reused across launches; synchronize() reads it, raises if non-zero, and
  // zeros it for the next window.
  std::unique_ptr<DeviceAllocationGuard> adstack_overflow_buffer_;

  // Per-dispatch heaps for SPIR-V adstack primal/adjoint storage. The float heap backs f32-valued adstacks; the
  // int heap backs i32 and u1 adstacks (u1 stored as i32 to match the historical Function-scope path's bool->int
  // remap). Other primitive types (f64, i64, ...) are hard-errored in the shader codegen (no fallback). Each heap
  // is sized at `stride * (group_x * block_dim) * sizeof(element)` and grown lazily; reused across launches
  // whenever the current allocation is already big enough. On grow, the previous buffer is moved into
  // `ctx_buffers_` rather than freed synchronously, so any in-flight cmdlist still referencing it stays valid
  // until the stream drains.
  std::unique_ptr<DeviceAllocationGuard> adstack_heap_buffer_float_;
  size_t adstack_heap_buffer_float_size_{0};
  std::unique_ptr<DeviceAllocationGuard> adstack_heap_buffer_int_;
  size_t adstack_heap_buffer_int_size_{0};
  // Per-`GfxRuntime` compiled sizer pipeline and bytecode scratch buffer for the on-device adstack
  // SizeExpr interpreter (see `quadrants/codegen/spirv/adstack_sizer_shader.{h,cpp}`). The pipeline is
  // built once lazily on the first reverse-mode kernel launch that has adstack allocas and reused across
  // every such launch afterwards; the bytecode buffer is grown on demand with the same
  // amortised-doubling policy as the float / int heaps. Both are null on backends that don't advertise
  // both `spirv_has_physical_storage_buffer` and `spirv_has_int64`, in which case the adstack-allocating
  // kernel is hard-errored at launch time rather than routed to a broken host-eval fallback.
  std::unique_ptr<Pipeline> adstack_sizer_pipeline_{nullptr};
  std::unique_ptr<DeviceAllocationGuard> adstack_sizer_bytecode_buffer_;
  size_t adstack_sizer_bytecode_buffer_size_{0};

  // Owning `ProgramImpl` back-reference; propagated from `Params::program_impl`. See the comment on
  // `Params::program_impl` for the contract.
  ProgramImpl *program_impl_{nullptr};

  // Set by the destructor before its own `synchronize()` call so the adstack-overflow poll in `synchronize()`
  // short-circuits instead of raising from an implicitly-noexcept `~GfxRuntime()` unwinding path (a throw
  // there would call `std::terminate()` and crash the process; the user-visible raise should happen at the
  // user's own `qd.sync()` site, not during teardown). Mirrors LlvmProgramImpl's `finalizing_` flag.
  bool finalizing_{false};

  std::unique_ptr<CommandList> current_cmdlist_{nullptr};
  high_res_clock::time_point current_cmdlist_pending_since_;

  // Counts kernel launches since the last `synchronize()`. `submit_current_cmdlist_if_timeout` forces a
  // drain once this crosses a threshold, bounding the growth of `VulkanStream::submitted_cmdbuffers_` (and
  // the fences, semaphores and descriptor sets those entries keep alive) on tight kernel-launch loops that
  // never touch a Python-side observable - workloads like MPM88 where every substep is a pure GPU update
  // and the host only reads state once at the end. See the assignment site for the MoltenVK SIGSEGV this
  // guards against.
  size_t pending_launches_since_sync_{0};

  std::vector<std::unique_ptr<CompiledQuadrantsKernel>> ti_kernels_;

  std::unordered_map<DeviceAllocation *, size_t> root_buffers_size_map_;
  std::unordered_map<DeviceAllocationId, ImageLayout> last_image_layouts_;
  // [Note] Why do we need to track ndarrays that are in use?
  // Since we separate cmdlist is async, quadrants needs a way to know whether
  // ndarrays are still used by pending kernels to be executed. So we use
  // ndarray_in_use_ to track this so that we can free memory allocated for
  // ndarray whenever it's safe to do so.
  std::unordered_set<DeviceAllocationId> ndarrays_in_use_;
};

GfxRuntime::RegisterParams run_codegen(Kernel *kernel,
                                       Arch arch,
                                       const DeviceCapabilityConfig &caps,
                                       const std::vector<CompiledSNodeStructs> &compiled_structs,
                                       const CompileConfig &compile_config);

}  // namespace gfx
}  // namespace quadrants::lang
