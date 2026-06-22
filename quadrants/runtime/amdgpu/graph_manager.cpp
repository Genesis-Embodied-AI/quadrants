#include "quadrants/runtime/amdgpu/graph_manager.h"
#include "quadrants/runtime/amdgpu/amdgpu_utils.h"
#include "quadrants/runtime/amdgpu/checkpoint_yield_check_hsaco.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace quadrants::lang {
namespace amdgpu {

namespace {

// Layout of the body-kernel `extra_config` block. Reused for the yield-check kernel block too; only the `packed_args`
// contents and `pack_size` differ between the two.
constexpr unsigned long long kHipLaunchParamBufferPointer = 0x01;
constexpr unsigned long long kHipLaunchParamBufferSize = 0x02;
constexpr unsigned long long kHipLaunchParamEnd = 0x03;

// (Re)bind a CachedKernelArgs's `extra_config` to point at its own `packed_args` / `pack_size`. Called by the
// constructor and after every move (the in-place pointers would otherwise still reference the moved-from object).
void rebind_kernel_args_self(CachedKernelArgs &kernel_args) {
  kernel_args.extra_config[0] = reinterpret_cast<void *>(kHipLaunchParamBufferPointer);
  kernel_args.extra_config[1] = &kernel_args.packed_args[0];
  kernel_args.extra_config[2] = reinterpret_cast<void *>(kHipLaunchParamBufferSize);
  kernel_args.extra_config[3] = &kernel_args.pack_size;
  kernel_args.extra_config[4] = reinterpret_cast<void *>(kHipLaunchParamEnd);
}

// Initialise a body-kernel CachedKernelArgs: 8-byte packed buffer carrying the device-side RuntimeContext pointer.
void initialize_body_kernel_args(CachedKernelArgs &kernel_args, void *device_runtime_ctx) {
  std::memcpy(&kernel_args.packed_args[0], &device_runtime_ctx, sizeof(void *));
  kernel_args.pack_size = sizeof(void *);
  rebind_kernel_args_self(kernel_args);
}

}  // namespace

CachedGraph::CachedGraph(std::size_t arg_buf_size,
                         std::size_t result_buf_size,
                         bool needs_checkpoint_scalars,
                         LlvmRuntimeExecutor *executor)
    : arg_buffer_size(arg_buf_size), result_buffer_size(result_buf_size) {
  AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_result_buffer),
                                      std::max(result_buffer_size, sizeof(uint64)));

  if (arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_arg_buffer), arg_buffer_size);
  }

  // Device-side `RuntimeContext` for graph kernel-node args. Unlike the CUDA path which passes a host pointer (relying
  // on UVA / HMM), AMDGPU kernels dereference the pointer directly on the GPU, so it must point at device memory.
  AMDGPUDriver::get_instance().malloc(&device_runtime_ctx, sizeof(RuntimeContext));

  if (needs_checkpoint_scalars) {
    // resume_point and yield_signal int32 device scalars read by the codegen prologue (in every cp_id >= 0 body kernel)
    // and the yield-check kernel. Zero-init / -1-init matches "no resume in progress" / "no yield observed". The
    // per-launch HtoD in `launch_cached_graph` overwrites them before every launch, so the initial values are only
    // consumed by the very first launch's first body kernel before the per-launch HtoD finishes; we still set them
    // defensively here.
    AMDGPUDriver::get_instance().malloc(&resume_point_dev_ptr, sizeof(int32_t));
    int32_t zero = 0;
    AMDGPUDriver::get_instance().memcpy_host_to_device(resume_point_dev_ptr, &zero, sizeof(int32_t));

    AMDGPUDriver::get_instance().malloc(&yield_signal_dev_ptr, sizeof(int32_t));
    int32_t neg_one = -1;
    AMDGPUDriver::get_instance().memcpy_host_to_device(yield_signal_dev_ptr, &neg_one, sizeof(int32_t));
  }

  persistent_ctx.runtime = executor->get_llvm_runtime();
  persistent_ctx.arg_buffer = persistent_device_arg_buffer;
  persistent_ctx.result_buffer = reinterpret_cast<uint64 *>(persistent_device_result_buffer);
  persistent_ctx.cpu_thread_id = 0;
  // GPU-side checkpoint gating: publish the device-side resume_point / yield_signal pointers to the RuntimeContext
  // slots that the codegen-emitted body-kernel prologue reads. Null-pointers stay null for kernels without checkpoints
  // (the prologue is never emitted, so the read never happens; we still write nullptr to keep the runtime struct fully
  // initialised).
  persistent_ctx.checkpoint_resume_point_ptr = reinterpret_cast<int32_t *>(resume_point_dev_ptr);
  persistent_ctx.checkpoint_yield_signal_ptr = reinterpret_cast<int32_t *>(yield_signal_dev_ptr);

  // Stage the single shared body-kernel CachedKernelArgs. Yield-check CachedKernelArgs are created per yielding
  // checkpoint during graph build (see initialize_yield_check_kernel_args).
  initialize_body_kernel_args(kernel_args, device_runtime_ctx);
}

CachedGraph::~CachedGraph() {
  if (graph_exec) {
    AMDGPUDriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_result_buffer);
  }
  if (device_runtime_ctx) {
    AMDGPUDriver::get_instance().mem_free(device_runtime_ctx);
  }
  if (resume_point_dev_ptr) {
    AMDGPUDriver::get_instance().mem_free(resume_point_dev_ptr);
  }
  if (yield_signal_dev_ptr) {
    AMDGPUDriver::get_instance().mem_free(yield_signal_dev_ptr);
  }
  for (void *slot : checkpoint_yield_on_ptr_slots) {
    if (slot) {
      AMDGPUDriver::get_instance().mem_free(slot);
    }
  }
}

CachedGraph::CachedGraph(CachedGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      device_runtime_ctx(other.device_runtime_ctx),
      kernel_args(other.kernel_args),
      yield_check_kernel_args(std::move(other.yield_check_kernel_args)),
      yield_check_cp_id_storage(std::move(other.yield_check_cp_id_storage)),
      checkpoint_yield_on_ptr_slots(std::move(other.checkpoint_yield_on_ptr_slots)),
      resume_point_dev_ptr(other.resume_point_dev_ptr),
      yield_signal_dev_ptr(other.yield_signal_dev_ptr),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      num_nodes(other.num_nodes),
      num_checkpoints(other.num_checkpoints) {
  // The moved-from object must not free our resources at destruction time. Rebind every CachedKernelArgs's
  // `extra_config` so it points at *our* `packed_args` / `pack_size`, not the moved-from object's.
  rebind_kernel_args_self(kernel_args);
  for (auto &yk : yield_check_kernel_args) {
    rebind_kernel_args_self(yk);
  }
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.device_runtime_ctx = nullptr;
  other.resume_point_dev_ptr = nullptr;
  other.yield_signal_dev_ptr = nullptr;
  // The std::move'd vectors are left empty by the moves above, so `other.~CachedGraph()` will not try to release any
  // per-cp yield-on ptr slots that now belong to us.
}

CachedGraph &CachedGraph::operator=(CachedGraph &&other) noexcept {
  // Move-and-swap: after the swaps, `raii_guard` holds our old resources and its destructor frees them.
  CachedGraph raii_guard(std::move(other));
  std::swap(graph_exec, raii_guard.graph_exec);
  std::swap(persistent_device_arg_buffer, raii_guard.persistent_device_arg_buffer);
  std::swap(persistent_device_result_buffer, raii_guard.persistent_device_result_buffer);
  std::swap(persistent_ctx, raii_guard.persistent_ctx);
  std::swap(device_runtime_ctx, raii_guard.device_runtime_ctx);
  std::swap(kernel_args, raii_guard.kernel_args);
  std::swap(yield_check_kernel_args, raii_guard.yield_check_kernel_args);
  std::swap(yield_check_cp_id_storage, raii_guard.yield_check_cp_id_storage);
  std::swap(checkpoint_yield_on_ptr_slots, raii_guard.checkpoint_yield_on_ptr_slots);
  std::swap(resume_point_dev_ptr, raii_guard.resume_point_dev_ptr);
  std::swap(yield_signal_dev_ptr, raii_guard.yield_signal_dev_ptr);
  std::swap(arg_buffer_size, raii_guard.arg_buffer_size);
  std::swap(result_buffer_size, raii_guard.result_buffer_size);
  std::swap(num_nodes, raii_guard.num_nodes);
  std::swap(num_checkpoints, raii_guard.num_checkpoints);
  // After the swap, every CachedKernelArgs's `extra_config` still references the *raii_guard*'s storage (which is about
  // to die). Rebind both sides to point at their post-swap owner's `packed_args` / `pack_size`.
  rebind_kernel_args_self(kernel_args);
  for (auto &yk : yield_check_kernel_args) {
    rebind_kernel_args_self(yk);
  }
  rebind_kernel_args_self(raii_guard.kernel_args);
  for (auto &yk : raii_guard.yield_check_kernel_args) {
    rebind_kernel_args_self(yk);
  }
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device pointers, writing them into the arg buffer
// via `set_ndarray_ptrs`. Unlike the normal launch path, does not handle host-resident arrays (no temporary device
// allocation or host-to-device transfer). Errors if any external array is on the host, since graph mode bakes device
// pointers into the cached graph.
void GraphManager::resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                            const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                            LlvmRuntimeExecutor *executor) {
  ctx.checkpoint_yield_on_dev_ptrs.assign(ctx.checkpoint_yield_on_arg_ids.size(), nullptr);

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0)
        continue;

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      QD_ERROR_IF(grad_ptr != nullptr,
                  "graph does not support autograd; "
                  "ndarray arg {} has a non-null gradient pointer",
                  arg_id);

      void *resolved_data = nullptr;
      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        QD_ERROR_IF(!on_amdgpu_device(data_ptr),
                    "graph requires all ndarrays to be device-resident; "
                    "ndarray arg {} is host-resident",
                    arg_id);
        resolved_data = data_ptr;
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        resolved_data = executor->get_device_alloc_info_ptr(*ptr);
      }

      if (resolved_data) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_data, (uint64) nullptr);
        // Resolve every graph_do_while level whose condition ndarray is this arg (multi-level table).
        ctx.resolve_graph_do_while_flag(arg_id, resolved_data);
        // Route this ndarray into the per-cp yield-flag table for every checkpoint that named it.
        for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_arg_ids.size(); ++cp) {
          if (ctx.checkpoint_yield_on_arg_ids[cp] == arg_id) {
            ctx.checkpoint_yield_on_dev_ptrs[cp] = resolved_data;
          }
        }
      }
    }
  }
}

void *GraphManager::add_kernel_node(void *graph,
                                    void *prev_node,
                                    void *func,
                                    unsigned int grid_dim,
                                    unsigned int block_dim,
                                    unsigned int shared_mem,
                                    CachedKernelArgs &kernel_args) {
  // Opt in to the requested dynamic shared memory size. `hipFuncSetAttribute` is mostly a no-op for the AMD backend
  // per its header comments, but we call it for parity with the CUDA path and to forward the request to any backend
  // that honours it.
  if (shared_mem > 0) {
    AMDGPUDriver::get_instance().kernel_set_attribute(func, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                      static_cast<int>(shared_mem));
  }

  HipKernelNodeParams params{};
  params.blockDimX = block_dim;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  // HIP's AMD backend expects kernel args via the `extra` byte-buffer convention (HIP_LAUNCH_PARAM_BUFFER_POINTER /
  // SIZE / END markers), not via the per-arg `kernelParams` array. See the matching setup in
  // `rhi/amdgpu/amdgpu_context.cpp::AMDGPUContext::launch`. Passing `kernelParams` here instead silently corrupts
  // kernel arg loads on RDNA3 and the launched kernels fault asynchronously, surfacing as `hipErrorIllegalAddress` at
  // the next host-visible sync point.
  params.extra = kernel_args.extra_config;
  params.func = func;
  params.gridDimX = grid_dim;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.kernelParams = nullptr;
  params.sharedMemBytes = shared_mem;

  void *node = nullptr;
  AMDGPUDriver::get_instance().graph_add_kernel_node(&node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
                                                     &params);
  return node;
}

// Lazy load of the AMDGPU yield-check kernel from the pre-built bundled HSACO. The bundle in
// `checkpoint_yield_check_hsaco.h` is built by `scripts/build_checkpoint_yield_check_hsaco.py` for every AMD arch
// quadrants targets (gfx90a / gfx942 / gfx1030 / gfx1100 / 1101 / 1102 / 1200 / 1201). `hipModuleLoadData`
// demultiplexes the bundle and picks the right HSACO for the current device.
void GraphManager::ensure_checkpoint_yield_check_kernel_loaded() {
  if (yield_check_kernel_func_)
    return;

  auto &driver = AMDGPUDriver::get_instance();

  static_assert(kCheckpointYieldCheckKernelHsacoSize > 0,
                "AMDGPU checkpoint yield-check kernel HSACO bundle is empty -- regenerate with "
                "scripts/build_checkpoint_yield_check_hsaco.py on a machine with ROCm installed");

  uint32_t ret = driver.module_load_data.call(&yield_check_kernel_module_, kCheckpointYieldCheckKernelHsaco);
  QD_ERROR_IF(ret != 0,
              "Failed to load AMDGPU qd.checkpoint yield-check kernel HSACO (HIP error {}). The current device's "
              "arch may not be in the bundle -- regenerate with "
              "scripts/build_checkpoint_yield_check_hsaco.py and add the missing arch to OFFLOAD_ARCHES",
              ret);

  driver.module_get_function(&yield_check_kernel_func_, yield_check_kernel_module_, "_qd_checkpoint_yield_check");
  QD_TRACE("Loaded AMDGPU qd.checkpoint yield-check kernel from pre-built HSACO bundle");
}

// Initialise a yield-check kernel's CachedKernelArgs: 32-byte packed buffer with C-struct layout matching
// `extern "C" __global__ void _qd_checkpoint_yield_check(int32_t **yield_on_ptr_slot, int32_t cp_id,
// int32_t *yield_signal, int32_t *resume_point)`. Layout:
//   offset 0:  yield_on_ptr_slot ptr (8 bytes)
//   offset 8:  cp_id int32          (4 bytes)
//   offset 12: padding              (4 bytes)
//   offset 16: yield_signal ptr     (8 bytes)
//   offset 24: resume_point ptr     (8 bytes)
//   total:     32 bytes
void GraphManager::initialize_yield_check_kernel_args(CachedKernelArgs &kernel_args,
                                                      void *yield_on_ptr_slot_addr,
                                                      int32_t *cp_id_storage,
                                                      void *yield_signal_dev_ptr,
                                                      void *resume_point_dev_ptr) {
  unsigned char *p = kernel_args.packed_args;
  std::memset(p, 0, sizeof(kernel_args.packed_args));
  std::memcpy(p + 0, &yield_on_ptr_slot_addr, sizeof(void *));
  std::memcpy(p + 8, cp_id_storage, sizeof(int32_t));
  std::memcpy(p + 16, &yield_signal_dev_ptr, sizeof(void *));
  std::memcpy(p + 24, &resume_point_dev_ptr, sizeof(void *));
  kernel_args.pack_size = 32;
  rebind_kernel_args_self(kernel_args);
}

bool GraphManager::launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx) {
  // Reset the yield bookkeeping. Mirrors the CUDA equivalent: `-1` means "no yield observed (yet)" for both plain-graph
  // kernels (where the question has no meaning) and checkpoint kernels (where the post-launch DtoH below will overwrite
  // it if a yield fired).
  last_yield_cp_id_on_last_call_ = -1;

  auto *stream = AMDGPUContext::get_instance().get_stream();

  if (cached.arg_buffer_size > 0) {
    // Async HtoD on the launch stream: the subsequent `graph_launch` is queued on the same stream, so the kernel nodes
    // are ordered after the arg-buffer upload without a host-side barrier.
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer, cached.arg_buffer_size, stream);
  }

  const bool has_checkpoints = (cached.resume_point_dev_ptr != nullptr);
  const bool has_yield_state = (cached.yield_signal_dev_ptr != nullptr);

  if (has_checkpoints) {
    // GPU-side checkpoint gating: write the per-launch resume_point so the codegen prologue / yield-check kernels see
    // the right value. `ctx.resume_from_checkpoint < 0` means "no resume requested"; map to 0 so every checkpoint
    // (cp_id >= 0 >= 0) runs.
    int32_t resume_point = (ctx.resume_from_checkpoint < 0) ? 0 : ctx.resume_from_checkpoint;
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(cached.resume_point_dev_ptr, &resume_point,
                                                             sizeof(int32_t), stream);
    if (has_yield_state) {
      int32_t neg_one = -1;
      AMDGPUDriver::get_instance().memcpy_host_to_device_async(cached.yield_signal_dev_ptr, &neg_one, sizeof(int32_t),
                                                               stream);
    }
    // Rewrite the per-cp yield-on slot pointers so each yield-check kernel sees the user's current `yield_on=` ndarray
    // address. The slot addresses are baked into the graph; the contents are host-updated each launch.
    for (std::size_t cp = 0; cp < cached.checkpoint_yield_on_ptr_slots.size(); ++cp) {
      if (cached.checkpoint_yield_on_ptr_slots[cp] == nullptr) {
        continue;
      }
      // Look up the user's current device pointer for this cp (resolve_ctx_ndarray_ptrs populated
      // ctx.checkpoint_yield_on_dev_ptrs). Defensive default to nullptr if the user passed no ndarray; the yield- check
      // kernel reads through the slot and would null-deref, but the graph build rejected that earlier (a yielding
      // checkpoint without a resolved ndarray fails resolve_ctx_ndarray_ptrs).
      void *user_ptr = nullptr;
      if (cp < ctx.checkpoint_yield_on_dev_ptrs.size()) {
        user_ptr = ctx.checkpoint_yield_on_dev_ptrs[cp];
      }
      AMDGPUDriver::get_instance().memcpy_host_to_device_async(cached.checkpoint_yield_on_ptr_slots[cp], &user_ptr,
                                                               sizeof(void *), stream);
    }
  }

  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);
  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;

  if (has_yield_state) {
    // Post-launch DtoH of yield_signal. Mirrors CUDA's single-D2H-per-launch model: one sync per launch regardless of
    // how many yielding checkpoints the graph contains. The synchronize is implicit in the synchronous DtoH memcpy; HIP
    // guarantees the host-visible value reflects all prior graph writes on this stream.
    int32_t signal = -1;
    AMDGPUDriver::get_instance().stream_synchronize(stream);
    AMDGPUDriver::get_instance().memcpy_device_to_host(&signal, cached.yield_signal_dev_ptr, sizeof(int32_t));
    if (signal != -1) {
      last_yield_cp_id_on_last_call_ = signal;
    }
  }

  return true;
}

bool GraphManager::try_launch(int launch_id,
                              LaunchContextBuilder &ctx,
                              JITModule *amdgpu_module,
                              const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              LlvmRuntimeExecutor *executor) {
  if (offloaded_tasks.empty()) {
    return false;
  }

  QD_ERROR_IF(ctx.result_buffer_size > 0,
              "graph=True is not supported for kernels with struct return "
              "values; remove graph=True or avoid returning values");

  // Adstack-bearing kernels cannot go through the graph path. See the matching comment in
  // `runtime/cuda/graph_manager.cpp::try_launch` for the full rationale: the per-task adstack setup runs host-side
  // between the serial range_for-bounds kernel and the range_for kernel itself, and there is no host hook between graph
  // nodes.
  for (const auto &task : offloaded_tasks) {
    QD_ERROR_IF(!task.ad_stack.allocas.empty(),
                "graph=True is not supported for kernels that use the reverse-mode autodiff stack "
                "(task '{}' has {} adstack allocas). Launch without graph=True.",
                task.name, task.ad_stack.allocas.size());
  }

  resolve_ctx_ndarray_ptrs(ctx, parameters, executor);

  auto it = cache_.find(launch_id);
  if (it != cache_.end()) {
    return launch_cached_graph(it->second, ctx);
  }

  // Determine checkpoint shape up-front so the CachedGraph constructor can size the device-side resume_point /
  // yield_signal scalars appropriately and the build loop below can route yield-check kernels in.
  bool has_checkpoints = false;
  int max_cp_id = -1;
  std::size_t num_distinct_checkpoints = 0;
  {
    int prev_cp = -1;
    for (const auto &task : offloaded_tasks) {
      if (task.checkpoint_id >= 0) {
        has_checkpoints = true;
        max_cp_id = std::max(max_cp_id, task.checkpoint_id);
        if (task.checkpoint_id != prev_cp) {
          ++num_distinct_checkpoints;
        }
        prev_cp = task.checkpoint_id;
      } else {
        prev_cp = -1;
      }
    }
  }

  bool has_yield = false;
  for (void *p : ctx.checkpoint_yield_on_dev_ptrs) {
    if (p) {
      has_yield = true;
      break;
    }
  }

  if (has_yield) {
    ensure_checkpoint_yield_check_kernel_loaded();
    if (!yield_check_kernel_func_) {
      // Yield-check fatbin didn't cover the current AMD arch. Fall back to the streaming launcher (which still does
      // host-branch gating today; tracked separately for the same prologue treatment as the flat-graph path).
      return false;
    }
  }

  AMDGPUContext::get_instance().make_current();
  CachedGraph cached(ctx.arg_buffer_size, ctx.result_buffer_size, has_checkpoints, executor);
  cached.num_checkpoints = num_distinct_checkpoints;

  // Per-cp yield-on slot table. Sized by max_cp_id + 1 so the host-side update loop in launch_cached_graph can index by
  // cp_id directly. Slots for non-yielding checkpoints stay nullptr. Mirrors CUDA's slot table.
  if (has_yield && max_cp_id >= 0) {
    cached.checkpoint_yield_on_ptr_slots.assign((std::size_t)max_cp_id + 1, nullptr);
    for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_dev_ptrs.size() && (int)cp <= max_cp_id; ++cp) {
      if (ctx.checkpoint_yield_on_dev_ptrs[cp]) {
        void *slot = nullptr;
        AMDGPUDriver::get_instance().malloc(&slot, sizeof(void *));
        void *user_ptr = ctx.checkpoint_yield_on_dev_ptrs[cp];
        AMDGPUDriver::get_instance().memcpy_host_to_device(slot, &user_ptr, sizeof(void *));
        cached.checkpoint_yield_on_ptr_slots[cp] = slot;
      }
    }
  }

  // Reserve the per-yielding-checkpoint storage up-front so the cp_id literals and CachedKernelArgs entries get stable
  // addresses for the lifetime of the graph (the kernel-node `extra` config baked into the HIP graph holds raw pointers
  // into these vectors).
  cached.yield_check_cp_id_storage.reserve(num_distinct_checkpoints);
  cached.yield_check_kernel_args.reserve(num_distinct_checkpoints);

  auto *stream_for_setup = AMDGPUContext::get_instance().get_stream();
  if (cached.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer, cached.arg_buffer_size, stream_for_setup);
  }
  // Stage the RuntimeContext on device. Its arg_buffer / result_buffer pointers reference the persistent device
  // buffers above; none of its fields change between graph launches so one copy is sufficient.
  AMDGPUDriver::get_instance().memcpy_host_to_device_async(cached.device_runtime_ctx, &cached.persistent_ctx,
                                                           sizeof(RuntimeContext), stream_for_setup);
  AMDGPUDriver::get_instance().stream_synchronize(stream_for_setup);

  // --- Build the flat HIP graph ---
  void *graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&graph, 0);

  // Helper: emit the yield-check kernel node for a just-closed checkpoint. Stable addresses are required: cp_id is
  // stored in `cached.yield_check_cp_id_storage` (reserved above), the packed-args block lives in the CachedKernelArgs
  // stashed into `cached.yield_check_kernel_args`, and the yield-on slot's address comes from
  // `cached.checkpoint_yield_on_ptr_slots[cp_id]`.
  std::size_t total_nodes = 0;
  void *prev_node = nullptr;
  int current_cp_id = -1;

  auto emit_yield_check_for_closed_cp = [&]() {
    if (current_cp_id < 0)
      return;
    if ((std::size_t)current_cp_id >= cached.checkpoint_yield_on_ptr_slots.size())
      return;
    if (!cached.checkpoint_yield_on_ptr_slots[current_cp_id])
      return;
    cached.yield_check_cp_id_storage.push_back(current_cp_id);
    cached.yield_check_kernel_args.emplace_back();
    auto &kargs = cached.yield_check_kernel_args.back();
    // The kernel expects `int32_t **yield_on_ptr_slot` to be a *device* pointer to a device-side slot that holds the
    // user's `yield_on=` ndarray address. `checkpoint_yield_on_ptr_slots[cp]` already IS that device pointer (it was
    // returned by `AMDGPUDriver::malloc` above), so we pass it directly -- NOT `&...slots[cp]`, which would be the
    // host address of the std::vector element and would null-deref on the GPU.
    initialize_yield_check_kernel_args(kargs, cached.checkpoint_yield_on_ptr_slots[current_cp_id],
                                       &cached.yield_check_cp_id_storage.back(), cached.yield_signal_dev_ptr,
                                       cached.resume_point_dev_ptr);
    prev_node = add_kernel_node(graph, prev_node, yield_check_kernel_func_, /*grid=*/1, /*block=*/1, /*smem=*/0, kargs);
    ++total_nodes;
  };

  // Each body-kernel node receives the device-side RuntimeContext pointer via the shared `cached.kernel_args`
  // extra-config (see graph_manager.h for why all body nodes share one). Stream-parallel groups
  // (`stream_parallel_group_id != 0`) are silently serialized inside the graph, matching the CUDA implementation.
  for (const auto &task : offloaded_tasks) {
    if (task.checkpoint_id != current_cp_id) {
      emit_yield_check_for_closed_cp();
      current_cp_id = task.checkpoint_id;
    }
    void *func = amdgpu_module->lookup_function(task.name);
    prev_node = add_kernel_node(graph, prev_node, func, (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
                                (unsigned int)task.dynamic_shared_array_bytes, cached.kernel_args);
    ++total_nodes;
  }
  // Flush the trailing checkpoint's yield-check kernel if the loop ended inside one.
  emit_yield_check_for_closed_cp();

  AMDGPUDriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);
  AMDGPUDriver::get_instance().graph_destroy(graph);
  cached.num_nodes = total_nodes;

  QD_TRACE("HIP graph created with {} nodes ({} checkpoints) for launch_id={}", cached.num_nodes,
           cached.num_checkpoints, launch_id);

  ++total_builds_;
  auto [it_inserted, ok] = cache_.emplace(launch_id, std::move(cached));
  // Funnel through `launch_cached_graph` so the post-launch bookkeeping (`used_on_last_call_`,
  // `num_nodes_on_last_call_`, yield D2H) is identical for the initial build and every subsequent cache hit.
  return launch_cached_graph(it_inserted->second, ctx);
}

}  // namespace amdgpu
}  // namespace quadrants::lang
