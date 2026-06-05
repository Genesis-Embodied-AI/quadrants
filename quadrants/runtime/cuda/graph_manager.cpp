#include "quadrants/runtime/cuda/graph_manager.h"
#include "quadrants/runtime/cuda/checkpoint_gate_fatbin.h"
#include "quadrants/runtime/cuda/checkpoint_yield_check_fatbin.h"
#include "quadrants/runtime/cuda/graph_do_while_cond_fatbin.h"
#include "quadrants/runtime/cuda/cuda_utils.h"
#include "quadrants/rhi/cuda/cuda_context.h"

#include <climits>
#include <cstdint>
#include <vector>

namespace quadrants::lang {
namespace cuda {

CachedGraph::CachedGraph(std::size_t arg_buf_size,
                         std::size_t result_buf_size,
                         bool needs_counter_ptr_slot,
                         bool needs_resume_point_slot,
                         bool needs_yield_signal_slot,
                         LlvmRuntimeExecutor *executor)
    : arg_buffer_size(arg_buf_size), result_buffer_size(result_buf_size) {
  CUDADriver::get_instance().malloc((void **)&persistent_device_result_buffer,
                                    std::max(result_buffer_size, sizeof(uint64)));

  if (arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc((void **)&persistent_device_arg_buffer, arg_buffer_size);
  }

  if (needs_counter_ptr_slot) {
    CUDADriver::get_instance().malloc(&counter_ptr_slot, sizeof(void *));
  }

  if (needs_resume_point_slot) {
    // One int32 device scalar that the checkpoint gate kernels read every launch. Zero-init
    // matches "no resume in progress" -- every gate sees `cp_id >= 0` and enables its body.
    CUDADriver::get_instance().malloc(&resume_point_dev_ptr, sizeof(int32_t));
    int32_t zero = 0;
    CUDADriver::get_instance().memcpy_host_to_device(resume_point_dev_ptr, &zero, sizeof(int32_t));
  }

  if (needs_yield_signal_slot) {
    // Single int32 yield_signal device scalar. -1 means "no checkpoint has yielded this
    // launch". The yield-check kernel atomically CASes the first yielding cp_id in;
    // `launch_cached_graph` resets it to -1 before every launch and copies it back after.
    CUDADriver::get_instance().malloc(&yield_signal_dev_ptr, sizeof(int32_t));
    int32_t neg_one = -1;
    CUDADriver::get_instance().memcpy_host_to_device(yield_signal_dev_ptr, &neg_one, sizeof(int32_t));
  }

  persistent_ctx.runtime = executor->get_llvm_runtime();
  persistent_ctx.arg_buffer = persistent_device_arg_buffer;
  persistent_ctx.result_buffer = (uint64 *)persistent_device_result_buffer;
  persistent_ctx.cpu_thread_id = 0;
}

CachedGraph::~CachedGraph() {
  if (graph_exec) {
    CUDADriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_result_buffer);
  }
  if (counter_ptr_slot) {
    CUDADriver::get_instance().mem_free(counter_ptr_slot);
  }
  if (resume_point_dev_ptr) {
    CUDADriver::get_instance().mem_free(resume_point_dev_ptr);
  }
  if (yield_signal_dev_ptr) {
    CUDADriver::get_instance().mem_free(yield_signal_dev_ptr);
  }
  for (void *slot : checkpoint_yield_on_ptr_slots) {
    if (slot) {
      CUDADriver::get_instance().mem_free(slot);
    }
  }
}

CachedGraph::CachedGraph(CachedGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      counter_ptr_slot(other.counter_ptr_slot),
      resume_point_dev_ptr(other.resume_point_dev_ptr),
      yield_signal_dev_ptr(other.yield_signal_dev_ptr),
      checkpoint_yield_on_ptr_slots(std::move(other.checkpoint_yield_on_ptr_slots)),
      num_checkpoints(other.num_checkpoints),
      num_nodes(other.num_nodes) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.counter_ptr_slot = nullptr;
  other.resume_point_dev_ptr = nullptr;
  other.yield_signal_dev_ptr = nullptr;
  other.checkpoint_yield_on_ptr_slots.clear();
}

CachedGraph &CachedGraph::operator=(CachedGraph &&other) noexcept {
  // Move-and-swap: after the swaps, `raii_guard` holds our old resources and
  // its destructor frees them, so every owned pointer is released uniformly.
  CachedGraph raii_guard(std::move(other));
  std::swap(graph_exec, raii_guard.graph_exec);
  std::swap(persistent_device_arg_buffer, raii_guard.persistent_device_arg_buffer);
  std::swap(persistent_device_result_buffer, raii_guard.persistent_device_result_buffer);
  std::swap(persistent_ctx, raii_guard.persistent_ctx);
  std::swap(arg_buffer_size, raii_guard.arg_buffer_size);
  std::swap(result_buffer_size, raii_guard.result_buffer_size);
  std::swap(counter_ptr_slot, raii_guard.counter_ptr_slot);
  std::swap(resume_point_dev_ptr, raii_guard.resume_point_dev_ptr);
  std::swap(yield_signal_dev_ptr, raii_guard.yield_signal_dev_ptr);
  std::swap(checkpoint_yield_on_ptr_slots, raii_guard.checkpoint_yield_on_ptr_slots);
  std::swap(num_checkpoints, raii_guard.num_checkpoints);
  std::swap(num_nodes, raii_guard.num_nodes);
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device
// pointers, writing them into the arg buffer via set_ndarray_ptrs.
//
// Unlike the normal launch path, this does not handle host-resident arrays
// (no temporary device allocation or host-to-device transfer). Errors if
// any external array is on the host, since graph requires all arrays
// to be device-resident.
void GraphManager::resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                            const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                            LlvmRuntimeExecutor *executor) {
  // Pre-size the per-checkpoint yield_on device-pointer table so the GraphManager can index it
  // by cp_id even when some checkpoints have no yield_on (their slot stays nullptr). Sized
  // exactly once per launch from the matching arg-id table populated in Python `Kernel.__call__`.
  ctx.checkpoint_yield_on_dev_ptrs.assign(ctx.checkpoint_yield_on_arg_ids.size(), nullptr);

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    // Scalar parameters are already in the arg buffer and need no resolution;
    // only array parameters require translating handles to device pointers.
    // Fields are template parameters, and would never arrive here.
    // We only need to handle ndarrays and external arrays.
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

      // Raw device pointer to the array data, resolved from either an
      // external array (raw pointer) or a DeviceAllocation handle.
      void *resolved_data = nullptr;

      if (ctx.device_allocation_type[arg_id] == LaunchContextBuilder::DevAllocType::kNone) {
        QD_ERROR_IF(!on_cuda_device(data_ptr),
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
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = resolved_data;
        }
        // Mirror the graph_do_while resolution for every `qd.checkpoint(yield_on=foo)` --
        // walk the per-cp_id arg-id table and stash the device pointer where the GraphManager
        // build path can find it. Same-arg shared between checkpoints is fine; we just
        // assign the same pointer to multiple slots.
        for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_arg_ids.size(); ++cp) {
          if (ctx.checkpoint_yield_on_arg_ids[cp] == arg_id) {
            ctx.checkpoint_yield_on_dev_ptrs[cp] = resolved_data;
          }
        }
      }
    }
  }
}

// Loads the graph_do_while condition kernel from the pre-built fatbin.
// The fatbin is generated by scripts/build_condition_kernel_fatbin.py and
// contains SASS for all supported SM architectures. Only called once;
// subsequent calls are no-ops.
void GraphManager::ensure_condition_kernel_loaded() {
  if (cond_kernel_func_)
    return;

  int cc = CUDAContext::get_instance().get_compute_capability();
  if (cc < 90) {
    QD_INFO(
        "CUDA graph conditional nodes require SM 9.0+, but this device is "
        "SM {}. Falling back to host-side do-while loop, which is slower.",
        cc);
    return;
  }

  auto &driver = CUDADriver::get_instance();

  static_assert(kConditionKernelFatbinSize > 0,
                "Condition kernel fatbin is empty — regenerate with "
                "scripts/build_condition_kernel_fatbin.py");

  uint32_t ret = driver.module_load_data.call(&cond_kernel_module_, kConditionKernelFatbin);
  QD_ERROR_IF(ret != CUDA_SUCCESS,
              "Failed to load graph_do_while condition kernel fatbin "
              "(CUDA error {}). This SM ({}) may not be included in the "
              "fatbin — regenerate with scripts/build_condition_kernel_fatbin.py",
              ret, cc);

  driver.module_get_function(&cond_kernel_func_, cond_kernel_module_, "_qd_graph_do_while_cond");
  QD_TRACE("Loaded graph_do_while condition kernel from pre-built fatbin");
}

// Loads the cond-with-yield variant from the same module the plain condition kernel lives in
// (both are emitted by `build_condition_kernel_fatbin.py`). Looked up lazily because most
// `qd.graph_do_while` kernels don't contain `qd.checkpoint(yield_on=...)` and don't need it.
// Returns silently when the base module isn't loaded yet -- callers are expected to call
// `ensure_condition_kernel_loaded()` immediately before, mirroring the existing pattern.
void GraphManager::ensure_cond_with_yield_kernel_loaded() {
  if (cond_with_yield_kernel_func_)
    return;
  if (!cond_kernel_module_)
    return;
  CUDADriver::get_instance().module_get_function(&cond_with_yield_kernel_func_, cond_kernel_module_,
                                                 "_qd_graph_do_while_cond_with_yield");
  QD_TRACE("Loaded graph_do_while cond-with-yield kernel from pre-built fatbin");
}

// Loads the qd.checkpoint() IF-gate kernel from the pre-built fatbin (same shape and
// SM-coverage policy as `ensure_condition_kernel_loaded`). The CUDA 12.4+ IF conditional
// node mechanism is gated on SM 9.0+; on older devices we early-return and the caller
// falls back to flattening checkpoints into unconditional top-level kernels (slice 1c
// keeps the same behaviour-equivalent fallback as today's `graph_do_while` plus a clear
// log so users know why they aren't seeing the IF path).
void GraphManager::ensure_checkpoint_gate_kernel_loaded() {
  if (gate_kernel_func_)
    return;

  int cc = CUDAContext::get_instance().get_compute_capability();
  if (cc < 90) {
    QD_INFO(
        "CUDA graph IF conditional nodes (used by qd.checkpoint) require SM 9.0+, "
        "but this device is SM {}. Falling back to flat scheduling (every checkpoint "
        "body runs unconditionally); slice 4 will add an indirect-dispatch alternative.",
        cc);
    return;
  }

  auto &driver = CUDADriver::get_instance();

  static_assert(kCheckpointGateKernelFatbinSize > 0,
                "Checkpoint gate kernel fatbin is empty -- regenerate with "
                "scripts/build_checkpoint_gate_fatbin.py");

  uint32_t ret = driver.module_load_data.call(&gate_kernel_module_, kCheckpointGateKernelFatbin);
  QD_ERROR_IF(ret != CUDA_SUCCESS,
              "Failed to load qd.checkpoint gate kernel fatbin (CUDA error {}). This SM ({}) "
              "may not be included in the fatbin -- regenerate with "
              "scripts/build_checkpoint_gate_fatbin.py",
              ret, cc);

  driver.module_get_function(&gate_kernel_func_, gate_kernel_module_, "_qd_checkpoint_if_gate");
  QD_TRACE("Loaded qd.checkpoint IF-gate kernel from pre-built fatbin");
}

// Loads the yield-check kernel from its pre-built fatbin. Mirrors the gate-kernel loader: same
// SM 9.0+ requirement (it lives inside the IF body, which already needs conditional nodes),
// same lazy-load pattern, separate fatbin so kernels without `yield_on=` don't pay for it.
void GraphManager::ensure_checkpoint_yield_check_kernel_loaded() {
  if (yield_check_kernel_func_)
    return;

  int cc = CUDAContext::get_instance().get_compute_capability();
  if (cc < 90) {
    return;  // Already logged by ensure_checkpoint_gate_kernel_loaded(); silently skip.
  }

  auto &driver = CUDADriver::get_instance();

  static_assert(kCheckpointYieldCheckKernelFatbinSize > 0,
                "Checkpoint yield-check kernel fatbin is empty -- regenerate with "
                "scripts/build_checkpoint_yield_check_fatbin.py");

  uint32_t ret = driver.module_load_data.call(&yield_check_kernel_module_, kCheckpointYieldCheckKernelFatbin);
  QD_ERROR_IF(ret != CUDA_SUCCESS,
              "Failed to load qd.checkpoint yield-check kernel fatbin (CUDA error {}). This SM "
              "({}) may not be included in the fatbin -- regenerate with "
              "scripts/build_checkpoint_yield_check_fatbin.py",
              ret, cc);

  driver.module_get_function(&yield_check_kernel_func_, yield_check_kernel_module_, "_qd_checkpoint_yield_check");
  QD_TRACE("Loaded qd.checkpoint yield-check kernel from pre-built fatbin");
}

void *GraphManager::add_kernel_node(void *graph,
                                    void *prev_node,
                                    void *func,
                                    unsigned int grid_dim,
                                    unsigned int block_dim,
                                    unsigned int shared_mem,
                                    void **kernel_params) {
  // Opt-in to the requested dynamic shared memory size, just as
  // CUDAContext::launch does for the non-graph path.
  if (shared_mem > 0) {
    CUDADriver::get_instance().kernel_set_attribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_mem);
  }

  CudaKernelNodeParams params{};
  params.func = func;
  params.gridDimX = grid_dim;
  params.gridDimY = 1;
  params.gridDimZ = 1;
  params.blockDimX = block_dim;
  params.blockDimY = 1;
  params.blockDimZ = 1;
  params.sharedMemBytes = shared_mem;
  params.kernelParams = kernel_params;
  params.extra = nullptr;

  void *node = nullptr;
  CUDADriver::get_instance().graph_add_kernel_node(&node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
                                                   &params);
  return node;
}

void *GraphManager::add_conditional_while_node(void *graph, unsigned long long *cond_handle_out) {
  ensure_condition_kernel_loaded();
  QD_ASSERT(cond_kernel_func_);

  void *cu_ctx = CUDAContext::get_instance().get_context();

  CUDADriver::get_instance().graph_conditional_handle_create(cond_handle_out, graph, cu_ctx,
                                                             /*defaultLaunchValue=*/1,
                                                             /*flags=CU_GRAPH_COND_ASSIGN_DEFAULT=*/1);

  GraphNodeParams cond_node_params{};
  cond_node_params.type = 13;  // CU_GRAPH_NODE_TYPE_CONDITIONAL
  cond_node_params.handle = *cond_handle_out;
  cond_node_params.condType = 1;  // CU_GRAPH_COND_TYPE_WHILE
  cond_node_params.size = 1;
  cond_node_params.phGraph_out = nullptr;  // CUDA will populate this
  cond_node_params.ctx = cu_ctx;

  void *cond_node = nullptr;
  CUDADriver::get_instance().graph_add_node(&cond_node, graph, nullptr, 0, &cond_node_params);

  // CUDA replaces phGraph_out with a pointer to its owned array
  void **body_graphs = (void **)cond_node_params.phGraph_out;
  QD_ASSERT(body_graphs && body_graphs[0]);

  QD_TRACE("CUDA graph_do_while: conditional node created, body graph={}", body_graphs[0]);
  return body_graphs[0];
}

bool GraphManager::launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx, bool use_graph_do_while) {
  // TODO: these memcpy_host_to_device calls could be async (cuMemcpyHtoDAsync) on the launch
  // stream for better CPU-GPU overlap. All are tiny (<= 8 bytes), so synchronous is fine for
  // now.
  if (use_graph_do_while && cached.counter_ptr_slot) {
    void *flag_ptr = ctx.graph_do_while_flag_dev_ptr;
    CUDADriver::get_instance().memcpy_host_to_device(cached.counter_ptr_slot, &flag_ptr, sizeof(void *));
  }

  if (cached.resume_point_dev_ptr) {
    // Slice 1d: always reset resume_point to 0 before each launch so every checkpoint runs.
    // The on-yield kernel bumps this to INT_MAX during a launch; the reset here ensures the
    // next launch starts clean. Slice 2 will route this through the `GraphStatus` host API
    // and only reset when not resuming from a checkpoint.
    int32_t zero = 0;
    CUDADriver::get_instance().memcpy_host_to_device(cached.resume_point_dev_ptr, &zero, sizeof(int32_t));
  }

  if (cached.yield_signal_dev_ptr) {
    // Slice 1d: reset yield_signal to -1 before each launch so the yield-check kernel sees a
    // clean "no yield yet this launch" state. The first cp_id whose yield_on fires will CAS
    // its value in; later yields are no-ops thanks to atomicCAS semantics.
    int32_t neg_one = -1;
    CUDADriver::get_instance().memcpy_host_to_device(cached.yield_signal_dev_ptr, &neg_one, sizeof(int32_t));
  }

  // For each `qd.checkpoint(yield_on=foo)` with a resolved device pointer this launch, refresh
  // the persistent indirection slot. Re-runs are cheap and unconditional so a user can pass a
  // different ndarray each call without invalidating the cached graph (same trick as
  // `counter_ptr_slot` for graph_do_while).
  for (std::size_t cp = 0;
       cp < cached.checkpoint_yield_on_ptr_slots.size() && cp < ctx.checkpoint_yield_on_dev_ptrs.size(); ++cp) {
    void *slot = cached.checkpoint_yield_on_ptr_slots[cp];
    void *user_ptr = ctx.checkpoint_yield_on_dev_ptrs[cp];
    if (slot && user_ptr) {
      CUDADriver::get_instance().memcpy_host_to_device(slot, &user_ptr, sizeof(void *));
    }
  }

  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device(cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
                                                     cached.arg_buffer_size);
  }
  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);

  // Capture the post-launch yield_signal so introspection (and slice 2's GraphStatus) can see
  // which cp_id yielded. The sync is heavy-handed but matches the rest of the launch path,
  // which has been sync-by-default since `qd.kernel(graph=True)` shipped; slice 2 will move
  // it onto the launch stream once `GraphStatus` requires async semantics.
  last_yield_cp_id_on_last_call_ = -1;
  if (cached.yield_signal_dev_ptr) {
    CUDADriver::get_instance().stream_synchronize(stream);
    int32_t signal = -1;
    CUDADriver::get_instance().memcpy_device_to_host(&signal, cached.yield_signal_dev_ptr, sizeof(int32_t));
    last_yield_cp_id_on_last_call_ = signal;
  }

  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;
  num_checkpoints_on_last_call_ = cached.num_checkpoints;
  return true;
}

bool GraphManager::try_launch(int launch_id,
                              LaunchContextBuilder &ctx,
                              JITModule *cuda_module,
                              const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              LlvmRuntimeExecutor *executor) {
  if (offloaded_tasks.empty()) {
    return false;
  }

  const bool use_graph_do_while = ctx.graph_do_while_arg_id >= 0;

  QD_ERROR_IF(ctx.result_buffer_size > 0,
              "graph=True is not supported for kernels with struct return "
              "values; remove graph=True or avoid returning values");

  // Adstack-bearing kernels cannot go through the graph path. `ensure_adstack_heap` must run on the host
  // between the serial range_for-bounds kernel and the range_for kernel itself (the serial stores
  // `end_value` into `runtime->temporaries`, the host reads it back via DtoH and sizes the heap
  // accordingly); both kernels are baked into the graph so the host never gets a chance to run in between.
  // For graph-compatible, statically-bounded adstack kernels, codegen still sets
  // `static_num_threads = grid_dim * block_dim` and we could size the heap once at graph build, but that
  // path is not exercised today and the existing `grad_ptr != nullptr` guard below rejects the standard
  // autograd entry points that would hit it. Fail loudly instead of silently running with a nullptr
  // `runtime->adstack_heap_buffer`.
  for (const auto &task : offloaded_tasks) {
    QD_ERROR_IF(!task.ad_stack.allocas.empty(),
                "graph=True is not supported for kernels that use the reverse-mode autodiff stack "
                "(task '{}' has {} adstack allocas). Launch without graph=True.",
                task.name, task.ad_stack.allocas.size());
  }

  resolve_ctx_ndarray_ptrs(ctx, parameters, executor);

  auto it = cache_.find(launch_id);
  if (it != cache_.end()) {
    return launch_cached_graph(it->second, ctx, use_graph_do_while);
  }

  // Scan for qd.checkpoint() metadata once: any task with `checkpoint_id >= 0` opts the kernel
  // into the IF-conditional path. We need this before constructing CachedGraph so the latter
  // can allocate the `resume_point` and `yield_signal` scalars exactly when (and only when)
  // they will be referenced.
  bool has_checkpoints = false;
  std::size_t num_distinct_checkpoints = 0;
  int max_cp_id = -1;
  {
    int prev_cp = -1;
    for (const auto &task : offloaded_tasks) {
      if (task.checkpoint_id >= 0) {
        has_checkpoints = true;
        if (task.checkpoint_id != prev_cp) {
          ++num_distinct_checkpoints;
          prev_cp = task.checkpoint_id;
        }
        if (task.checkpoint_id > max_cp_id) {
          max_cp_id = task.checkpoint_id;
        }
      } else {
        prev_cp = -1;
      }
    }
  }

  // Determine which checkpoints actually have a resolved `yield_on=` ndarray this launch.
  // Only those need a per-cp persistent slot + a yield-check kernel inside the IF body. If
  // every cp has a null dev_ptr, treat this graph as yield-free even if the kernel declared
  // yield_on= for some cp (covers the case where the user passes the same kernel without a
  // ready ndarray -- though `set_args` should reject that earlier; this is defence in depth).
  bool has_yield = false;
  for (void *p : ctx.checkpoint_yield_on_dev_ptrs) {
    if (p) {
      has_yield = true;
      break;
    }
  }

  // Reject IF-conditional path pre-SM 9.0 -- match the graph_do_while pattern and let the
  // non-graph kernel launcher run the tasks flat (every checkpoint body runs unconditionally,
  // matching slice 1a's `with`-block semantics before slice 4 adds the indirect-dispatch
  // alternative). Falling back here keeps the user's code running while losing the IF
  // optimisation, instead of failing the launch.
  if (has_checkpoints) {
    ensure_checkpoint_gate_kernel_loaded();
    if (!gate_kernel_func_) {
      return false;
    }
    if (has_yield) {
      ensure_checkpoint_yield_check_kernel_loaded();
      if (!yield_check_kernel_func_) {
        return false;
      }
    }
  }

  CUDAContext::get_instance().make_current();

  CachedGraph cached(ctx.arg_buffer_size, ctx.result_buffer_size, use_graph_do_while, has_checkpoints, has_yield,
                     executor);
  cached.num_checkpoints = num_distinct_checkpoints;

  // Pre-size the per-checkpoint ptr-slot table to (max_cp_id + 1) so we can index by cp_id
  // directly. Slots for checkpoints without `yield_on=` stay `nullptr` and are skipped at
  // launch time. Only allocate when this graph actually has yield-bearing checkpoints --
  // otherwise the table stays empty and `launch_cached_graph`'s memcpy loop iterates zero
  // times.
  if (has_yield && max_cp_id >= 0) {
    cached.checkpoint_yield_on_ptr_slots.assign((std::size_t)max_cp_id + 1, nullptr);
    for (std::size_t cp = 0; cp < ctx.checkpoint_yield_on_dev_ptrs.size() && (int)cp <= max_cp_id; ++cp) {
      if (ctx.checkpoint_yield_on_dev_ptrs[cp]) {
        void *slot = nullptr;
        CUDADriver::get_instance().malloc(&slot, sizeof(void *));
        // Initial pointer write -- subsequent launches re-memcpy via launch_cached_graph.
        void *user_ptr = ctx.checkpoint_yield_on_dev_ptrs[cp];
        CUDADriver::get_instance().memcpy_host_to_device(slot, &user_ptr, sizeof(void *));
        cached.checkpoint_yield_on_ptr_slots[cp] = slot;
      }
    }
  }

  if (cached.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device(cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
                                                     cached.arg_buffer_size);
  }

  // --- Build CUDA graph ---
  void *graph = nullptr;
  CUDADriver::get_instance().graph_create(&graph, 0);

  // Target graph for kernel nodes. Without graph_do_while, work kernels go
  // directly into the top-level graph. With graph_do_while, they go into
  // a body graph inside a conditional while node. With qd.checkpoint() blocks, each
  // contiguous run of same-cp_id tasks is further wrapped in an IF conditional node
  // (gated by `resume_point`); tasks with cp_id == -1 stay siblings of the IF nodes.
  //
  //   Top-level graph
  //     └── (Conditional while node when use_graph_do_while, repeats while flag != 0)
  //           └── Body graph
  //                 ├── Non-checkpoint task A (cp_id = -1)
  //                 ├── Gate kernel for cp 0   (sets handle_0 from cp_id >= *resume_point)
  //                 ├── IF conditional node (handle_0)
  //                 │     └── Body: cp_id=0 tasks B, C
  //                 ├── Gate kernel for cp 1
  //                 ├── IF conditional node (handle_1)
  //                 │     └── Body: cp_id=1 task D
  //                 └── Condition kernel (only when use_graph_do_while; reads counter)
  //
  // The condition kernel must remain the last node in the WHILE body so it observes the
  // counter writes made inside this iteration's IF bodies.
  void *kernel_target_graph = graph;
  unsigned long long cond_handle = 0;

  if (use_graph_do_while) {
    ensure_condition_kernel_loaded();
    if (!cond_kernel_func_) {
      int cc = CUDAContext::get_instance().get_compute_capability();
      if (cc >= 90) {
        // SM 9.0+ should always be able to load the condition kernel.
        // Failing here means prerequisites are missing.
        QD_ERROR(
            "Condition kernel not available on SM {}; "
            "cannot build graph_do_while",
            cc);
      }
      // Pre-SM 9.0: fall back to host-side do-while loop.
      return false;
    }
    kernel_target_graph = add_conditional_while_node(graph, &cond_handle);
  }

  // Walk offloaded_tasks once, opening an IF node every time `checkpoint_id` transitions to a
  // non-negative value (and a different value than the previous task's). `prev_outer` tracks
  // the dependency chain in `kernel_target_graph`; `prev_inner` tracks the chain inside the
  // currently open IF body. We never re-enter a closed IF -- if the user writes
  // `checkpoint(); something_else; checkpoint()` with the same cp_id, that's two separate
  // contiguous runs and gets two IF nodes (which is what the user sees source-wise as well).
  void *prev_outer = nullptr;
  void *prev_inner = nullptr;
  void *current_body_graph = nullptr;
  int current_cp_id = -1;
  // cp_id_val is read by reference by the gate kernel at every IF body's start. cuGraphAddKernelNode
  // copies the *pointers* in `kernelParams`, not the values they point at, so we must keep each
  // cp_id alive for the lifetime of the graph build. Store them in this vector (stable address
  // because we reserve up front).
  std::vector<int32_t> cp_id_storage;
  cp_id_storage.reserve(num_distinct_checkpoints + 1);
  std::size_t total_nodes = 0;

  for (const auto &task : offloaded_tasks) {
    void *task_func = cuda_module->lookup_function(task.name);
    void *ctx_ptr = &cached.persistent_ctx;
    unsigned int grid = (unsigned int)task.grid_dim;
    unsigned int block = (unsigned int)task.block_dim;
    unsigned int smem = (unsigned int)task.dynamic_shared_array_bytes;

    if (task.checkpoint_id != current_cp_id) {
      // Close the previous IF body. Before we open the next one, if the previous checkpoint
      // had a `yield_on=` ptr slot, append a yield-check kernel as the last node inside its IF
      // body so it observes the body's writes to the user's yield_on ndarray. Sequencing is
      // handled by `prev_inner` -- the yield-check depends on the last body node.
      if (current_cp_id >= 0 && current_body_graph &&
          (std::size_t)current_cp_id < cached.checkpoint_yield_on_ptr_slots.size() &&
          cached.checkpoint_yield_on_ptr_slots[current_cp_id]) {
        void *yield_args[4] = {&cached.checkpoint_yield_on_ptr_slots[current_cp_id], &cp_id_storage.back(),
                               &cached.yield_signal_dev_ptr, &cached.resume_point_dev_ptr};
        prev_inner = add_kernel_node(current_body_graph, prev_inner, yield_check_kernel_func_,
                                     /*grid=*/1, /*block=*/1, /*smem=*/0, yield_args);
        ++total_nodes;
      }

      current_cp_id = task.checkpoint_id;
      if (current_cp_id >= 0) {
        // 1. Allocate a per-checkpoint conditional handle (default OFF so the body is skipped
        //    unless the gate explicitly enables it; the gate always runs and is the source of
        //    truth here -- the default only matters for slice 1d's yield path).
        cp_id_storage.push_back(current_cp_id);
        int32_t &cp_id_val = cp_id_storage.back();
        unsigned long long if_handle = 0;
        void *cu_ctx_local = CUDAContext::get_instance().get_context();
        CUDADriver::get_instance().graph_conditional_handle_create(&if_handle, kernel_target_graph, cu_ctx_local,
                                                                   /*defaultLaunchValue=*/0,
                                                                   /*flags=CU_GRAPH_COND_ASSIGN_DEFAULT=*/1);
        // 2. Gate kernel BEFORE the IF: writes (cp_id_val >= *resume_point) into the handle.
        void *gate_args[3] = {&if_handle, &cp_id_val, &cached.resume_point_dev_ptr};
        prev_outer = add_kernel_node(kernel_target_graph, prev_outer, gate_kernel_func_,
                                     /*grid=*/1, /*block=*/1, /*smem=*/0, gate_args);
        ++total_nodes;
        // 3. IF conditional node depending on the gate.
        GraphNodeParams cond_node_params{};
        cond_node_params.type = 13;     // CU_GRAPH_NODE_TYPE_CONDITIONAL
        cond_node_params.handle = if_handle;
        cond_node_params.condType = 0;  // CU_GRAPH_COND_TYPE_IF
        cond_node_params.size = 1;
        cond_node_params.phGraph_out = nullptr;
        cond_node_params.ctx = cu_ctx_local;
        void *if_node = nullptr;
        CUDADriver::get_instance().graph_add_node(&if_node, kernel_target_graph,
                                                  prev_outer ? &prev_outer : nullptr, prev_outer ? 1 : 0,
                                                  &cond_node_params);
        void **body_graphs = (void **)cond_node_params.phGraph_out;
        QD_ASSERT(body_graphs && body_graphs[0]);
        current_body_graph = body_graphs[0];
        prev_outer = if_node;
        prev_inner = nullptr;
        ++total_nodes;
      } else {
        current_body_graph = nullptr;
        prev_inner = nullptr;
      }
    }

    if (current_cp_id < 0) {
      // Non-checkpoint task: chain it directly into the outer (kernel_target_graph) chain.
      prev_outer = add_kernel_node(kernel_target_graph, prev_outer, task_func, grid, block, smem, &ctx_ptr);
    } else {
      // Checkpoint task: chain into the currently open IF body.
      prev_inner = add_kernel_node(current_body_graph, prev_inner, task_func, grid, block, smem, &ctx_ptr);
    }
    ++total_nodes;
  }

  // Flush the trailing checkpoint's yield-check kernel if the loop ended inside one (the
  // transition logic above only fires when cp_id *changes*, so a kernel whose last task is in
  // a checkpoint wouldn't otherwise get its yield-check appended).
  if (current_cp_id >= 0 && current_body_graph &&
      (std::size_t)current_cp_id < cached.checkpoint_yield_on_ptr_slots.size() &&
      cached.checkpoint_yield_on_ptr_slots[current_cp_id]) {
    void *yield_args[4] = {&cached.checkpoint_yield_on_ptr_slots[current_cp_id], &cp_id_storage.back(),
                           &cached.yield_signal_dev_ptr, &cached.resume_point_dev_ptr};
    add_kernel_node(current_body_graph, prev_inner, yield_check_kernel_func_,
                    /*grid=*/1, /*block=*/1, /*smem=*/0, yield_args);
    ++total_nodes;
  }

  if (use_graph_do_while) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);

    // Write the initial counter address into the persistent indirection slot
    // (allocated by the constructor). The condition kernel reads through this
    // slot, so swapping the counter ndarray later only requires updating it.
    void *flag_ptr = ctx.graph_do_while_flag_dev_ptr;
    CUDADriver::get_instance().memcpy_host_to_device(cached.counter_ptr_slot, &flag_ptr, sizeof(void *));

    if (has_yield) {
      // Use the cond-with-yield variant so a yield inside the WHILE body exits the loop
      // immediately. Without this the body would re-enter, see resume_point==INT_MAX in
      // every gate, skip every checkpoint, never decrement the counter, and spin forever.
      ensure_cond_with_yield_kernel_loaded();
      QD_ASSERT(cond_with_yield_kernel_func_);
      void *cond_args[3] = {&cond_handle, &cached.counter_ptr_slot, &cached.yield_signal_dev_ptr};
      add_kernel_node(kernel_target_graph, prev_outer, cond_with_yield_kernel_func_, 1, 1, 0, cond_args);
    } else {
      void *cond_args[2] = {&cond_handle, &cached.counter_ptr_slot};
      add_kernel_node(kernel_target_graph, prev_outer, cond_kernel_func_, 1, 1, 0, cond_args);
    }
    ++total_nodes;
  }

  // --- Instantiate and launch ---
  CUDADriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);

  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);

  CUDADriver::get_instance().graph_destroy(graph);

  cached.num_nodes = total_nodes;

  QD_TRACE("CUDA graph created with {} nodes ({} checkpoints) for launch_id={}{}", cached.num_nodes,
           cached.num_checkpoints, launch_id, use_graph_do_while ? " (with graph_do_while)" : "");

  num_nodes_on_last_call_ = cached.num_nodes;
  num_checkpoints_on_last_call_ = cached.num_checkpoints;
  ++total_builds_;
  cache_.emplace(launch_id, std::move(cached));
  used_on_last_call_ = true;
  return true;
}

}  // namespace cuda
}  // namespace quadrants::lang
