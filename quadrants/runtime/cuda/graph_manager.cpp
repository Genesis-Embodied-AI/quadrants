#include "quadrants/runtime/cuda/graph_manager.h"
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
                         int num_graph_do_while_levels,
                         bool needs_resume_point_slot,
                         bool needs_yield_signal_slot,
                         LlvmRuntimeExecutor *executor)
    : arg_buffer_size(arg_buf_size), result_buffer_size(result_buf_size) {
  CUDADriver::get_instance().malloc((void **)&persistent_device_result_buffer,
                                    std::max(result_buffer_size, sizeof(uint64)));

  if (arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc((void **)&persistent_device_arg_buffer, arg_buffer_size);
  }

  if (num_graph_do_while_levels > 0) {
    counter_ptr_slots.resize(num_graph_do_while_levels, nullptr);
    for (auto &slot : counter_ptr_slots) {
      CUDADriver::get_instance().malloc(&slot, sizeof(void *));
    }
    // Persistent constant-1 int + a slot pointing at it, used to re-arm nested conditional handles.
    CUDADriver::get_instance().malloc(&const_one_dev, sizeof(int32_t));
    int32_t one = 1;
    CUDADriver::get_instance().memcpy_host_to_device(const_one_dev, &one, sizeof(int32_t));
    CUDADriver::get_instance().malloc(&const_one_slot, sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(const_one_slot, &const_one_dev, sizeof(void *));
  }

  if (needs_resume_point_slot) {
    // One int32 device scalar that the checkpoint gate kernels read every launch. Zero-init matches "no resume in
    // progress" -- every gate sees `cp_id >= 0` and enables its body.
    CUDADriver::get_instance().malloc(&resume_point_dev_ptr, sizeof(int32_t));
    int32_t zero = 0;
    CUDADriver::get_instance().memcpy_host_to_device(resume_point_dev_ptr, &zero, sizeof(int32_t));
  }

  if (needs_yield_signal_slot) {
    // Single int32 yield_signal device scalar. -1 means "no checkpoint has yielded this launch". The yield-check kernel
    // atomically CASes the first yielding cp_id in; `launch_cached_graph` resets it to -1 before every launch and
    // copies it back after.
    CUDADriver::get_instance().malloc(&yield_signal_dev_ptr, sizeof(int32_t));
    int32_t neg_one = -1;
    CUDADriver::get_instance().memcpy_host_to_device(yield_signal_dev_ptr, &neg_one, sizeof(int32_t));
  }

  persistent_ctx.runtime = executor->get_llvm_runtime();
  persistent_ctx.arg_buffer = persistent_device_arg_buffer;
  persistent_ctx.result_buffer = (uint64 *)persistent_device_result_buffer;
  persistent_ctx.cpu_thread_id = 0;
  // GPU-side checkpoint gating: publish the device-side resume_point / yield_signal pointers to the RuntimeContext
  // slots that the codegen-emitted body-kernel prologue reads. Null-pointers stay null for kernels without checkpoints
  // (the prologue is never emitted, so the read never happens; we still write nullptr defensively to keep the runtime
  // struct fully initialised). On SM 9.0+ the conditional gate prevents the body kernel from launching when skip is
  // needed, so the prologue is dead code in the common path; on pre-Hopper CUDA the prologue is the gating mechanism.
  persistent_ctx.checkpoint_resume_point_ptr = reinterpret_cast<int32_t *>(resume_point_dev_ptr);
  persistent_ctx.checkpoint_yield_signal_ptr = reinterpret_cast<int32_t *>(yield_signal_dev_ptr);
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
  for (void *slot : counter_ptr_slots) {
    if (slot) {
      CUDADriver::get_instance().mem_free(slot);
    }
  }
  if (const_one_dev) {
    CUDADriver::get_instance().mem_free(const_one_dev);
  }
  if (const_one_slot) {
    CUDADriver::get_instance().mem_free(const_one_slot);
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
      counter_ptr_slots(std::move(other.counter_ptr_slots)),
      const_one_dev(other.const_one_dev),
      const_one_slot(other.const_one_slot),
      resume_point_dev_ptr(other.resume_point_dev_ptr),
      yield_signal_dev_ptr(other.yield_signal_dev_ptr),
      checkpoint_yield_on_ptr_slots(std::move(other.checkpoint_yield_on_ptr_slots)),
      num_checkpoints(other.num_checkpoints),
      num_nodes(other.num_nodes) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.counter_ptr_slots.clear();
  other.const_one_dev = nullptr;
  other.const_one_slot = nullptr;
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
  std::swap(counter_ptr_slots, raii_guard.counter_ptr_slots);
  std::swap(const_one_dev, raii_guard.const_one_dev);
  std::swap(const_one_slot, raii_guard.const_one_slot);
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
  // Pre-size the per-checkpoint yield_on device-pointer table so the GraphManager can index it by cp_id even when some
  // checkpoints have no yield_on (their slot stays nullptr). Sized exactly once per launch from the matching arg-id
  // table populated in Python `Kernel.__call__`.
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
        // Resolve every graph_do_while level whose condition ndarray is this arg (multi-level table).
        ctx.resolve_graph_do_while_flag(arg_id, resolved_data);
        // Mirror the resolution for every `qd.checkpoint(yield_on=foo)` -- walk the per-cp_id arg-id table and stash
        // the device pointer where the GraphManager build path can find it. Same-arg shared between checkpoints is
        // fine; we just assign the same pointer to multiple slots.
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

// The three qd.checkpoint() fatbin loaders -- `ensure_cond_with_yield_kernel_loaded`,
// `ensure_checkpoint_gate_kernel_loaded`, `ensure_checkpoint_yield_check_kernel_loaded` -- are defined in
// `graph_manager_checkpoint.cpp`. They are declared in `graph_manager.h` and called from `try_launch` below.

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

unsigned long long GraphManager::create_cond_handle(void *graph) {
  void *cu_ctx = CUDAContext::get_instance().get_context();
  unsigned long long handle = 0;
  // The handle is created on the same graph the conditional node lives in (the root graph for a top-level loop, or a
  // parent body graph for a nested loop), matching the validated structure in tmp/nested_cond_validate.cu. The default
  // value (1) is only auto-applied at top-level graph launch; nested loops are re-armed each parent iteration by an
  // explicit init kernel (see build_level).
  CUDADriver::get_instance().graph_conditional_handle_create(&handle, graph, cu_ctx,
                                                             /*defaultLaunchValue=*/1,
                                                             /*flags=CU_GRAPH_COND_ASSIGN_DEFAULT=*/1);
  return handle;
}

void *GraphManager::add_conditional_while_node(void *graph,
                                               void *prev_node,
                                               unsigned long long handle,
                                               void **body_graph_out) {
  ensure_condition_kernel_loaded();
  QD_ASSERT(cond_kernel_func_);

  void *cu_ctx = CUDAContext::get_instance().get_context();

  GraphNodeParams cond_node_params{};
  cond_node_params.type = 13;  // CU_GRAPH_NODE_TYPE_CONDITIONAL
  cond_node_params.handle = handle;
  cond_node_params.condType = 1;  // CU_GRAPH_COND_TYPE_WHILE
  cond_node_params.size = 1;
  cond_node_params.phGraph_out = nullptr;  // CUDA will populate this
  cond_node_params.ctx = cu_ctx;

  void *cond_node = nullptr;
  CUDADriver::get_instance().graph_add_node(&cond_node, graph, prev_node ? &prev_node : nullptr, prev_node ? 1 : 0,
                                            &cond_node_params);

  // CUDA replaces phGraph_out with a pointer to its owned array
  void **body_graphs = (void **)cond_node_params.phGraph_out;
  QD_ASSERT(body_graphs && body_graphs[0]);

  *body_graph_out = body_graphs[0];
  QD_TRACE("CUDA graph_do_while: conditional node created, body graph={}", body_graphs[0]);
  return cond_node;
}

namespace {
// True if `level` is `ancestor` or a (transitive) descendant of it, walking parent pointers.
bool is_descendant_or_self(int level, int ancestor, const std::vector<GraphDoWhileLevel> &levels) {
  for (int c = level; c != -1; c = levels[c].parent_id) {
    if (c == ancestor) {
      return true;
    }
  }
  return false;
}

// Given a `descendant` of `parent_id`, return the direct child of `parent_id` on the path down to it.
int child_of(int parent_id, int descendant, const std::vector<GraphDoWhileLevel> &levels) {
  int c = descendant;
  while (levels[c].parent_id != parent_id) {
    c = levels[c].parent_id;
  }
  return c;
}
}  // namespace

void GraphManager::build_level(int parent_id,
                               void *target_graph,
                               int begin,
                               int end,
                               const std::vector<OffloadedTask> &tasks,
                               const std::vector<GraphDoWhileLevel> &levels,
                               std::vector<unsigned long long> &cond_handles,
                               JITModule *cuda_module,
                               CachedGraph &cached,
                               std::size_t &total_nodes) {
  // A yield-bearing kernel (has_yield) wires every loop level's condition kernel to the cond-with-yield variant, so a
  // yield raised inside any checkpoint exits this and every enclosing WHILE loop.
  const bool has_yield = (cached.yield_signal_dev_ptr != nullptr);
  void *prev_node = nullptr;
  int cursor = begin;
  while (cursor < end) {
    const int task_level = tasks[cursor].graph_do_while_level_id;
    if (task_level != parent_id) {
      // --- Start of a contiguous run belonging to a child level (or its descendants). ---
      const int child = child_of(parent_id, task_level, levels);
      int run_end = cursor;
      while (run_end < end && is_descendant_or_self(tasks[run_end].graph_do_while_level_id, child, levels)) {
        run_end++;
      }
      // Create the child's conditional handle up front: the re-arm init kernel below bakes the handle value into its
      // kernel params, so the handle must exist before that kernel node is added.
      cond_handles[child] = create_cond_handle(target_graph);
      // Re-arm the child handle at the start of each parent iteration. Only needed when this body itself re-executes
      // (parent_id != -1); at the kernel top level cudaGraphCondAssignDefault already sets the handle to 1 at each
      // graph launch. Reuses the condition kernel pointed at the constant-1 slot.
      if (parent_id != -1) {
        void *init_args[2] = {&cond_handles[child], &cached.const_one_slot};
        prev_node = add_kernel_node(target_graph, prev_node, cond_kernel_func_, 1, 1, 0, init_args);
        ++total_nodes;
      }
      // Conditional WHILE node for the child, depending on the init kernel (or the previous node).
      void *child_body = nullptr;
      void *cond_node = add_conditional_while_node(target_graph, prev_node, cond_handles[child], &child_body);
      ++total_nodes;
      build_level(child, child_body, cursor, run_end, tasks, levels, cond_handles, cuda_module, cached, total_nodes);
      // Subsequent siblings in this body depend on the conditional node.
      prev_node = cond_node;
      cursor = run_end;
      continue;
    }

    // --- A direct task of this level. Group consecutive tasks by checkpoint_id. ---
    const int cp = tasks[cursor].checkpoint_id;
    if (cp < 0) {
      // Non-checkpoint work kernel, chained directly into this level's graph.
      void *ctx_ptr = &cached.persistent_ctx;
      prev_node = add_kernel_node(target_graph, prev_node, cuda_module->lookup_function(tasks[cursor].name),
                                  (unsigned int)tasks[cursor].grid_dim, (unsigned int)tasks[cursor].block_dim,
                                  (unsigned int)tasks[cursor].dynamic_shared_array_bytes, &ctx_ptr);
      ++total_nodes;
      cursor++;
      continue;
    }

    // Checkpoint run: the maximal contiguous set of this level's direct tasks sharing cp_id == cp. (The try_launch
    // guard guarantees a checkpoint's tasks stay within one level, so the run is not interrupted by a nested loop.)
    int run_end = cursor;
    while (run_end < end && tasks[run_end].graph_do_while_level_id == parent_id && tasks[run_end].checkpoint_id == cp) {
      run_end++;
    }
    // Keep the cp_id alive for the graph's lifetime (gate / yield-check kernels read it by pointer); cp_id_storage_ is
    // reserved up front so push_back never reallocates.
    cp_id_storage_.push_back(cp);
    int32_t &cp_id_val = cp_id_storage_.back();
    const bool cp_has_yield = (std::size_t)cp < cached.checkpoint_yield_on_ptr_slots.size() &&
                              cached.checkpoint_yield_on_ptr_slots[cp] != nullptr;

    if (use_pre_hopper_flat_graph_) {
      // Pre-Hopper: no conditional node. Body kernels chain directly and self-gate via the codegen prologue; a trailing
      // yield-check kernel (if this cp has yield_on=) runs inline after the run.
      for (int t = cursor; t < run_end; t++) {
        void *ctx_ptr = &cached.persistent_ctx;
        prev_node = add_kernel_node(target_graph, prev_node, cuda_module->lookup_function(tasks[t].name),
                                    (unsigned int)tasks[t].grid_dim, (unsigned int)tasks[t].block_dim,
                                    (unsigned int)tasks[t].dynamic_shared_array_bytes, &ctx_ptr);
        ++total_nodes;
      }
      if (cp_has_yield) {
        void *yield_args[4] = {&cached.checkpoint_yield_on_ptr_slots[cp], &cp_id_val, &cached.yield_signal_dev_ptr,
                               &cached.resume_point_dev_ptr};
        prev_node = add_kernel_node(target_graph, prev_node, yield_check_kernel_func_, 1, 1, 0, yield_args);
        ++total_nodes;
      }
    } else {
      // SM 9.0+: gate kernel (sets the handle from cp_id >= *resume_point) -> IF conditional node -> body graph holding
      // the run's work kernels and an optional trailing yield-check.
      unsigned long long if_handle = 0;
      void *cu_ctx_local = CUDAContext::get_instance().get_context();
      CUDADriver::get_instance().graph_conditional_handle_create(&if_handle, target_graph, cu_ctx_local,
                                                                 /*defaultLaunchValue=*/0,
                                                                 /*flags=CU_GRAPH_COND_ASSIGN_DEFAULT=*/1);
      void *gate_args[3] = {&if_handle, &cp_id_val, &cached.resume_point_dev_ptr};
      prev_node = add_kernel_node(target_graph, prev_node, gate_kernel_func_, 1, 1, 0, gate_args);
      ++total_nodes;
      GraphNodeParams cond_node_params{};
      cond_node_params.type = 13;  // CU_GRAPH_NODE_TYPE_CONDITIONAL
      cond_node_params.handle = if_handle;
      cond_node_params.condType = 0;  // CU_GRAPH_COND_TYPE_IF
      cond_node_params.size = 1;
      cond_node_params.phGraph_out = nullptr;
      cond_node_params.ctx = cu_ctx_local;
      void *if_node = nullptr;
      CUDADriver::get_instance().graph_add_node(&if_node, target_graph, prev_node ? &prev_node : nullptr,
                                                prev_node ? 1 : 0, &cond_node_params);
      void **body_graphs = (void **)cond_node_params.phGraph_out;
      QD_ASSERT(body_graphs && body_graphs[0]);
      void *if_body = body_graphs[0];
      ++total_nodes;
      void *body_prev = nullptr;
      for (int t = cursor; t < run_end; t++) {
        void *ctx_ptr = &cached.persistent_ctx;
        body_prev = add_kernel_node(if_body, body_prev, cuda_module->lookup_function(tasks[t].name),
                                    (unsigned int)tasks[t].grid_dim, (unsigned int)tasks[t].block_dim,
                                    (unsigned int)tasks[t].dynamic_shared_array_bytes, &ctx_ptr);
        ++total_nodes;
      }
      if (cp_has_yield) {
        void *yield_args[4] = {&cached.checkpoint_yield_on_ptr_slots[cp], &cp_id_val, &cached.yield_signal_dev_ptr,
                               &cached.resume_point_dev_ptr};
        body_prev = add_kernel_node(if_body, body_prev, yield_check_kernel_func_, 1, 1, 0, yield_args);
        ++total_nodes;
      }
      prev_node = if_node;
    }
    cursor = run_end;
  }
  // For a real loop level, append its condition kernel last so it reads the flag after this iteration's work has
  // updated it. Use the cond-with-yield variant when the kernel has yielding checkpoints so a yield exits this (and
  // every enclosing) WHILE loop.
  if (parent_id >= 0) {
    if (has_yield) {
      void *cond_args[4] = {&cond_handles[parent_id], &cached.counter_ptr_slots[parent_id],
                            &cached.yield_signal_dev_ptr, &cached.resume_point_dev_ptr};
      add_kernel_node(target_graph, prev_node, cond_with_yield_kernel_func_, 1, 1, 0, cond_args);
    } else {
      void *cond_args[2] = {&cond_handles[parent_id], &cached.counter_ptr_slots[parent_id]};
      add_kernel_node(target_graph, prev_node, cond_kernel_func_, 1, 1, 0, cond_args);
    }
    ++total_nodes;
  }
}

bool GraphManager::launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx, bool use_graph_do_while) {
  // TODO: these memcpy_host_to_device calls could be async (cuMemcpyHtoDAsync) on the launch stream for better CPU-GPU
  // overlap. All are tiny (<= 8 bytes), so synchronous is fine for now.
  if (use_graph_do_while) {
    // Refresh every level's indirection slot with this launch's resolved condition ndarray pointer, so swapping any
    // level's counter ndarray between launches works without a rebuild.
    QD_ASSERT(cached.counter_ptr_slots.size() == ctx.graph_do_while_levels.size());
    for (size_t level = 0; level < ctx.graph_do_while_levels.size(); level++) {
      void *flag_ptr = ctx.graph_do_while_levels[level].flag_dev_ptr;
      CUDADriver::get_instance().memcpy_host_to_device(cached.counter_ptr_slots[level], &flag_ptr, sizeof(void *));
    }
  }

  if (cached.resume_point_dev_ptr) {
    // Slice 2: honour `ctx.resume_from_checkpoint`. -1 (the default) is the fresh-launch convention -- set the device
    // slot to 0 so every checkpoint runs. Any non-negative value is the cp_id supplied by `kernel.resume(...,
    // from_checkpoint=cp)` and gates skip every cp_id strictly below it. The yield-check kernel may bump this to
    // INT_MAX mid-launch; the reset here ensures the next launch starts from a clean baseline.
    int32_t rp = (ctx.resume_from_checkpoint < 0) ? 0 : ctx.resume_from_checkpoint;
    CUDADriver::get_instance().memcpy_host_to_device(cached.resume_point_dev_ptr, &rp, sizeof(int32_t));
  }

  if (cached.yield_signal_dev_ptr) {
    // Slice 1d: reset yield_signal to -1 before each launch so the yield-check kernel sees a clean "no yield yet this
    // launch" state. The first cp_id whose yield_on fires will CAS its value in; later yields are no-ops thanks to
    // atomicCAS semantics.
    int32_t neg_one = -1;
    CUDADriver::get_instance().memcpy_host_to_device(cached.yield_signal_dev_ptr, &neg_one, sizeof(int32_t));
  }

  // For each `qd.checkpoint(yield_on=foo)` with a resolved device pointer this launch, refresh the persistent
  // indirection slot. Re-runs are cheap and unconditional so a user can pass a different ndarray each call without
  // invalidating the cached graph (same trick as `counter_ptr_slots` for graph_do_while).
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

  // Capture the post-launch yield_signal so introspection (and slice 2's GraphStatus) can see which cp_id yielded. The
  // sync is heavy-handed but matches the rest of the launch path, which has been sync-by-default since
  // `qd.kernel(graph=True)` shipped; slice 2 will move it onto the launch stream once `GraphStatus` requires async
  // semantics.
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

  const bool use_graph_do_while = ctx.has_graph_do_while();

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

  // Up-front scan of qd.checkpoint() metadata: counts distinct cp_ids, decides whether any of them yield, picks the SM
  // 9.0+ IF-node path vs the pre-Hopper flat-graph fallback, and lazily loads the gate / yield-check fatbins. Defined
  // in `graph_manager_checkpoint.cpp`.
  CheckpointBuildPlan cp_plan = compute_checkpoint_plan_for_build(offloaded_tasks, ctx, use_graph_do_while);
  if (cp_plan.reject_graph_build) {
    return false;
  }

  CUDAContext::get_instance().make_current();

  CachedGraph cached(ctx.arg_buffer_size, ctx.result_buffer_size, (int)ctx.graph_do_while_levels.size(),
                     cp_plan.has_checkpoints, cp_plan.has_yield, executor);
  cached.num_checkpoints = cp_plan.num_distinct_checkpoints;

  allocate_checkpoint_yield_on_slots(cached, ctx, cp_plan);

  if (cached.arg_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_host_to_device(cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
                                                     cached.arg_buffer_size);
  }

  // --- Build CUDA graph ---
  //
  // Work kernels go directly into the top-level graph when the kernel has neither graph_do_while nor
  // checkpoints. Each graph_do_while loop level becomes a conditional WHILE node whose body graph holds
  // that level's direct work kernels, any nested conditional nodes, and finally that level's condition
  // kernel (last, so it reads the flag after this iteration's work). Within any level's body, a
  // contiguous run of same-cp_id tasks is wrapped in a gate-kernel + IF conditional node (SM 9.0+,
  // gated by `resume_point`); tasks with cp_id == -1 stay siblings of the IF nodes. A yielding
  // checkpoint appends a yield-check kernel that exits every enclosing WHILE loop. Combined example:
  //
  //   Top-level graph
  //     └── Conditional while node (outer, repeats while outer flag != 0)
  //           └── Outer body graph
  //                 ├── init kernel (re-arm inner handle = 1)
  //                 ├── Conditional while node (inner, repeats while inner flag != 0)
  //                 │     └── Inner body graph: work kernels + inner condition kernel
  //                 ├── Gate kernel for cp 0   (sets handle_0 from cp_id >= *resume_point)
  //                 ├── IF conditional node (handle_0)
  //                 │     └── Body: cp_id=0 tasks (+ yield-check if yield_on=)
  //                 └── Outer condition kernel (cond-with-yield when the kernel has yielding cps)
  //
  // The recursive builder (build_level) places direct/checkpoint/child/condition nodes from the
  // per-task level + checkpoint tags. The non-nested / no-checkpoint cases are special cases of it.
  void *graph = nullptr;
  CUDADriver::get_instance().graph_create(&graph, 0);

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
    if (cp_plan.has_yield) {
      // Use the cond-with-yield variant for each loop level's condition kernel so a yield raised inside any WHILE body
      // exits that loop immediately. Without it the body would re-enter, see resume_point==INT_MAX in every gate, skip
      // every checkpoint, never decrement the counter, and spin forever. The kernel also resets `resume_point` to 0 on
      // continue, so a `resume(from_checkpoint=cp)` call only skips checkpoints on its first iteration. Matches qipc's
      // `YieldResume::This/Next` semantics.
      ensure_cond_with_yield_kernel_loaded();
      QD_ASSERT(cond_with_yield_kernel_func_);
    }
    // Initialise each level's indirection slot with this launch's resolved flag pointer (refreshed on every relaunch in
    // launch_cached_graph).
    for (size_t level = 0; level < ctx.graph_do_while_levels.size(); level++) {
      QD_ASSERT(ctx.graph_do_while_levels[level].flag_dev_ptr);
      void *flag_ptr = ctx.graph_do_while_levels[level].flag_dev_ptr;
      CUDADriver::get_instance().memcpy_host_to_device(cached.counter_ptr_slots[level], &flag_ptr, sizeof(void *));
    }
  }

  // Recursively build the graph from the per-task level + checkpoint tags. build_level wraps each contiguous same-cp_id
  // run in a gate + IF conditional node (SM 9.0+) or a flat self-gating run (pre-Hopper), emits each graph_do_while
  // level's condition kernel last, and falls through to a plain chain of work kernels when the kernel has neither loops
  // nor checkpoints. cp_id_storage_ keeps each cp_id alive for the graph's lifetime; reserve up front so push_back
  // never reallocates.
  use_pre_hopper_flat_graph_ = cp_plan.use_pre_hopper_flat_graph;
  cp_id_storage_.clear();
  cp_id_storage_.reserve(cp_plan.num_distinct_checkpoints + 1);
  std::size_t total_nodes = 0;
  std::vector<unsigned long long> cond_handles(ctx.graph_do_while_levels.size(), 0);
  build_level(/*parent_id=*/-1, graph, 0, (int)offloaded_tasks.size(), offloaded_tasks, ctx.graph_do_while_levels,
              cond_handles, cuda_module, cached, total_nodes);

  // --- Instantiate ---
  CUDADriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);

  CUDADriver::get_instance().graph_destroy(graph);

  cached.num_nodes = total_nodes;

  QD_TRACE("CUDA graph created with {} nodes ({} checkpoints) for launch_id={}{}", cached.num_nodes,
           cached.num_checkpoints, launch_id, use_graph_do_while ? " (with graph_do_while)" : "");

  ++total_builds_;
  auto [cache_it, _inserted] = cache_.emplace(launch_id, std::move(cached));
  // First launch goes through the same code path as every subsequent launch so the yield_signal read-back (and any
  // other per-launch bookkeeping) doesn't have to be duplicated. Slightly redundant memcpys for arg buffer / counter
  // slot, but those happen once per graph build and are well under a microsecond each.
  return launch_cached_graph(cache_it->second, ctx, use_graph_do_while);
}

}  // namespace cuda
}  // namespace quadrants::lang
