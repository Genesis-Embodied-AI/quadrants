#include "quadrants/runtime/amdgpu/graph_manager.h"
#include "quadrants/runtime/amdgpu/amdgpu_utils.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"

#include <algorithm>
#include <vector>

namespace quadrants::lang {
namespace amdgpu {

CachedGraph::CachedGraph(std::size_t arg_buf_size, std::size_t result_buf_size, LlvmRuntimeExecutor *executor)
    : arg_buffer_size(arg_buf_size), result_buffer_size(result_buf_size) {
  AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_result_buffer),
                                      std::max(result_buffer_size, sizeof(uint64)));

  if (arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&persistent_device_arg_buffer), arg_buffer_size);
  }

  // Device-side `RuntimeContext` for graph kernel-node args. Unlike the CUDA path which passes a host pointer (relying
  // on UVA / HMM), AMDGPU kernels dereference the pointer directly on the GPU, so it must point at device memory. See
  // `runtime/amdgpu/graph_manager.h` for the full rationale.
  AMDGPUDriver::get_instance().malloc(&device_runtime_ctx, sizeof(RuntimeContext));

  persistent_ctx.runtime = executor->get_llvm_runtime();
  persistent_ctx.arg_buffer = persistent_device_arg_buffer;
  persistent_ctx.result_buffer = reinterpret_cast<uint64 *>(persistent_device_result_buffer);
  persistent_ctx.cpu_thread_id = 0;

  // The single kernel arg every offloaded task takes is a RuntimeContext *. Stage the device pointer value into the
  // persistent CachedKernelArgs so the extra-config addresses are stable for the graph's lifetime.
  kernel_args.packed_runtime_ctx_ptr = device_runtime_ctx;
  kernel_args.pack_size = sizeof(void *);
  kernel_args.extra_config[0] = reinterpret_cast<void *>(0x01);  // HIP_LAUNCH_PARAM_BUFFER_POINTER
  kernel_args.extra_config[1] = &kernel_args.packed_runtime_ctx_ptr;
  kernel_args.extra_config[2] = reinterpret_cast<void *>(0x02);  // HIP_LAUNCH_PARAM_BUFFER_SIZE
  kernel_args.extra_config[3] = &kernel_args.pack_size;
  kernel_args.extra_config[4] = reinterpret_cast<void *>(0x03);  // HIP_LAUNCH_PARAM_END
}

CachedGraph::~CachedGraph() {
  if (graph_exec) {
    AMDGPUDriver::get_instance().graph_exec_destroy(graph_exec);
  }
  // Slice 4: release every per-batch sub-graph exec. The destructor runs once per cache slot, so this also covers
  // the move-from path (sub_graph_execs is cleared on the moved-from instance below).
  for (void *exec : sub_graph_execs) {
    if (exec) {
      AMDGPUDriver::get_instance().graph_exec_destroy(exec);
    }
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
}

CachedGraph::CachedGraph(CachedGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      device_runtime_ctx(other.device_runtime_ctx),
      kernel_args(other.kernel_args),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      num_nodes(other.num_nodes),
      sub_graph_execs(std::move(other.sub_graph_execs)),
      batch_cp_ids(std::move(other.batch_cp_ids)) {
  // The moved-from object must not free our resources at destruction time. Rebind `kernel_args.extra_config` so it
  // points at *our* members after the move, not the moved-from object's (which is about to be destroyed).
  kernel_args.extra_config[1] = &kernel_args.packed_runtime_ctx_ptr;
  kernel_args.extra_config[3] = &kernel_args.pack_size;
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.device_runtime_ctx = nullptr;
  // `std::move`'d vectors are left empty by the moves above, so `other.~CachedGraph()` will not try to release
  // exec handles that now belong to us. No explicit clear needed but kept symmetrical with the destructor invariant.
}

CachedGraph &CachedGraph::operator=(CachedGraph &&other) noexcept {
  // Move-and-swap: after the swaps, `raii_guard` holds our old resources and its destructor frees them, so every owned
  // pointer is released uniformly.
  CachedGraph raii_guard(std::move(other));
  std::swap(graph_exec, raii_guard.graph_exec);
  std::swap(persistent_device_arg_buffer, raii_guard.persistent_device_arg_buffer);
  std::swap(persistent_device_result_buffer, raii_guard.persistent_device_result_buffer);
  std::swap(persistent_ctx, raii_guard.persistent_ctx);
  std::swap(device_runtime_ctx, raii_guard.device_runtime_ctx);
  std::swap(kernel_args, raii_guard.kernel_args);
  // After the swap, kernel_args is `raii_guard`'s old kernel_args (which had extra_config[1,3] bound to its own
  // members). Rebind to ours so the addresses stay valid through the swap.
  kernel_args.extra_config[1] = &kernel_args.packed_runtime_ctx_ptr;
  kernel_args.extra_config[3] = &kernel_args.pack_size;
  raii_guard.kernel_args.extra_config[1] = &raii_guard.kernel_args.packed_runtime_ctx_ptr;
  raii_guard.kernel_args.extra_config[3] = &raii_guard.kernel_args.pack_size;
  std::swap(arg_buffer_size, raii_guard.arg_buffer_size);
  std::swap(result_buffer_size, raii_guard.result_buffer_size);
  std::swap(num_nodes, raii_guard.num_nodes);
  std::swap(sub_graph_execs, raii_guard.sub_graph_execs);
  std::swap(batch_cp_ids, raii_guard.batch_cp_ids);
  return *this;
}

// Resolves ndarray parameter handles in the launch context to raw device pointers, writing them into the arg buffer
// via `set_ndarray_ptrs`.
//
// Unlike the normal launch path, this does not handle host-resident arrays (no temporary device allocation or
// host-to-device transfer). Errors if any external array is on the host, since graph mode bakes device pointers into
// the cached graph.
void GraphManager::resolve_ctx_ndarray_ptrs(LaunchContextBuilder &ctx,
                                            const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                                            LlvmRuntimeExecutor *executor) {
  // Slice 4: every launch starts with a fresh `checkpoint_yield_on_dev_ptrs` shaped to match the per-cp arg-id
  // table the frontend filled in. Sized by-cp_id (not per-yielding-cp) so non-yielding checkpoints get a
  // `nullptr` slot rather than collapsing the index space. Mirrors the matching line in the CUDA
  // `resolve_ctx_ndarray_ptrs`.
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
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = resolved_data;
        }
        // Slice 4: route this ndarray's device pointer into the per-cp yield-flag table for every checkpoint
        // that named it as its `yield_on=`. One arg may serve multiple checkpoints, so this is a scan rather
        // than a single assignment.
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

bool GraphManager::launch_cached_graph(CachedGraph &cached, LaunchContextBuilder &ctx) {
  // Slice 4: dispatch to the per-batch path when this entry was built from a checkpoint kernel. Detected by
  // `sub_graph_execs` being non-empty; the plain path's `graph_exec` stays `nullptr` in that layout.
  if (!cached.sub_graph_execs.empty()) {
    return launch_cached_checkpoint_graph(cached, ctx);
  }
  // Reset the slice-4 yield bookkeeping. The previous launch may have been a checkpoint kernel; for a plain
  // launch the question "did anything yield?" has no meaning, so report -1 like the CUDA path does for
  // non-graph aux launches.
  last_yield_cp_id_on_last_call_ = -1;
  auto *stream = AMDGPUContext::get_instance().get_stream();
  if (ctx.arg_buffer_size > 0) {
    // Async HtoD on the launch stream: the subsequent `graph_launch` is queued on the same stream,
    // so the kernel nodes are ordered after the arg-buffer upload without a host-side barrier.
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer, cached.arg_buffer_size, stream);
  }
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);
  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;
  return true;
}

bool GraphManager::launch_cached_checkpoint_graph(CachedGraph &cached, LaunchContextBuilder &ctx) {
  // Mirrors the CPU slice-6 launcher's host-branch gating loop, but with HIP sub-graph launches and a per-batch
  // D2H of the yield flag instead of single-task launches. The contract on `ctx.resume_from_checkpoint` and
  // `ctx.checkpoint_yield_on_dev_ptrs` matches what `Kernel.__call__` / `kernel.resume(...)` plumbs through
  // for the CUDA-native path.
  auto *stream = AMDGPUContext::get_instance().get_stream();
  if (ctx.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer, cached.arg_buffer_size, stream);
  }
  // resume_point governs the first-iteration skip when the caller used `kernel.resume(from_checkpoint=N)`.
  // yield_signal mirrors the device-side scalar from the CUDA path: -1 == "no yield yet", otherwise == cp_id
  // of the first yielding checkpoint observed this launch (first-yielder-wins matches the atomic-CAS semantics
  // CUDA gets for free).
  int32_t resume_point = (ctx.resume_from_checkpoint < 0) ? 0 : ctx.resume_from_checkpoint;
  int32_t yield_signal = -1;
  last_yield_cp_id_on_last_call_ = -1;
  num_nodes_on_last_call_ = cached.num_nodes;
  used_on_last_call_ = true;
  // Local zero-int constant used to memcpy-clear a device-side yield_on flag. Declared once instead of pushing
  // a fresh stack slot per yielding-checkpoint hit; the call site below uses `&zero` as the host source.
  int32_t zero = 0;
  for (std::size_t i = 0; i < cached.sub_graph_execs.size(); ++i) {
    int32_t cp_id = cached.batch_cp_ids[i];
    if (cp_id >= 0) {
      if (cp_id < resume_point) {
        continue;
      }
      if (yield_signal != -1) {
        continue;
      }
    }
    AMDGPUDriver::get_instance().graph_launch(cached.sub_graph_execs[i], stream);
    // Mirror the CUDA yield-check kernel: after a yielding checkpoint's body, read the user's `yield_on` flag
    // and, if non-zero, set our local `yield_signal` + record `last_yield_cp_id_on_last_call_` + clear the
    // user's flag back to 0 so the next launch starts clean. The D2H is a hard sync because HIP graphs do not
    // expose a post-node "ship a value to host" primitive (no `hipGraphAddHostNode` in the device flag path;
    // pinned host memory would let us avoid the sync but is out of scope for slice 4 per reentrant.md §10).
    if (cp_id >= 0 && (std::size_t)cp_id < ctx.checkpoint_yield_on_dev_ptrs.size() &&
        ctx.checkpoint_yield_on_dev_ptrs[cp_id] != nullptr) {
      AMDGPUDriver::get_instance().stream_synchronize(stream);
      int32_t flag_val = 0;
      AMDGPUDriver::get_instance().memcpy_device_to_host(&flag_val, ctx.checkpoint_yield_on_dev_ptrs[cp_id],
                                                         sizeof(int32_t));
      if (flag_val != 0) {
        yield_signal = cp_id;
        last_yield_cp_id_on_last_call_ = cp_id;
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(ctx.checkpoint_yield_on_dev_ptrs[cp_id], &zero,
                                                                 sizeof(int32_t), stream);
      }
    }
  }
  return true;
}

void *GraphManager::build_subgraph_exec_for_tasks(const std::vector<OffloadedTask> &tasks,
                                                  std::size_t begin,
                                                  std::size_t end,
                                                  JITModule *amdgpu_module,
                                                  CachedKernelArgs &kernel_args) {
  // Build a fresh HIP graph carrying just the tasks in `[begin, end)`, instantiate it into a launch-ready exec,
  // and destroy the staging graph (the exec keeps its own copy). Used at slice-4 build time; the resulting exec
  // is stored in `cached.sub_graph_execs[i]` and freed by the destructor.
  void *graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&graph, 0);
  void *prev_node = nullptr;
  for (std::size_t i = begin; i < end; ++i) {
    const auto &task = tasks[i];
    void *func = amdgpu_module->lookup_function(task.name);
    prev_node = add_kernel_node(graph, prev_node, func, (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
                                (unsigned int)task.dynamic_shared_array_bytes, kernel_args);
  }
  void *exec = nullptr;
  AMDGPUDriver::get_instance().graph_instantiate(&exec, graph, nullptr, nullptr, 0);
  AMDGPUDriver::get_instance().graph_destroy(graph);
  return exec;
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
  // `runtime/cuda/graph_manager.cpp::try_launch` for the full rationale: the per-task adstack
  // setup runs host-side between the serial range_for-bounds kernel and the range_for kernel
  // itself, and there is no host hook between graph nodes.
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

  AMDGPUContext::get_instance().make_current();
  CachedGraph cached(ctx.arg_buffer_size, ctx.result_buffer_size, executor);

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

  // Slice 4: detect whether this kernel carries any `qd.checkpoint(...)` blocks. If yes, build one HIP graph per
  // contiguous run of same-`cp_id` tasks so the launcher can skip individual checkpoints based on `resume_point`
  // and the post-launch yield_signal D2H. If no, take the original single-graph path unchanged.
  bool has_checkpoint = false;
  for (const auto &task : offloaded_tasks) {
    if (task.checkpoint_id >= 0) {
      has_checkpoint = true;
      break;
    }
  }

  auto *stream = AMDGPUContext::get_instance().get_stream();
  if (has_checkpoint) {
    // Walk tasks in source order and start a new batch whenever `checkpoint_id` changes. cp_id < 0 batches (work
    // outside any `qd.checkpoint`) always launch; cp_id >= 0 batches gate on `resume_point` / `yield_signal`
    // inside `launch_cached_checkpoint_graph`.
    std::size_t i = 0;
    while (i < offloaded_tasks.size()) {
      int32_t batch_cp = offloaded_tasks[i].checkpoint_id;
      std::size_t j = i + 1;
      while (j < offloaded_tasks.size() && offloaded_tasks[j].checkpoint_id == batch_cp) {
        ++j;
      }
      void *exec = build_subgraph_exec_for_tasks(offloaded_tasks, i, j, amdgpu_module, cached.kernel_args);
      cached.sub_graph_execs.push_back(exec);
      cached.batch_cp_ids.push_back(batch_cp);
      i = j;
    }
    cached.num_nodes = offloaded_tasks.size();
    QD_TRACE("HIP sub-graphs created: {} batches over {} tasks for launch_id={}", cached.sub_graph_execs.size(),
             cached.num_nodes, launch_id);
  } else {
    void *graph = nullptr;
    AMDGPUDriver::get_instance().graph_create(&graph, 0);
    // Each kernel node receives the device-side RuntimeContext pointer via the shared `cached.kernel_args`
    // extra-config (see graph_manager.h for why all nodes share one). Stream-parallel groups
    // (`stream_parallel_group_id != 0`) are silently serialized inside the graph, matching the CUDA implementation.
    void *prev_node = nullptr;
    for (const auto &task : offloaded_tasks) {
      void *func = amdgpu_module->lookup_function(task.name);
      prev_node = add_kernel_node(graph, prev_node, func, (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
                                  (unsigned int)task.dynamic_shared_array_bytes, cached.kernel_args);
    }
    AMDGPUDriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);
    AMDGPUDriver::get_instance().graph_destroy(graph);
    cached.num_nodes = offloaded_tasks.size();
    QD_TRACE("HIP graph created with {} kernel nodes for launch_id={}", cached.num_nodes, launch_id);
  }

  ++total_builds_;
  auto [it_inserted, ok] = cache_.emplace(launch_id, std::move(cached));
  // Funnel through `launch_cached_graph` so the post-launch bookkeeping (`used_on_last_call_`,
  // `num_nodes_on_last_call_`, slice-4 yield path) is identical for the initial build and every subsequent cache
  // hit. Without this the first launch after a build would skip the yield D2H in the checkpoint case.
  return launch_cached_graph(it_inserted->second, ctx);
}

}  // namespace amdgpu
}  // namespace quadrants::lang
