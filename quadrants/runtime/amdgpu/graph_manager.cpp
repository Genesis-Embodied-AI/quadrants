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
  if (persistent_device_arg_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    AMDGPUDriver::get_instance().mem_free(persistent_device_result_buffer);
  }
  if (device_runtime_ctx) {
    AMDGPUDriver::get_instance().mem_free(device_runtime_ctx);
  }
  for (auto &child : children) {
    if (child.persistent_device_arg_buffer) {
      AMDGPUDriver::get_instance().mem_free(child.persistent_device_arg_buffer);
    }
    if (child.device_runtime_ctx) {
      AMDGPUDriver::get_instance().mem_free(child.device_runtime_ctx);
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
      children(std::move(other.children)),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size),
      num_nodes(other.num_nodes) {
  // The moved-from object must not free our resources at destruction time. Rebind `kernel_args.extra_config` so it
  // points at *our* members after the move, not the moved-from object's (which is about to be destroyed). The child
  // `kernel_args.extra_config` self-pointers need no rebinding: `std::move` on the vector transfers the heap buffer
  // so each ChildGraphState keeps its address. `other.children` is left empty by the vector move, so the moved-from
  // destructor frees no child buffers (no double free).
  kernel_args.extra_config[1] = &kernel_args.packed_runtime_ctx_ptr;
  kernel_args.extra_config[3] = &kernel_args.pack_size;
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
  other.device_runtime_ctx = nullptr;
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
  // std::swap on the vectors exchanges their heap buffers, so each ChildGraphState keeps its address and the per-child
  // `kernel_args.extra_config` self-pointers stay valid without rebinding.
  std::swap(children, raii_guard.children);
  std::swap(arg_buffer_size, raii_guard.arg_buffer_size);
  std::swap(result_buffer_size, raii_guard.result_buffer_size);
  std::swap(num_nodes, raii_guard.num_nodes);
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

void *GraphManager::add_child_graph_node(void *graph, void *prev_node, void *child_graph) {
  void *node = nullptr;
  AMDGPUDriver::get_instance().graph_add_child_graph_node(&node, graph, prev_node ? &prev_node : nullptr,
                                                          prev_node ? 1 : 0, child_graph);
  return node;
}

void GraphManager::refresh_child_arg_buffer(const ChildLaunchInfo &info,
                                            CachedGraph::ChildGraphState &state,
                                            LlvmRuntimeExecutor *executor,
                                            void *stream) {
  // Resolve the child's ndarray handles to device pointers in its own (host-side) arg buffer, then upload to the
  // child's persistent device buffer. The child kernel nodes read this through `state.persistent_ctx.arg_buffer`
  // (baked into `state.device_runtime_ctx`). Re-run every launch so a realloc of any forwarded ndarray (new device
  // pointer) is reflected in the embedded subgraph without rebuilding it.
  resolve_ctx_ndarray_ptrs(*info.child_ctx, *info.child_parameters, executor);
  if (state.arg_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        state.persistent_device_arg_buffer, info.child_ctx->get_context().arg_buffer, state.arg_buffer_size, stream);
  }
}

void *GraphManager::build_child_subgraph(const ChildLaunchInfo &info, CachedGraph::ChildGraphState &state) {
  void *child_graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&child_graph, 0);
  void *prev_node = nullptr;
  for (const auto &task : *info.child_tasks) {
    QD_ERROR_IF(task.is_launch_child,
                "Nested qd.kernel subgraph calls more than one level deep are not yet supported "
                "(child task '{}' itself launches a child).",
                task.name);
    // Every child kernel node shares the child's own CachedKernelArgs (its device RuntimeContext pointer), just as the
    // parent's nodes share `cached.kernel_args`. See graph_manager.h for the per-child D1 rationale.
    prev_node = add_kernel_node(child_graph, prev_node, info.child_module->lookup_function(task.name),
                                (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
                                (unsigned int)task.dynamic_shared_array_bytes, state.kernel_args);
  }
  return child_graph;
}

bool GraphManager::launch_cached_graph(CachedGraph &cached,
                                       LaunchContextBuilder &ctx,
                                       const std::vector<ChildLaunchInfo> &child_infos,
                                       LlvmRuntimeExecutor *executor) {
  auto *stream = AMDGPUContext::get_instance().get_stream();
  if (ctx.arg_buffer_size > 0) {
    // Async HtoD on the launch stream: the subsequent `graph_launch` is queued on the same stream,
    // so the kernel nodes are ordered after the arg-buffer upload without a host-side barrier.
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer, cached.arg_buffer_size, stream);
  }
  // Refresh each embedded child's arg buffer (async on the same launch stream) so its ndarray data pointers track this
  // call's arguments. The graph topology (kernel / child-graph nodes) is unchanged; only the device-side arg bytes are
  // re-uploaded, ordered before `graph_launch` by same-stream queueing.
  for (std::size_t i = 0; i < child_infos.size() && i < cached.children.size(); i++) {
    if (child_infos[i].child_ctx != nullptr) {
      refresh_child_arg_buffer(child_infos[i], cached.children[i], executor, stream);
    }
  }
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);
  used_on_last_call_ = true;
  num_nodes_on_last_call_ = cached.num_nodes;
  return true;
}

bool GraphManager::try_launch(int launch_id,
                              LaunchContextBuilder &ctx,
                              JITModule *amdgpu_module,
                              const std::vector<std::pair<int, Callable::Parameter>> &parameters,
                              const std::vector<OffloadedTask> &offloaded_tasks,
                              LlvmRuntimeExecutor *executor,
                              const std::vector<ChildLaunchInfo> &child_infos) {
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
    return launch_cached_graph(it->second, ctx, child_infos, executor);
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

  // Allocate one persistent device arg buffer + device RuntimeContext + kernel-arg packing per embedded child (D1).
  // `children` is sized once here and never resized after, so each ChildGraphState's address (and thus its
  // `kernel_args.extra_config` self-pointers) stays stable for the cached graph_exec's lifetime (the vector's heap
  // buffer survives the CachedGraph move into `cache_`). Children share the parent's result buffer: graph kernels
  // never return values.
  if (!child_infos.empty()) {
    cached.children.resize(child_infos.size());
    for (std::size_t i = 0; i < child_infos.size(); i++) {
      const auto &info = child_infos[i];
      if (info.child_ctx == nullptr) {
        continue;
      }
      auto &state = cached.children[i];
      state.arg_buffer_size = info.child_ctx->arg_buffer_size;
      if (state.arg_buffer_size > 0) {
        AMDGPUDriver::get_instance().malloc(reinterpret_cast<void **>(&state.persistent_device_arg_buffer),
                                            state.arg_buffer_size);
      }
      AMDGPUDriver::get_instance().malloc(&state.device_runtime_ctx, sizeof(RuntimeContext));
      state.persistent_ctx.runtime = executor->get_llvm_runtime();
      state.persistent_ctx.arg_buffer = state.persistent_device_arg_buffer;
      state.persistent_ctx.result_buffer = reinterpret_cast<uint64 *>(cached.persistent_device_result_buffer);
      state.persistent_ctx.cpu_thread_id = 0;
      // Pack the child's device RuntimeContext pointer into its own extra-config (HIP `extra` byte-buffer convention),
      // mirroring the parent CachedGraph constructor.
      state.kernel_args.packed_runtime_ctx_ptr = state.device_runtime_ctx;
      state.kernel_args.pack_size = sizeof(void *);
      state.kernel_args.extra_config[0] = reinterpret_cast<void *>(0x01);  // HIP_LAUNCH_PARAM_BUFFER_POINTER
      state.kernel_args.extra_config[1] = &state.kernel_args.packed_runtime_ctx_ptr;
      state.kernel_args.extra_config[2] = reinterpret_cast<void *>(0x02);  // HIP_LAUNCH_PARAM_BUFFER_SIZE
      state.kernel_args.extra_config[3] = &state.kernel_args.pack_size;
      state.kernel_args.extra_config[4] = reinterpret_cast<void *>(0x03);  // HIP_LAUNCH_PARAM_END
      refresh_child_arg_buffer(info, state, executor, stream_for_setup);
      AMDGPUDriver::get_instance().memcpy_host_to_device_async(state.device_runtime_ctx, &state.persistent_ctx,
                                                              sizeof(RuntimeContext), stream_for_setup);
    }
  }
  AMDGPUDriver::get_instance().stream_synchronize(stream_for_setup);

  void *graph = nullptr;
  AMDGPUDriver::get_instance().graph_create(&graph, 0);

  // Each kernel node receives the device-side RuntimeContext pointer via the shared `cached.kernel_args` extra-config
  // (see graph_manager.h for why all nodes share one). Stream-parallel groups (`stream_parallel_group_id != 0`) are
  // silently serialized inside the graph, matching the CUDA implementation. `launch_child` tasks are embedded as
  // child-graph nodes instead of kernel nodes.
  std::vector<void *> child_graphs;  // owned standalone child graphs; destroyed after parent instantiation
  void *prev_node = nullptr;
  for (const auto &task : offloaded_tasks) {
    if (task.is_launch_child) {
      QD_ERROR_IF(task.child_call_index < 0 || (std::size_t)task.child_call_index >= child_infos.size() ||
                      child_infos[task.child_call_index].child_ctx == nullptr,
                  "launch_child task references child_call_index {} but no child launch info was provided",
                  task.child_call_index);
      const auto &info = child_infos[task.child_call_index];
      void *child_graph = build_child_subgraph(info, cached.children[task.child_call_index]);
      child_graphs.push_back(child_graph);
      prev_node = add_child_graph_node(graph, prev_node, child_graph);
      continue;
    }
    void *func = amdgpu_module->lookup_function(task.name);
    prev_node = add_kernel_node(graph, prev_node, func, (unsigned int)task.grid_dim, (unsigned int)task.block_dim,
                                (unsigned int)task.dynamic_shared_array_bytes, cached.kernel_args);
  }

  AMDGPUDriver::get_instance().graph_instantiate(&cached.graph_exec, graph, nullptr, nullptr, 0);

  auto *stream = AMDGPUContext::get_instance().get_stream();
  AMDGPUDriver::get_instance().graph_launch(cached.graph_exec, stream);

  AMDGPUDriver::get_instance().graph_destroy(graph);
  // The standalone child graphs were deep-copied into the parent graph by add_child_graph_node, and the parent graph
  // was deep-copied into graph_exec by graph_instantiate; the originals are no longer referenced.
  for (void *child_graph : child_graphs) {
    AMDGPUDriver::get_instance().graph_destroy(child_graph);
  }

  cached.num_nodes = offloaded_tasks.size();
  QD_TRACE("HIP graph created with {} kernel nodes for launch_id={}", cached.num_nodes, launch_id);

  num_nodes_on_last_call_ = cached.num_nodes;
  ++total_builds_;
  cache_.emplace(launch_id, std::move(cached));
  used_on_last_call_ = true;
  return true;
}

}  // namespace amdgpu
}  // namespace quadrants::lang
