#include "quadrants/runtime/cuda/kernel_launcher.h"
#include "quadrants/rhi/cuda/cuda_context.h"

#include <cstring>
#include <filesystem>

namespace quadrants::lang {
namespace cuda {

// PTX for a tiny condition kernel that reads a device-side int32 flag and
// calls cudaGraphSetConditional(handle, flag != 0 ? 1 : 0).
// Compiled from CUDA C with: nvcc -ptx -arch=sm_90 -rdc=true
// Requires JIT linking with libcudadevrt.a at runtime.
static const char *kConditionKernelPTX = R"PTX(
.version 8.8
.target sm_90
.address_size 64
.extern .func cudaGraphSetConditional
(
    .param .b64 cudaGraphSetConditional_param_0,
    .param .b32 cudaGraphSetConditional_param_1
)
;
.visible .entry _qd_graph_while_cond(
    .param .u64 _qd_graph_while_cond_param_0,
    .param .u64 _qd_graph_while_cond_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [_qd_graph_while_cond_param_0];
    ld.param.u64 %rd2, [_qd_graph_while_cond_param_1];
    cvta.to.global.u64 %rd3, %rd2;
    ld.global.u32 %r1, [%rd3];
    setp.ne.s32 %p1, %r1, 0;
    selp.u32 %r2, 1, 0, %p1;
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    .param .b32 param1;
    st.param.b32 [param1+0], %r2;
    call.uni cudaGraphSetConditional, (param0, param1);
    } // callseq 0
    ret;
}
)PTX";

CachedCudaGraph::~CachedCudaGraph() {
  if (graph_exec) {
    CUDADriver::get_instance().graph_exec_destroy(graph_exec);
  }
  if (persistent_device_arg_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_arg_buffer);
  }
  if (persistent_device_result_buffer) {
    CUDADriver::get_instance().mem_free(persistent_device_result_buffer);
  }
}

CachedCudaGraph::CachedCudaGraph(CachedCudaGraph &&other) noexcept
    : graph_exec(other.graph_exec),
      persistent_device_arg_buffer(other.persistent_device_arg_buffer),
      persistent_device_result_buffer(other.persistent_device_result_buffer),
      persistent_ctx(other.persistent_ctx),
      arg_buffer_size(other.arg_buffer_size),
      result_buffer_size(other.result_buffer_size) {
  other.graph_exec = nullptr;
  other.persistent_device_arg_buffer = nullptr;
  other.persistent_device_result_buffer = nullptr;
}

CachedCudaGraph &CachedCudaGraph::operator=(CachedCudaGraph &&other) noexcept {
  if (this != &other) {
    if (graph_exec)
      CUDADriver::get_instance().graph_exec_destroy(graph_exec);
    if (persistent_device_arg_buffer)
      CUDADriver::get_instance().mem_free(persistent_device_arg_buffer);
    if (persistent_device_result_buffer)
      CUDADriver::get_instance().mem_free(persistent_device_result_buffer);

    graph_exec = other.graph_exec;
    persistent_device_arg_buffer = other.persistent_device_arg_buffer;
    persistent_device_result_buffer = other.persistent_device_result_buffer;
    persistent_ctx = other.persistent_ctx;
    arg_buffer_size = other.arg_buffer_size;
    result_buffer_size = other.result_buffer_size;

    other.graph_exec = nullptr;
    other.persistent_device_arg_buffer = nullptr;
    other.persistent_device_result_buffer = nullptr;
  }
  return *this;
}

bool KernelLauncher::on_cuda_device(void *ptr) {
  unsigned int attr_val = 0;
  uint32_t ret_code = CUDADriver::get_instance().mem_get_attribute.call(
      &attr_val, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (void *)ptr);

  return ret_code == CUDA_SUCCESS && attr_val == CU_MEMORYTYPE_DEVICE;
}

bool KernelLauncher::resolve_ctx_ndarray_ptrs(
    LaunchContextBuilder &ctx,
    const std::vector<std::pair<int, Callable::Parameter>> &parameters) {
  auto *executor = get_runtime_executor();
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

      void *resolved_data = nullptr;
      void *resolved_grad = nullptr;

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (!on_cuda_device(data_ptr)) {
          return false;
        }
        resolved_data = data_ptr;
        resolved_grad = grad_ptr;
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        resolved_data = executor->get_device_alloc_info_ptr(*ptr);
        if (grad_ptr) {
          resolved_grad = executor->get_device_alloc_info_ptr(
              *static_cast<DeviceAllocation *>(grad_ptr));
        }
      }

      if (resolved_data) {
        ctx.set_ndarray_ptrs(arg_id, (uint64)resolved_data,
                             (uint64)resolved_grad);
        if (arg_id == ctx.graph_while_arg_id) {
          ctx.graph_while_flag_dev_ptr = resolved_data;
        }
      }
    }
  }
  return true;
}

void KernelLauncher::ensure_condition_kernel_loaded() {
  if (cond_kernel_func_)
    return;

  auto &driver = CUDADriver::get_instance();

  // Find libcudadevrt.a — required for cudaGraphSetConditional in device code
  std::string cudadevrt_path;
  for (const auto &candidate : {
           std::string("/usr/local/cuda/lib64/libcudadevrt.a"),
           std::string("/usr/lib/x86_64-linux-gnu/libcudadevrt.a"),
       }) {
    if (std::filesystem::exists(candidate)) {
      cudadevrt_path = candidate;
      break;
    }
  }
  if (cudadevrt_path.empty()) {
    QD_WARN("Cannot find libcudadevrt.a — graph_while will not work");
    return;
  }

  void *link_state = nullptr;
  driver.link_create(0, nullptr, nullptr, &link_state);

  std::size_t ptx_len = std::strlen(kConditionKernelPTX) + 1;
  driver.link_add_data(link_state, /*CU_JIT_INPUT_PTX=*/1,
                       const_cast<char *>(kConditionKernelPTX), ptx_len,
                       "qd_cond", 0, nullptr, nullptr);

  driver.link_add_file(link_state, /*CU_JIT_INPUT_LIBRARY=*/4,
                       cudadevrt_path.c_str(), 0, nullptr, nullptr);

  void *cubin = nullptr;
  std::size_t cubin_size = 0;
  driver.link_complete(link_state, &cubin, &cubin_size);

  driver.module_load_data(&cond_kernel_module_, cubin);
  driver.module_get_function(&cond_kernel_func_, cond_kernel_module_,
                             "_qd_graph_while_cond");
  driver.link_destroy(link_state);

  QD_TRACE("Loaded graph_while condition kernel ({} bytes cubin)", cubin_size);
}

bool KernelLauncher::launch_llvm_kernel_graph(Handle handle,
                                              LaunchContextBuilder &ctx) {
  int launch_id = handle.get_launch_id();

  auto &launcher_ctx = contexts_[launch_id];
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  if (offloaded_tasks.size() < 2 && ctx.graph_while_arg_id < 0) {
    return false;
  }

  if (!resolve_ctx_ndarray_ptrs(ctx, parameters)) {
    return false;
  }

  const bool use_graph_while = ctx.graph_while_arg_id >= 0;

  auto it = cuda_graph_cache_.find(launch_id);
  if (it != cuda_graph_cache_.end()) {
    auto &cached = it->second;
    if (ctx.arg_buffer_size > 0) {
      CUDADriver::get_instance().memcpy_host_to_device(
          cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
          cached.arg_buffer_size);
    }
    auto *stream = CUDAContext::get_instance().get_stream();
    CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);
    return true;
  }

  CUDAContext::get_instance().make_current();

  auto *executor = get_runtime_executor();
  auto *cuda_module = launcher_ctx.jit_module;

  CachedCudaGraph cached;

  // --- Allocate persistent buffers ---
  cached.result_buffer_size = std::max(ctx.result_buffer_size, sizeof(uint64));
  CUDADriver::get_instance().malloc(
      (void **)&cached.persistent_device_result_buffer,
      cached.result_buffer_size);

  cached.arg_buffer_size = ctx.arg_buffer_size;
  if (cached.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc(
        (void **)&cached.persistent_device_arg_buffer, cached.arg_buffer_size);
    CUDADriver::get_instance().memcpy_host_to_device(
        cached.persistent_device_arg_buffer, ctx.get_context().arg_buffer,
        cached.arg_buffer_size);
  }

  // --- Build persistent RuntimeContext ---
  cached.persistent_ctx.runtime = executor->get_llvm_runtime();
  cached.persistent_ctx.arg_buffer = cached.persistent_device_arg_buffer;
  cached.persistent_ctx.result_buffer =
      (uint64 *)cached.persistent_device_result_buffer;
  cached.persistent_ctx.cpu_thread_id = 0;

  // --- Build CUDA graph ---
  void *graph = nullptr;
  CUDADriver::get_instance().graph_create(&graph, 0);

  // Determine the target graph for kernel nodes.
  // With graph_while, kernels go into the conditional while body graph.
  void *kernel_target_graph = graph;
  unsigned long long cond_handle = 0;

  if (use_graph_while) {
    ensure_condition_kernel_loaded();
    if (!cond_kernel_func_) {
      QD_WARN("Condition kernel not available, falling back to non-graph");
      CUDADriver::get_instance().graph_destroy(graph);
      return false;
    }

    void *cu_ctx = CUDAContext::get_instance().get_context();

    CUDADriver::get_instance().graph_conditional_handle_create(
        &cond_handle, graph, cu_ctx,
        /*defaultLaunchValue=*/1,
        /*flags=CU_GRAPH_COND_ASSIGN_DEFAULT=*/1);

    CudaGraphNodeParams cond_node_params{};
    cond_node_params.type = 13;  // CU_GRAPH_NODE_TYPE_CONDITIONAL
    cond_node_params.handle = cond_handle;
    cond_node_params.condType = 1;  // CU_GRAPH_COND_TYPE_WHILE
    cond_node_params.size = 1;
    cond_node_params.phGraph_out = nullptr;  // CUDA will populate this
    cond_node_params.ctx = cu_ctx;

    void *cond_node = nullptr;
    CUDADriver::get_instance().graph_add_node(&cond_node, graph, nullptr, 0,
                                              &cond_node_params);

    // CUDA replaces phGraph_out with a pointer to its owned array
    void **body_graphs = (void **)cond_node_params.phGraph_out;
    QD_ASSERT(body_graphs && body_graphs[0]);
    kernel_target_graph = body_graphs[0];

    QD_TRACE("CUDA graph_while: conditional node created, body graph={}",
             kernel_target_graph);
  }

  // Add work kernel nodes to the target graph
  void *prev_node = nullptr;
  for (const auto &task : offloaded_tasks) {
    void *func = cuda_module->lookup_function(task.name);

    void *ctx_ptr = &cached.persistent_ctx;
    CudaKernelNodeParams node_params{};
    node_params.func = func;
    node_params.gridDimX = (unsigned int)task.grid_dim;
    node_params.gridDimY = 1;
    node_params.gridDimZ = 1;
    node_params.blockDimX = (unsigned int)task.block_dim;
    node_params.blockDimY = 1;
    node_params.blockDimZ = 1;
    node_params.sharedMemBytes = (unsigned int)task.dynamic_shared_array_bytes;
    node_params.kernelParams = &ctx_ptr;
    node_params.extra = nullptr;

    void *node = nullptr;
    const void *deps = prev_node;
    std::size_t num_deps = prev_node ? 1 : 0;
    CUDADriver::get_instance().graph_add_kernel_node(
        &node, kernel_target_graph, prev_node ? &deps : nullptr, num_deps,
        &node_params);
    prev_node = node;
  }

  // For graph_while: add condition kernel as the last node in the body graph
  if (use_graph_while) {
    QD_ASSERT(ctx.graph_while_flag_dev_ptr);

    void *flag_ptr = ctx.graph_while_flag_dev_ptr;
    void *cond_args[2] = {&cond_handle, &flag_ptr};

    CudaKernelNodeParams cond_kp{};
    cond_kp.func = cond_kernel_func_;
    cond_kp.gridDimX = 1;
    cond_kp.gridDimY = 1;
    cond_kp.gridDimZ = 1;
    cond_kp.blockDimX = 1;
    cond_kp.blockDimY = 1;
    cond_kp.blockDimZ = 1;
    cond_kp.sharedMemBytes = 0;
    cond_kp.kernelParams = cond_args;
    cond_kp.extra = nullptr;

    void *cond_kernel_node = nullptr;
    CUDADriver::get_instance().graph_add_kernel_node(
        &cond_kernel_node, kernel_target_graph,
        prev_node ? &prev_node : nullptr, prev_node ? 1 : 0, &cond_kp);
  }

  // --- Instantiate and launch ---
  CUDADriver::get_instance().graph_instantiate(&cached.graph_exec, graph,
                                               nullptr, nullptr, 0);

  auto *stream = CUDAContext::get_instance().get_stream();
  CUDADriver::get_instance().graph_launch(cached.graph_exec, stream);

  CUDADriver::get_instance().graph_destroy(graph);

  QD_TRACE(
      "CUDA graph created with {} kernel nodes for launch_id={}"
      "{}",
      offloaded_tasks.size(), launch_id,
      use_graph_while ? " (with graph_while)" : "");

  cuda_graph_cache_.emplace(launch_id, std::move(cached));
  return true;
}

void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());

  if (ctx.use_cuda_graph) {
    if (launch_llvm_kernel_graph(handle, ctx)) {
      return;
    }
  }

  auto launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *cuda_module = launcher_ctx.jit_module;
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  CUDAContext::get_instance().make_current();

  std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>,
                     ArgArrayPtrKeyHasher>
      transfers;

  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> device_ptrs;

  char *device_result_buffer{nullptr};
  CUDADriver::get_instance().malloc_async(
      (void **)&device_result_buffer,
      std::max(ctx.result_buffer_size, sizeof(uint64)), nullptr);
  ctx.get_context().runtime = executor->get_llvm_runtime();

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0) {
        continue;
      }

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];
      auto grad_ptr = ctx.array_ptrs[grad_ptr_idx];

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (on_cuda_device(data_ptr)) {
          device_ptrs[data_ptr_idx] = data_ptr;
          device_ptrs[grad_ptr_idx] = grad_ptr;
        } else {
          DeviceAllocation devalloc = executor->allocate_memory_on_device(
              arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          CUDADriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
          if (grad_ptr != nullptr) {
            DeviceAllocation grad_devalloc =
                executor->allocate_memory_on_device(
                    arr_sz, (uint64 *)device_result_buffer);
            device_ptrs[grad_ptr_idx] =
                executor->get_device_alloc_info_ptr(grad_devalloc);
            transfers[grad_ptr_idx] = {grad_ptr, grad_devalloc};

            CUDADriver::get_instance().memcpy_host_to_device(
                (void *)device_ptrs[grad_ptr_idx], grad_ptr, arr_sz);
          } else {
            device_ptrs[grad_ptr_idx] = nullptr;
          }
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_while_arg_id) {
          ctx.graph_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      } else if (arr_sz > 0) {
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        if (grad_ptr != nullptr) {
          ptr = static_cast<DeviceAllocation *>(grad_ptr);
          device_ptrs[grad_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);
        } else {
          device_ptrs[grad_ptr_idx] = nullptr;
        }

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)device_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_while_arg_id) {
          ctx.graph_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      }
    }
  }
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
  }
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  char *device_arg_buffer = nullptr;
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().malloc_async((void **)&device_arg_buffer,
                                            ctx.arg_buffer_size, nullptr);
    CUDADriver::get_instance().memcpy_host_to_device_async(
        device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size,
        nullptr);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  do {
    for (auto task : offloaded_tasks) {
      QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
               task.block_dim);
      cuda_module->launch(task.name, task.grid_dim, task.block_dim, 0,
                          {&ctx.get_context()}, {});
    }
    if (ctx.graph_while_arg_id >= 0 && ctx.graph_while_flag_dev_ptr) {
      int32_t counter_val = 0;
      auto *stream = CUDAContext::get_instance().get_stream();
      CUDADriver::get_instance().stream_synchronize(stream);
      CUDADriver::get_instance().memcpy_device_to_host(
          &counter_val, ctx.graph_while_flag_dev_ptr, sizeof(int32_t));
      if (counter_val == 0)
        break;
    }
  } while (ctx.graph_while_arg_id >= 0);
  if (ctx.arg_buffer_size > 0) {
    CUDADriver::get_instance().mem_free_async(device_arg_buffer, nullptr);
  }
  if (ctx.result_buffer_size > 0) {
    CUDADriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
  }
  CUDADriver::get_instance().mem_free_async(device_result_buffer, nullptr);
  if (transfers.size() > 0) {
    CUDADriver::get_instance().stream_synchronize(nullptr);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      CUDADriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)device_ptrs[idx],
          ctx.array_runtime_sizes[idx.arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(compiled.arch() == Arch::cuda);

  if (!compiled.get_handle()) {
    auto handle = make_handle();
    auto index = handle.get_launch_id();
    contexts_.resize(index + 1);

    auto &ctx = contexts_[index];
    auto *executor = get_runtime_executor();

    auto data = compiled.get_internal_data().compiled_data.clone();
    auto *jit_module = executor->create_jit_module(std::move(data.module));

    // Populate ctx
    ctx.jit_module = jit_module;
    ctx.parameters = &compiled.get_internal_data().args;
    ctx.offloaded_tasks = std::move(data.tasks);

    compiled.set_handle(handle);
  }
  return *compiled.get_handle();
}

}  // namespace cuda
}  // namespace quadrants::lang
