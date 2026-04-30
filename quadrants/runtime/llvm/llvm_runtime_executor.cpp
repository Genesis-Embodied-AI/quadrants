#include "quadrants/runtime/llvm/llvm_runtime_executor.h"
#include "quadrants/program/adstack_size_expr_eval.h"

#include <cstring>
#include <limits>
#include <vector>

#include "quadrants/ir/static_adstack_bound_reducer_device.h"
#include "quadrants/ir/stmt_op_types.h"

#include "quadrants/rhi/common/host_memory_pool.h"
#include "quadrants/runtime/llvm/llvm_offline_cache.h"
#include "quadrants/rhi/cpu/cpu_device.h"
#include "quadrants/rhi/cuda/cuda_device.h"
#include "quadrants/platform/cuda/detect_cuda.h"
#include "quadrants/rhi/cuda/cuda_driver.h"
#include "quadrants/rhi/llvm/device_memory_pool.h"
#include "quadrants/program/program_impl.h"

#if defined(QD_WITH_CUDA)
#include "quadrants/rhi/cuda/cuda_context.h"
#endif

#include "quadrants/platform/amdgpu/detect_amdgpu.h"
#include "quadrants/rhi/amdgpu/amdgpu_driver.h"
#include "quadrants/rhi/amdgpu/amdgpu_device.h"
#if defined(QD_WITH_AMDGPU)
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#endif

namespace quadrants::lang {
namespace {
void assert_failed_host(const char *msg) {
  QD_ERROR("Assertion failure: {}", msg);
}

void *host_allocate_aligned(HostMemoryPool *memory_pool, std::size_t size, std::size_t alignment) {
  return memory_pool->allocate(size, alignment);
}

}  // namespace

LlvmRuntimeExecutor::LlvmRuntimeExecutor(CompileConfig &config, KernelProfilerBase *profiler, ProgramImpl *program_impl)
    : config_(config), program_impl_(program_impl) {
  if (config.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    if (!is_cuda_api_available()) {
      QD_WARN("No CUDA driver API detected.");
      config.arch = host_arch();
    } else if (!CUDAContext::get_instance().detected()) {
      QD_WARN("No CUDA device detected.");
      config.arch = host_arch();
    } else {
      // CUDA runtime created successfully
      use_device_memory_pool_ = CUDAContext::get_instance().supports_mem_pool();
    }
#else
    QD_WARN("Quadrants is not compiled with CUDA.");
    config.arch = host_arch();
#endif

    if (config.arch != Arch::cuda) {
      QD_WARN("Falling back to {}.", arch_name(host_arch()));
    }
  } else if (config.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    if (!is_rocm_api_available()) {
      QD_WARN("No AMDGPU ROCm API detected.");
      config.arch = host_arch();
    } else if (!AMDGPUContext::get_instance().detected()) {
      QD_WARN("No AMDGPU device detected.");
      config.arch = host_arch();
    } else {
      // AMDGPU runtime created successfully
      use_device_memory_pool_ = AMDGPUContext::get_instance().supports_mem_pool();
    }
#else
    QD_WARN("Quadrants is not compiled with AMDGPU.");
    config.arch = host_arch();
#endif
  }

  if (config.kernel_profiler) {
    profiler_ = profiler;
  }

  snode_tree_buffer_manager_ = std::make_unique<SNodeTreeBufferManager>(this);
  thread_pool_ = std::make_unique<ThreadPool>(config.cpu_max_num_threads);

  llvm_runtime_ = nullptr;

  if (arch_is_cpu(config.arch)) {
    config.max_block_dim = 1024;
    device_ = std::make_shared<cpu::CpuDevice>();

  }
#if defined(QD_WITH_CUDA)
  else if (config.arch == Arch::cuda) {
    int num_SMs{1};
    CUDADriver::get_instance().device_get_attribute(&num_SMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, nullptr);
    int query_max_block_dim{1024};
    CUDADriver::get_instance().device_get_attribute(&query_max_block_dim, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, nullptr);
    int version{0};
    CUDADriver::get_instance().driver_get_version(&version);
    int query_max_block_per_sm{16};
    if (version >= 11000) {
      // query this attribute only when CUDA version is above 11.0
      CUDADriver::get_instance().device_get_attribute(&query_max_block_per_sm,
                                                      CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, nullptr);
    }

    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }

    if (config.saturating_grid_dim == 0) {
      if (version >= 11000) {
        QD_TRACE("CUDA max blocks per SM = {}", query_max_block_per_sm);
      }
      config.saturating_grid_dim = num_SMs * query_max_block_per_sm * 2;
    }
    if (config.kernel_profiler) {
      CUDAContext::get_instance().set_profiler(profiler);
    } else {
      CUDAContext::get_instance().set_profiler(nullptr);
    }
    CUDAContext::get_instance().set_debug(config.debug);
    if (config.cuda_stack_limit != 0) {
      CUDADriver::get_instance().context_set_limit(CU_LIMIT_STACK_SIZE, config.cuda_stack_limit);
    }
    device_ = std::make_shared<cuda::CudaDevice>();
  }
#endif
#if defined(QD_WITH_AMDGPU)
  else if (config.arch == Arch::amdgpu) {
    int num_workgroups{1};
    AMDGPUDriver::get_instance().device_get_attribute(&num_workgroups, HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    int query_max_block_dim{1024};
    AMDGPUDriver::get_instance().device_get_attribute(&query_max_block_dim, HIP_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, 0);
    // magic number 32
    // I didn't find the relevant parameter to limit the max block num per CU
    // So ....
    int query_max_block_per_cu{32};
    if (config.max_block_dim == 0) {
      config.max_block_dim = query_max_block_dim;
    }
    if (config.saturating_grid_dim == 0) {
      config.saturating_grid_dim = num_workgroups * query_max_block_per_cu * 2;
    }
    if (config.kernel_profiler) {
      AMDGPUContext::get_instance().set_profiler(profiler);
    } else {
      AMDGPUContext::get_instance().set_profiler(nullptr);
    }
    AMDGPUContext::get_instance().set_debug(config.debug);
    device_ = std::make_shared<amdgpu::AmdgpuDevice>();
  }
#endif
  else {
    QD_NOT_IMPLEMENTED
  }
  llvm_context_ = std::make_unique<QuadrantsLLVMContext>(config_, arch_is_cpu(config.arch) ? host_arch() : config.arch);
  jit_session_ = JITSession::create(llvm_context_.get(), config, config.arch, program_impl_);
  init_runtime_jit_module(llvm_context_->clone_runtime_module());
}

QuadrantsLLVMContext *LlvmRuntimeExecutor::get_llvm_context() {
  return llvm_context_.get();
}

JITModule *LlvmRuntimeExecutor::create_jit_module(std::unique_ptr<llvm::Module> module) {
  return jit_session_->add_module(std::move(module));
}

JITModule *LlvmRuntimeExecutor::get_runtime_jit_module() {
  return runtime_jit_module_;
}

void LlvmRuntimeExecutor::print_list_manager_info(void *list_manager, uint64 *result_buffer) {
  auto list_manager_len = runtime_query<int32>("ListManager_get_num_elements", result_buffer, list_manager);

  auto element_size = runtime_query<int32>("ListManager_get_element_size", result_buffer, list_manager);

  auto elements_per_chunk =
      runtime_query<int32>("ListManager_get_max_num_elements_per_chunk", result_buffer, list_manager);

  auto num_active_chunks = runtime_query<int32>("ListManager_get_num_active_chunks", result_buffer, list_manager);

  auto size_MB = 1e-6f * num_active_chunks * elements_per_chunk * element_size;

  fmt::print(" length={:n}     {:n} chunks x [{:n} x {:n} B]  total={:.4f} MB\n", list_manager_len, num_active_chunks,
             elements_per_chunk, element_size, size_MB);
}

void LlvmRuntimeExecutor::synchronize() {
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().stream_synchronize(nullptr);
#else
    QD_ERROR("No CUDA support");
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
#else
    QD_ERROR("No AMDGPU support");
#endif
  }
  fflush(stdout);
}

uint64 LlvmRuntimeExecutor::fetch_result_uint64(int i, uint64 *result_buffer) {
  // TODO: We are likely doing more synchronization than necessary. Simplify the
  // sync logic when we fetch the result.
  synchronize();
  uint64 ret;
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i, sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_device_to_host(&ret, result_buffer + i, sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    ret = result_buffer[i];
  }
  return ret;
}

std::size_t LlvmRuntimeExecutor::get_snode_num_dynamically_allocated(SNode *snode, uint64 *result_buffer) {
  QD_ASSERT(arch_uses_llvm(config_.arch));

  auto node_allocator =
      runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer, llvm_runtime_, snode->id);
  auto data_list = runtime_query<void *>("NodeManager_get_data_list", result_buffer, node_allocator);

  return (std::size_t)runtime_query<int32>("ListManager_get_num_elements", result_buffer, data_list);
}

void LlvmRuntimeExecutor::check_adstack_overflow() {
  // Called from `synchronize()` on every sync so adstack overflow surfaces as a Python exception regardless of
  // `compile_config.debug`. The runtime / result buffer may not exist yet (e.g. a C++ test that constructs Program
  // without materializing the runtime and then triggers Program::finalize -> synchronize), so no-op in that case.
  if (llvm_runtime_ == nullptr || result_buffer_cache_ == nullptr) {
    return;
  }
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_adstack_overflow", llvm_runtime_);
  auto flag = fetch_result<int64>(quadrants_result_buffer_error_id, result_buffer_cache_);
  if (flag != 0) {
    throw QuadrantsAssertionError(
        "Adstack overflow: a reverse-mode autodiff kernel pushed more elements than the adstack capacity "
        "allows. Raised at the next qd.sync() rather than at the offending kernel launch. The pre-pass "
        "resolved this alloca to a bound tighter than the actual runtime push count - either the enclosing "
        "loop shape is outside the current `SizeExpr` grammar (rewrite it, or extend the grammar), or the "
        "Bellman-Ford analyzer undercounted the forward-pass accumulation on this stack (file a bug with "
        "the kernel IR via `QD_DUMP_IR=1`).");
  }
}

void LlvmRuntimeExecutor::check_runtime_error(uint64 *result_buffer) {
  synchronize();
  auto *runtime_jit_module = get_runtime_jit_module();
  runtime_jit_module->call<void *>("runtime_retrieve_and_reset_error_code", llvm_runtime_);
  auto error_code = fetch_result<int64>(quadrants_result_buffer_error_id, result_buffer);

  if (error_code) {
    std::string error_message_template;

    // Here we fetch the error_message_template char by char.
    // This is not efficient, but fortunately we only need to do this when an
    // assertion fails. Note that we may not have unified memory here, so using
    // "fetch_result" that works across device/host memory is necessary.
    for (int i = 0;; i++) {
      runtime_jit_module->call<void *>("runtime_retrieve_error_message", llvm_runtime_, i);
      auto c = fetch_result<char>(quadrants_result_buffer_error_id, result_buffer);
      error_message_template += c;
      if (c == '\0') {
        break;
      }
    }

    if (error_code == 1) {
      const auto error_message_formatted =
          format_error_message(error_message_template, [runtime_jit_module, result_buffer, this](int argument_id) {
            runtime_jit_module->call<void *>("runtime_retrieve_error_message_argument", llvm_runtime_, argument_id);
            return fetch_result<uint64>(quadrants_result_buffer_error_id, result_buffer);
          });
      throw QuadrantsAssertionError(error_message_formatted);
    } else {
      QD_NOT_IMPLEMENTED
    }
  }
}

void LlvmRuntimeExecutor::print_memory_profiler_info(std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
                                                     uint64 *result_buffer) {
  QD_ASSERT(arch_uses_llvm(config_.arch));

  fmt::print("\n[Memory Profiler]\n");

  std::locale::global(std::locale("en_US.UTF-8"));
  // So that thousand separators are added to "{:n}" slots in fmtlib.
  // E.g., 10000 is printed as "10,000".
  // TODO: is there a way to set locale only locally in this function?

  std::function<void(SNode *, int)> visit = [&](SNode *snode, int depth) {
    auto element_list = runtime_query<void *>("LLVMRuntime_get_element_lists", result_buffer, llvm_runtime_, snode->id);

    if (snode->type != SNodeType::place) {
      fmt::print("SNode {:10}\n", snode->get_node_type_name_hinted());

      if (element_list) {
        fmt::print("  active element list:");
        print_list_manager_info(element_list, result_buffer);

        auto node_allocator =
            runtime_query<void *>("LLVMRuntime_get_node_allocators", result_buffer, llvm_runtime_, snode->id);

        if (node_allocator) {
          auto free_list = runtime_query<void *>("NodeManager_get_free_list", result_buffer, node_allocator);
          auto recycled_list = runtime_query<void *>("NodeManager_get_recycled_list", result_buffer, node_allocator);

          auto free_list_len = runtime_query<int32>("ListManager_get_num_elements", result_buffer, free_list);

          auto recycled_list_len = runtime_query<int32>("ListManager_get_num_elements", result_buffer, recycled_list);

          auto free_list_used = runtime_query<int32>("NodeManager_get_free_list_used", result_buffer, node_allocator);

          auto data_list = runtime_query<void *>("NodeManager_get_data_list", result_buffer, node_allocator);
          fmt::print("  data list:          ");
          print_list_manager_info(data_list, result_buffer);

          fmt::print(
              "  Allocated elements={:n}; free list length={:n}; recycled list "
              "length={:n}\n",
              free_list_used, free_list_len, recycled_list_len);
        }
      }
    }
    for (const auto &ch : snode->ch) {
      visit(ch.get(), depth + 1);
    }
  };

  for (auto &a : snode_trees_) {
    visit(a->root(), /*depth=*/0);
  }

  auto total_requested_memory =
      runtime_query<std::size_t>("LLVMRuntime_get_total_requested_memory", result_buffer, llvm_runtime_);

  fmt::print("Total requested dynamic memory (excluding alignment padding): {:n} B\n", total_requested_memory);
}

DevicePtr LlvmRuntimeExecutor::get_snode_tree_device_ptr(int tree_id) {
  DeviceAllocation tree_alloc = snode_tree_allocs_[tree_id];
  return tree_alloc.get_ptr();
}

void LlvmRuntimeExecutor::initialize_llvm_runtime_snodes(const LlvmOfflineCache::FieldCacheData &field_cache_data,
                                                         uint64 *result_buffer) {
  auto *const runtime_jit = get_runtime_jit_module();
  // By the time this creator is called, "this" is already destroyed.
  // Therefore it is necessary to capture members by values.
  size_t root_size = field_cache_data.root_size;
  const auto snode_metas = field_cache_data.snode_metas;
  const int tree_id = field_cache_data.tree_id;
  const int root_id = field_cache_data.root_id;

  bool all_dense = config_.demote_dense_struct_fors;
  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (snode_metas[i].type != SNodeType::dense && snode_metas[i].type != SNodeType::place &&
        snode_metas[i].type != SNodeType::root) {
      all_dense = false;
      break;
    }
  }

  if ((config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) && use_device_memory_pool() && !all_dense) {
    // Sparse SNode trees allocate runtime state via runtime_memory_allocate_aligned during snode_initialize.
    // When the device memory pool is active, the eager preallocate_runtime_memory() in materialize_runtime is
    // skipped, so the bump allocator is only wired up lazily here when a sparse tree actually needs it.
    preallocate_runtime_memory();
  }

  QD_TRACE("Allocating data structure of size {} bytes", root_size);
  std::size_t rounded_size = quadrants::iroundup(root_size, quadrants_page_size);

  Ptr root_buffer = snode_tree_buffer_manager_->allocate(rounded_size, tree_id, result_buffer);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset(root_buffer, 0, rounded_size);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    std::memset(root_buffer, 0, rounded_size);
  }

  DeviceAllocation alloc = llvm_device()->import_memory(root_buffer, rounded_size);

  snode_tree_allocs_[tree_id] = alloc;

  runtime_jit->call<void *, std::size_t, int, int, int, std::size_t, Ptr>(
      "runtime_initialize_snodes", llvm_runtime_, root_size, root_id, (int)snode_metas.size(), tree_id, rounded_size,
      root_buffer, all_dense);

  for (size_t i = 0; i < snode_metas.size(); i++) {
    if (is_gc_able(snode_metas[i].type)) {
      const auto snode_id = snode_metas[i].id;
      std::size_t node_size;
      auto element_size = snode_metas[i].cell_size_bytes;
      if (snode_metas[i].type == SNodeType::pointer) {
        // pointer. Allocators are for single elements
        node_size = element_size;
      } else {
        // dynamic. Allocators are for the chunks
        node_size = sizeof(void *) + element_size * snode_metas[i].chunk_size;
      }
      QD_TRACE("Initializing allocator for snode {} (node size {})", snode_id, node_size);
      runtime_jit->call<void *, int, std::size_t>("runtime_NodeAllocator_initialize", llvm_runtime_, snode_id,
                                                  node_size);
      QD_TRACE("Allocating ambient element for snode {} (node size {})", snode_id, node_size);
      runtime_jit->call<void *, int>("runtime_allocate_ambient", llvm_runtime_, snode_id, node_size);
    }
  }
}

LlvmDevice *LlvmRuntimeExecutor::llvm_device() {
  QD_ASSERT(dynamic_cast<LlvmDevice *>(device_.get()));
  return static_cast<LlvmDevice *>(device_.get());
}

DeviceAllocation LlvmRuntimeExecutor::allocate_memory_on_device(std::size_t alloc_size, uint64 *result_buffer) {
  auto devalloc = llvm_device()->allocate_memory_runtime({{alloc_size, /*host_write=*/false, /*host_read=*/false,
                                                           /*export_sharing=*/false, AllocUsage::Storage},
                                                          get_runtime_jit_module(),
                                                          get_llvm_runtime(),
                                                          result_buffer,
                                                          use_device_memory_pool()});

  QD_ERROR_IF(!devalloc.is_valid(),
              "Failed to allocate memory for "
              "allocate_memory_on_device(alloc_size=0x{:x})",
              alloc_size);

  QD_ASSERT(allocated_runtime_memory_allocs_.find(devalloc.alloc_id) == allocated_runtime_memory_allocs_.end());
  allocated_runtime_memory_allocs_[devalloc.alloc_id] = devalloc;
  return devalloc;
}

void LlvmRuntimeExecutor::deallocate_memory_on_device(DeviceAllocation handle) {
  QD_ASSERT(allocated_runtime_memory_allocs_.find(handle.alloc_id) != allocated_runtime_memory_allocs_.end());
  llvm_device()->dealloc_memory(handle);
  allocated_runtime_memory_allocs_.erase(handle.alloc_id);
}

void LlvmRuntimeExecutor::fill_ndarray(const DeviceAllocation &alloc, std::size_t size, uint32_t data) {
  auto ptr = get_device_alloc_info_ptr(alloc);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memsetd32((void *)ptr, data, size);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memset((void *)ptr, data, size);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    std::fill((uint32_t *)ptr, (uint32_t *)ptr + size, data);
  }
}

uint64_t *LlvmRuntimeExecutor::get_device_alloc_info_ptr(const DeviceAllocation &alloc) {
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    return (uint64_t *)llvm_device()->as<cuda::CudaDevice>()->get_alloc_info(alloc).ptr;
#else
    QD_NOT_IMPLEMENTED
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    return (uint64_t *)llvm_device()->as<amdgpu::AmdgpuDevice>()->get_alloc_info(alloc).ptr;
#else
    QD_NOT_IMPLEMENTED;
#endif
  }

  return (uint64_t *)llvm_device()->as<cpu::CpuDevice>()->get_alloc_info(alloc).ptr;
}

void LlvmRuntimeExecutor::finalize() {
  profiler_ = nullptr;
  // Release the host-owned adstack heap before the device teardown below so its `DeviceAllocationGuard` destructor
  // runs while the RHI device is still valid. The destructor drops the allocation back to the driver memory pool
  // (or to the host allocator on CPU); deferring past `llvm_device()->clear()` would leak it.
  adstack_heap_alloc_.reset();
  adstack_heap_size_ = 0;
  runtime_temporaries_cache_ = nullptr;
  runtime_adstack_heap_buffer_field_ptr_ = nullptr;
  runtime_adstack_heap_size_field_ptr_ = nullptr;
  runtime_adstack_heap_buffer_float_field_ptr_ = nullptr;
  runtime_adstack_heap_size_float_field_ptr_ = nullptr;
  runtime_adstack_heap_buffer_int_field_ptr_ = nullptr;
  runtime_adstack_heap_size_int_field_ptr_ = nullptr;
  adstack_heap_alloc_float_.reset();
  adstack_heap_size_float_ = 0;
  runtime_adstack_row_counters_field_ptr_ = nullptr;
  runtime_adstack_bound_row_capacities_field_ptr_ = nullptr;
  adstack_row_counters_alloc_.reset();
  adstack_bound_row_capacities_alloc_.reset();
  adstack_lazy_claim_capacity_ = 0;
  // Release the pinned-host metadata scratch and its completion event. Sequence: first drain the pending in-flight
  // copy via `event_synchronize` (the next launch's reuse path would have done this lazily, but on shutdown there
  // is no next launch), then free the host pinning, then destroy the event. Skipping the synchronize before
  // `mem_free_host` would race the DMA engine's read against the host free; skipping `event_destroy` would leak a
  // CUDA / HIP event handle.
  if (pinned_metadata_event_ != nullptr) {
#if defined(QD_WITH_CUDA)
    if (config_.arch == Arch::cuda) {
      if (pinned_metadata_event_pending_) {
        CUDADriver::get_instance().event_synchronize(pinned_metadata_event_);
      }
      CUDADriver::get_instance().event_destroy(pinned_metadata_event_);
    }
#endif
#if defined(QD_WITH_AMDGPU)
    if (config_.arch == Arch::amdgpu) {
      if (pinned_metadata_event_pending_) {
        AMDGPUDriver::get_instance().event_synchronize(pinned_metadata_event_);
      }
      AMDGPUDriver::get_instance().event_destroy(pinned_metadata_event_);
    }
#endif
    pinned_metadata_event_ = nullptr;
    pinned_metadata_event_pending_ = false;
  }
  if (pinned_metadata_scratch_ != nullptr) {
#if defined(QD_WITH_CUDA)
    if (config_.arch == Arch::cuda) {
      CUDADriver::get_instance().mem_free_host(pinned_metadata_scratch_);
    }
#endif
#if defined(QD_WITH_AMDGPU)
    if (config_.arch == Arch::amdgpu) {
      AMDGPUDriver::get_instance().mem_free_host(pinned_metadata_scratch_);
    }
#endif
    pinned_metadata_scratch_ = nullptr;
    pinned_metadata_scratch_capacity_ = 0;
  }
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
    preallocated_runtime_objects_allocs_.reset();
    preallocated_runtime_memory_allocs_.reset();

    // Reset runtime memory
    auto allocated_runtime_memory_allocs_copy = allocated_runtime_memory_allocs_;
    for (auto &iter : allocated_runtime_memory_allocs_copy) {
      // The runtime allocation may have already been freed upon explicit
      // Ndarray/Field destruction Check if the allocation still alive
      void *ptr = llvm_device()->get_memory_addr(iter.second);
      if (ptr == nullptr)
        continue;

      deallocate_memory_on_device(iter.second);
    }
    allocated_runtime_memory_allocs_.clear();

    // Reset device
    llvm_device()->clear();

    // Reset memory pool
    DeviceMemoryPool::get_instance(config_.arch).reset();

    // Release unused memory from cuda memory pool
    synchronize();
  }
  finalized_ = true;
}

LlvmRuntimeExecutor::~LlvmRuntimeExecutor() {
  if (!finalized_) {
    finalize();
  }
}

void *LlvmRuntimeExecutor::preallocate_memory(std::size_t prealloc_size, DeviceAllocationUnique &devalloc) {
  DeviceAllocation preallocated_device_buffer_alloc;

  Device::AllocParams preallocated_device_buffer_alloc_params;
  preallocated_device_buffer_alloc_params.size = prealloc_size;
  RhiResult res =
      llvm_device()->allocate_memory(preallocated_device_buffer_alloc_params, &preallocated_device_buffer_alloc);
  QD_ERROR_IF(res != RhiResult::success, "Failed to pre-allocate device memory (err: {})", int(res));

  void *preallocated_device_buffer = llvm_device()->get_memory_addr(preallocated_device_buffer_alloc);
  devalloc = std::make_unique<DeviceAllocationGuard>(std::move(preallocated_device_buffer_alloc));
  return preallocated_device_buffer;
}

void *LlvmRuntimeExecutor::get_runtime_temporaries_device_ptr() {
  if (runtime_temporaries_cache_ != nullptr) {
    return runtime_temporaries_cache_;
  }
  QD_ASSERT(llvm_runtime_ != nullptr);
  QD_ASSERT(result_buffer_cache_ != nullptr);
  auto *const runtime_jit = get_runtime_jit_module();
  runtime_jit->call<void *>("runtime_get_temporaries_ptr", llvm_runtime_);
  runtime_temporaries_cache_ = quadrants_union_cast_with_different_sizes<void *>(
      fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
  return runtime_temporaries_cache_;
}

// Publish the per-task adstack metadata into the LLVMRuntime struct and size the heap. The codegen path loads
// stride / offset / max_size from these fields at every `AdStack*` site (see `ensure_ad_stack_metadata_llvm` in
// codegen_llvm.cpp), so we must write them before every launch even for tasks where the compile-time and
// launch-time bounds agree. `evaluate_adstack_size_expr` is called only when the symbolic tree is available; the
// offline cache does not currently serialize `SizeExpr`, so cache hits fall back to `max_size_compile_time`.
std::size_t LlvmRuntimeExecutor::publish_adstack_metadata(const AdStackSizingInfo &ad_stack,
                                                          std::size_t num_threads,
                                                          LaunchContextBuilder *ctx,
                                                          void *device_runtime_context_ptr) {
  const auto n_stacks = ad_stack.allocas.size();
  if (n_stacks == 0 || num_threads == 0) {
    return 0;
  }
  auto align_up_8 = [](std::size_t n) -> std::size_t { return (n + 7u) & ~std::size_t{7u}; };
  // Allocate / grow the two device-side metadata arrays. Capacity is in u64 entries, kept at or above n_stacks.
  // On GPU these buffers are written exclusively by the device-side sizer kernel (`runtime_eval_adstack_size_expr`);
  // on CPU the host evaluator writes them directly via `std::memcpy`. Either way the pointers published into
  // `runtime->adstack_offsets` / `adstack_max_sizes` stay stable across launches unless we grow here.
  auto grow_to = [&](DeviceAllocationUnique &alloc, std::size_t capacity_u64) {
    Device::AllocParams params{};
    params.size = capacity_u64 * sizeof(uint64_t);
    params.host_read = false;
    params.host_write = false;
    params.export_sharing = false;
    params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success, "Failed to allocate {} bytes for adstack metadata array (err: {})",
                params.size, int(res));
    alloc = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
  };
  if (n_stacks > adstack_metadata_capacity_) {
    std::size_t new_cap = std::max<std::size_t>(n_stacks, 2 * adstack_metadata_capacity_);
    grow_to(adstack_offsets_alloc_, new_cap);
    grow_to(adstack_max_sizes_alloc_, new_cap);
    adstack_metadata_capacity_ = new_cap;
  }
  void *offsets_dev_ptr = get_device_alloc_info_ptr(*adstack_offsets_alloc_);
  void *max_sizes_dev_ptr = get_device_alloc_info_ptr(*adstack_max_sizes_alloc_);

  auto copy_h2d = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };
  auto copy_d2h = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_device_to_host(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_device_to_host(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };

  // Cache the runtime-field addresses on the first call; then publish the metadata-array pointers into the
  // runtime struct. The stride field is written by the sizer on GPU and by this function on CPU, so we cache the
  // address either way.
  if (runtime_adstack_stride_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_metadata_field_ptrs", llvm_runtime_);
    // Slot order: combined-stride, offsets, max_sizes, float-stride, int-stride. Slots 0/1/2 keep the legacy
    // ordering for code paths that have not migrated to the split layout; slots 3/4 are new.
    runtime_adstack_stride_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_offsets_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_max_sizes_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_stride_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
    runtime_adstack_stride_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 4, result_buffer_cache_));
  }
  copy_h2d(runtime_adstack_offsets_field_ptr_, &offsets_dev_ptr, sizeof(void *));
  copy_h2d(runtime_adstack_max_sizes_field_ptr_, &max_sizes_dev_ptr, sizeof(void *));

  std::size_t stride = 0;
  const bool is_gpu_llvm = (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu);

  // Host-eval fast path. The on-device sizer kernel exists to handle one specific leaf, `ExternalTensorRead`,
  // whose ndarray data lives in GPU-private memory (`cudaMalloc` / `hipMalloc`, no UVA fallback) and thus
  // cannot be touched from the host. Every other SizeExpr leaf - `Const`, `BoundVariable`,
  // `ExternalTensorShape`, `FieldLoad` - is host-resolvable through the existing `evaluate_adstack_size_expr`
  // path, so when the kernel's SizeExprs are all `ExternalTensorRead`-free we can skip the encode + bytecode
  // h2d + sizer-kernel launch + d2h-stride pipeline entirely and write the metadata directly via `copy_h2d`.
  // On CUDA the saved `cuMemcpyDtoH` for the per-launch stride readback is the dominant cost: every reverse-
  // mode kernel launch in a 100-substep test paid one such synchronous DtoH each, and that compound stall
  // accounted for the bulk of the GPU launch overhead under adstack mode. The condition is computed once per
  // launch by scanning each stack's `nodes` vector for an `ExternalTensorRead` leaf; the scan is O(total
  // SizeExpr nodes), well below the cost of the cheapest h2d / d2h on any LLVM GPU backend.
  bool all_size_exprs_host_resolvable = true;
  for (std::size_t i = 0; i < n_stacks && all_size_exprs_host_resolvable; ++i) {
    if (i >= ad_stack.size_exprs.size()) {
      continue;
    }
    for (const auto &node : ad_stack.size_exprs[i].nodes) {
      if (static_cast<SizeExpr::Kind>(node.kind) == SizeExpr::Kind::ExternalTensorRead) {
        all_size_exprs_host_resolvable = false;
        break;
      }
    }
  }
  const bool use_host_eval = !is_gpu_llvm || all_size_exprs_host_resolvable;
  if (use_host_eval) {
    // CPU + GPU-without-ExternalTensorRead path: run the host evaluator directly. On CPU we use synchronous
    // `copy_h2d` (just `std::memcpy` for that arch), but on CUDA / AMDGPU we ship the same payload through
    // pinned-host memory via async `cuMemcpyHtoDAsync` / `hipMemcpyHtoDAsync` so the host returns immediately
    // after queueing the copies on the default stream and the subsequent main-kernel launch (also on the
    // default stream) stream-orders after the copies. The synchronous `cuMemcpyHtoD_v2` path used to block
    // the host on every one of the three writes we issue per launch; with thousands of reverse-mode launches
    // per `test_differentiable_rigid` run, those serial host stalls were a measurable fraction of wallclock.
    // `FieldLoad` is serviced by `SNodeRwAccessorsBank` regardless of arch.
    // Guard `program_impl_->program` lookups against the C++-only-tests setup where `program_impl_` itself is null;
    // the on-device branch below already does this and falls back to `max_size_compile_time`.
    Program *prog = (program_impl_ != nullptr) ? program_impl_->program : nullptr;
    std::vector<uint64_t> host_max_sizes(n_stacks);
    for (std::size_t i = 0; i < n_stacks; ++i) {
      const SerializedSizeExpr *expr = (i < ad_stack.size_exprs.size()) ? &ad_stack.size_exprs[i] : nullptr;
      int64_t v = -1;
      if (expr != nullptr && !expr->nodes.empty() && prog != nullptr) {
        v = evaluate_adstack_size_expr(*expr, prog, ctx);
      }
      if (v < 0) {
        v = static_cast<int64_t>(ad_stack.allocas[i].max_size_compile_time);
      }
      host_max_sizes[i] = static_cast<uint64_t>(std::max<int64_t>(v, 1));
    }
    // Two layouts depending on whether the task captured a `bound_expr`:
    //   * No bound_expr: legacy single-heap layout. `host_offsets[i]` is the cumulative byte offset within the
    //     combined slice; the published combined stride drives `linear_tid * stride + offset` addressing on the
    //     codegen side. Float and int allocas share the same slice.
    //   * Bound_expr captured: split-heap layout. Float allocas live in the dedicated float heap addressed by
    //     `row_id_var * stride_float + float_offset`; their `host_offsets[i]` is the byte offset within the
    //     float-only slice. Int allocas live in the combined heap addressed by `linear_tid * stride_int +
    //     int_offset`; their `host_offsets[i]` is the byte offset within the int-only slice. The combined-heap
    //     stride published to the runtime (`adstack_per_thread_stride`) is the int-only stride for these tasks
    //     because the combined slice no longer holds the float allocas - they have migrated to the float heap.
    // This split is what shrinks per-thread combined storage to int-only on tasks where the float allocations
    // dominate, and lets the float heap be sized from the captured `bound_row_capacities[i]` count instead of
    // the dispatched-threads worst case.
    const bool split_layout = ad_stack.bound_expr.has_value();
    std::vector<uint64_t> host_offsets(n_stacks);
    std::size_t stride_combined = 0;
    std::size_t stride_float = 0;
    std::size_t stride_int = 0;
    for (std::size_t i = 0; i < n_stacks; ++i) {
      const std::size_t step = align_up_8(sizeof(int64_t) + ad_stack.allocas[i].entry_size_bytes * host_max_sizes[i]);
      const bool is_float = ad_stack.allocas[i].heap_kind == AdStackAllocaInfo::HeapKind::Float;
      if (split_layout) {
        host_offsets[i] = is_float ? stride_float : stride_int;
      } else {
        host_offsets[i] = stride_combined;
      }
      stride_combined += step;
      if (is_float) {
        stride_float += step;
      } else {
        stride_int += step;
      }
    }
    // Stride published into `runtime->adstack_per_thread_stride` is what the legacy combined-heap codegen reads
    // for the `linear_tid * stride + offset` formula. On tasks with `bound_expr` the combined slice is int-only
    // (float allocas migrated to the float heap), so we publish `stride_int`; on tasks without, the full combined
    // stride. Tasks alternate using the same combined heap with different strides per launch.
    stride = split_layout ? stride_int : stride_combined;
    uint64_t stride_combined_u64 = static_cast<uint64_t>(stride);
    uint64_t stride_float_u64 = static_cast<uint64_t>(stride_float);
    uint64_t stride_int_u64 = static_cast<uint64_t>(stride_int);
    if (!is_gpu_llvm) {
      copy_h2d(offsets_dev_ptr, host_offsets.data(), n_stacks * sizeof(uint64_t));
      copy_h2d(max_sizes_dev_ptr, host_max_sizes.data(), n_stacks * sizeof(uint64_t));
      copy_h2d(runtime_adstack_stride_field_ptr_, &stride_combined_u64, sizeof(uint64_t));
      // Per-kind strides used by the split-heap codegen path; harmless when the codegen has not migrated yet
      // (the kernel reads only the combined stride). Skipped when the cache is empty (first launch on a stale
      // executor instance where `runtime_get_adstack_metadata_field_ptrs` populated only the legacy slots; the
      // null check is defensive - any host writing to `nullptr` would crash with no diagnostic).
      if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
        copy_h2d(runtime_adstack_stride_float_field_ptr_, &stride_float_u64, sizeof(uint64_t));
      }
      if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
        copy_h2d(runtime_adstack_stride_int_field_ptr_, &stride_int_u64, sizeof(uint64_t));
      }
    } else {
      // Five-block payload packed into the pinned-host scratch as `[stride_combined, stride_float, stride_int,
      // offsets[n_stacks], max_sizes[n_stacks]]`. Five async DMAs land on the matching device addresses; the
      // driver's H2D DMA engine reads from the pinned bytes at execution time, so we must not overwrite the
      // scratch before all copies have completed - hence the per-launch `event_record` after the last copy and
      // the `event_synchronize` at the top of the next launch.
      const std::size_t header_bytes = 3 * sizeof(uint64_t);
      const std::size_t array_bytes = n_stacks * sizeof(uint64_t);
      const std::size_t total_bytes = header_bytes + 2 * array_bytes;

      auto wait_pending = [this]() {
        if (!pinned_metadata_event_pending_) {
          return;
        }
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().event_synchronize(pinned_metadata_event_);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          AMDGPUDriver::get_instance().event_synchronize(pinned_metadata_event_);
        }
#endif
        pinned_metadata_event_pending_ = false;
      };

      // Grow / first-allocate the pinned host scratch and the per-launch completion event. Doubling growth
      // means the pinned alloc / free traffic is amortised to O(log peak_total_bytes) across a run.
      if (total_bytes > pinned_metadata_scratch_capacity_) {
        wait_pending();
        if (pinned_metadata_scratch_ != nullptr) {
#if defined(QD_WITH_CUDA)
          if (config_.arch == Arch::cuda) {
            CUDADriver::get_instance().mem_free_host(pinned_metadata_scratch_);
          }
#endif
#if defined(QD_WITH_AMDGPU)
          if (config_.arch == Arch::amdgpu) {
            AMDGPUDriver::get_instance().mem_free_host(pinned_metadata_scratch_);
          }
#endif
          pinned_metadata_scratch_ = nullptr;
        }
        std::size_t new_capacity = std::max<std::size_t>(total_bytes, 2 * pinned_metadata_scratch_capacity_);
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          // `hipHostMallocDefault == 0`. Coherent / portable / write-combined flags are intentionally not set;
          // the workload is small payloads written linearly by the host and DMA-read by the GPU once.
          AMDGPUDriver::get_instance().mem_alloc_host(&pinned_metadata_scratch_, new_capacity, 0u);
        }
#endif
        pinned_metadata_scratch_capacity_ = new_capacity;
      }
      if (pinned_metadata_event_ == nullptr) {
        // `cuEventCreate` flag `0` (CU_EVENT_DEFAULT) means timing-enabled, which the driver costs us nothing
        // to set up here and lets future profilers attach without re-creating the event. `hipEventCreateWithFlags`
        // takes the same encoding.
#if defined(QD_WITH_CUDA)
        if (config_.arch == Arch::cuda) {
          CUDADriver::get_instance().event_create(&pinned_metadata_event_, 0u);
        }
#endif
#if defined(QD_WITH_AMDGPU)
        if (config_.arch == Arch::amdgpu) {
          AMDGPUDriver::get_instance().event_create(&pinned_metadata_event_, 0u);
        }
#endif
      }
      // Block until any in-flight copies from the previous launch have finished pulling from the pinned scratch
      // before we overwrite it. In steady state this is a no-op because the small DMAs finish well before the
      // host loops back here; the wait exists only to defend against an unusual interleaving where the GPU
      // queue is backlogged and the next launch enters this function before the previous launch's last copy
      // has been consumed.
      wait_pending();

      auto *pinned = static_cast<uint64_t *>(pinned_metadata_scratch_);
      pinned[0] = stride_combined_u64;
      pinned[1] = stride_float_u64;
      pinned[2] = stride_int_u64;
      std::memcpy(pinned + 3, host_offsets.data(), array_bytes);
      std::memcpy(pinned + 3 + n_stacks, host_max_sizes.data(), array_bytes);

      // Queue the metadata copies on the same stream the subsequent main-kernel dispatch will run on, so the
      // GPU stream-orders the copies before the kernel reads `adstack_max_sizes` etc. On CUDA the active
      // stream is `CUDAContext::get_instance().get_stream()` - configurable via `set_stream`, defaults to the
      // null stream - and `CUDAContext::launch` dispatches kernels on the same handle. AMDGPU has no
      // public stream-selection API: `AMDGPUContext::launch` always passes `nullptr` to `hipLaunchKernel`
      // (i.e. the default stream), so the copies match that.
#if defined(QD_WITH_CUDA)
      if (config_.arch == Arch::cuda) {
        void *active_stream = CUDAContext::get_instance().get_stream();
        CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_field_ptr_, pinned,
                                                               sizeof(uint64_t), active_stream);
        if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
          CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_float_field_ptr_, pinned + 1,
                                                                 sizeof(uint64_t), active_stream);
        }
        if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
          CUDADriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_int_field_ptr_, pinned + 2,
                                                                 sizeof(uint64_t), active_stream);
        }
        CUDADriver::get_instance().memcpy_host_to_device_async(offsets_dev_ptr, pinned + 3, array_bytes, active_stream);
        CUDADriver::get_instance().memcpy_host_to_device_async(max_sizes_dev_ptr, pinned + 3 + n_stacks, array_bytes,
                                                               active_stream);
        CUDADriver::get_instance().event_record(pinned_metadata_event_, active_stream);
      }
#endif
#if defined(QD_WITH_AMDGPU)
      if (config_.arch == Arch::amdgpu) {
        void *active_stream = nullptr;  // AMDGPUContext::launch always uses the default stream.
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_field_ptr_, pinned,
                                                                 sizeof(uint64_t), active_stream);
        if (runtime_adstack_stride_float_field_ptr_ != nullptr) {
          AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_float_field_ptr_, pinned + 1,
                                                                   sizeof(uint64_t), active_stream);
        }
        if (runtime_adstack_stride_int_field_ptr_ != nullptr) {
          AMDGPUDriver::get_instance().memcpy_host_to_device_async(runtime_adstack_stride_int_field_ptr_, pinned + 2,
                                                                   sizeof(uint64_t), active_stream);
        }
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(offsets_dev_ptr, pinned + 3, array_bytes,
                                                                 active_stream);
        AMDGPUDriver::get_instance().memcpy_host_to_device_async(max_sizes_dev_ptr, pinned + 3 + n_stacks, array_bytes,
                                                                 active_stream);
        AMDGPUDriver::get_instance().event_record(pinned_metadata_event_, active_stream);
      }
#endif
      pinned_metadata_event_pending_ = true;
    }
  } else {
    // GPU (CUDA / AMDGPU): encode the SizeExpr trees into device bytecode, upload, launch the sizer runtime
    // function, read back just the computed stride. The sizer kernel writes `adstack_max_sizes[]`,
    // `adstack_offsets[]`, and `adstack_per_thread_stride` directly into the runtime struct and the metadata
    // arrays above - no further host-writes to those fields are needed this launch.
    //
    // Why this architecture rather than host-eval: on CUDA / AMDGPU the ndarray data lives in GPU-private memory
    // (plain `cudaMalloc` / `hipMalloc`, not managed / unified), so the host evaluator's `ExternalTensorRead`
    // deref reads garbage. Moving the interpreter on-device keeps the pointer semantics intact - it reads the
    // data pointer out of `ctx->arg_buffer` (which the kernel will read too) and dereferences it where the
    // memory lives, with no migration / readback of the ndarray payload itself.
    std::vector<uint8_t> bytecode;
    if (program_impl_ != nullptr && program_impl_->program != nullptr) {
      bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, program_impl_->program, ctx);
    } else {
      // No program attached (rare: C++-only tests that construct Program without a full runtime). Fall through
      // to compile-time bounds by emitting an empty-tree bytecode - the device interpreter sees
      // `root_node_idx == -1` for every stack and routes to `max_size_compile_time`.
      bytecode = encode_adstack_size_expr_device_bytecode(ad_stack, nullptr, ctx);
    }
    // Grow the scratch buffer if the bytecode outgrew the cached capacity. Amortised doubling keeps the
    // allocation traffic O(log max_bytecode_bytes) across a run.
    const std::size_t bytecode_bytes = bytecode.size();
    if (bytecode_bytes > adstack_sizer_bytecode_capacity_) {
      std::size_t new_cap = std::max<std::size_t>(bytecode_bytes, 2 * adstack_sizer_bytecode_capacity_);
      Device::AllocParams params{};
      params.size = new_cap;
      params.host_read = false;
      params.host_write = false;
      params.export_sharing = false;
      params.usage = AllocUsage::Storage;
      DeviceAllocation new_alloc;
      RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
      QD_ERROR_IF(res != RhiResult::success,
                  "Failed to allocate {} bytes for the adstack sizer bytecode scratch buffer (err: {})", params.size,
                  int(res));
      adstack_sizer_bytecode_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
      adstack_sizer_bytecode_capacity_ = new_cap;
    }
    void *bytecode_dev_ptr = get_device_alloc_info_ptr(*adstack_sizer_bytecode_alloc_);
    copy_h2d(bytecode_dev_ptr, bytecode.data(), bytecode_bytes);

    // Invoke the device interpreter. On CUDA / AMDGPU `JITModule::call` launches this as a single-thread kernel
    // on the default stream and stream-orders it before the subsequent main-kernel dispatch, so the writes we
    // do here are visible by the time the user's kernel reads `adstack_max_sizes` etc.
    //
    // The sizer kernel dereferences `ctx->arg_buffer` on device (that's how it resolves `ExternalTensorRead` leaves
    // against ndarray pointers the caller packed into the arg buffer). AMDGPU always stages a device-side copy of
    // `RuntimeContext` because HIP has no UVA fallback and the host pointer faults with `hipErrorIllegalAddress`. CUDA
    // stages the device copy only when the driver + kernel do not expose HMM / system-allocated memory (queried via
    // `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS`): CUDA UVA covers pinned / CUDA-managed memory only, not the plain
    // `std::make_unique<RuntimeContext>()` backing, so a host pointer works on HMM-capable setups but faults otherwise
    // (Turing without HMM, Windows, pre-535 Linux drivers) as `CUDA_ERROR_ILLEGAL_ADDRESS` at the next DtoH sync
    // `illegal memory access ... while calling memcpy_device_to_host`. When the caller passes `nullptr` (HMM-capable
    // CUDA) we fall back to the host pointer; the launcher gates the allocation so HMM-equipped setups pay no staging
    // cost.
    auto *const runtime_jit = get_runtime_jit_module();
    void *runtime_context_ptr_for_sizer =
        device_runtime_context_ptr != nullptr ? device_runtime_context_ptr : static_cast<void *>(&ctx->get_context());
    runtime_jit->call<void *, void *, void *>("runtime_eval_adstack_size_expr", llvm_runtime_,
                                              runtime_context_ptr_for_sizer, bytecode_dev_ptr);

    // Read back the computed per-thread stride so we can size the heap on host. One 8-byte `DtoH` per launch.
    uint64_t stride_u64 = 0;
    copy_d2h(&stride_u64, runtime_adstack_stride_field_ptr_, sizeof(uint64_t));
    stride = static_cast<std::size_t>(stride_u64);
  }

  std::size_t needed_bytes = stride * num_threads;
  ensure_adstack_heap(needed_bytes);
  // Float heap: when the task captured a `bound_expr`, allocate a dedicated float slab so the codegen-emitted
  // `heap_float + row_id_var * stride_float + offset` formula has backing storage. Sized at
  // `num_threads * per_thread_stride_float`, where `per_thread_stride_float` is the compile-time sum of float
  // alloca sizes (from `AdStackSizingInfo::per_thread_stride_float`). On the GPU path the per-launch value lives
  // in `runtime->adstack_per_thread_stride_float` written by the on-device sizer; reading it back here would
  // require an extra DtoH so we use the compile-time value, which serves as an upper bound for the same reason
  // the legacy combined `per_thread_stride` is - the runtime-evaluated SizeExprs only ever equal-or-shrink the
  // compile-time bound.
  if (ad_stack.bound_expr.has_value() && ad_stack.per_thread_stride_float > 0) {
    ensure_adstack_heap_float(ad_stack.per_thread_stride_float * num_threads);
  }
  return needed_bytes;
}

namespace {

// Encode the captured `BinaryOpType` (stored as int in `cmp_op`) and evaluate against typed operands. Mirrors the
// SPIR-V reducer's `OpSwitch` over the same encoding.
template <typename T>
inline bool eval_cmp(int cmp_op, T lhs, T rhs) {
  switch (static_cast<BinaryOpType>(cmp_op)) {
    case BinaryOpType::cmp_lt:
      return lhs < rhs;
    case BinaryOpType::cmp_le:
      return lhs <= rhs;
    case BinaryOpType::cmp_gt:
      return lhs > rhs;
    case BinaryOpType::cmp_ge:
      return lhs >= rhs;
    case BinaryOpType::cmp_eq:
      return lhs == rhs;
    case BinaryOpType::cmp_ne:
      return lhs != rhs;
    default:
      return false;
  }
}

}  // namespace

uint32_t LlvmRuntimeExecutor::publish_per_task_bound_count_cpu(std::size_t task_index,
                                                               const AdStackSizingInfo &ad_stack,
                                                               std::size_t length,
                                                               LaunchContextBuilder *ctx) {
  // Default to UINT32_MAX (no clamp); only override on a successful host evaluation. The codegen-emitted bounds
  // clamp at the float LCA-block claim site stays inert when the slot holds UINT32_MAX, so this fall-through is
  // a no-op that preserves the existing behaviour.
  if (config_.arch != Arch::x64 && config_.arch != Arch::arm64) {
    return std::numeric_limits<uint32_t>::max();
  }
  if (!ad_stack.bound_expr.has_value()) {
    return std::numeric_limits<uint32_t>::max();
  }
  const auto &be = ad_stack.bound_expr.value();
  if (be.field_source_kind != StaticAdStackBoundExpr::FieldSourceKind::NdArray) {
    return std::numeric_limits<uint32_t>::max();  // SNode-backed gates are not captured on the LLVM analysis path.
  }
  if (ctx == nullptr || ctx->args_type == nullptr || ctx->get_context().arg_buffer == nullptr) {
    return std::numeric_limits<uint32_t>::max();
  }

  // Resolve the ndarray data pointer: walk `ctx->args_type->get_element_offset(arg_id + DATA_PTR_POS_IN_NDARRAY)`
  // to find where the data pointer lives in the arg buffer, then dereference. Mirrors the SPIR-V reducer's
  // `resolve_ndarray_data_ptr_byte_offset`. On CPU `arg_buffer` is host memory, so the deref is direct.
  std::vector<int> indices = be.ndarray_arg_id;
  indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  std::size_t data_ptr_byte_off = ctx->args_type->get_element_offset(indices);
  const char *arg_buffer = static_cast<const char *>(ctx->get_context().arg_buffer);
  void *data_ptr = *reinterpret_cast<void *const *>(arg_buffer + data_ptr_byte_off);
  if (data_ptr == nullptr) {
    return std::numeric_limits<uint32_t>::max();
  }

  // Walk `[0, length)` evaluating the captured predicate on each thread's `field[i]`. The polarity bit selects
  // enter-on-true vs enter-on-false at the LCA's IfStmt; the count we publish is always the number of threads
  // that REACH the LCA, regardless of the gate orientation.
  uint32_t count = 0;
  if (be.field_dtype_is_float) {
    const float *fdata = static_cast<const float *>(data_ptr);
    for (std::size_t i = 0; i < length; ++i) {
      const bool match = eval_cmp<float>(be.cmp_op, fdata[i], be.literal_f32);
      if (be.polarity ? match : !match) {
        ++count;
      }
    }
  } else {
    const int32_t *idata = static_cast<const int32_t *>(data_ptr);
    for (std::size_t i = 0; i < length; ++i) {
      const bool match = eval_cmp<int32_t>(be.cmp_op, idata[i], be.literal_i32);
      if (be.polarity ? match : !match) {
        ++count;
      }
    }
  }

  // Publish the count into `runtime->adstack_bound_row_capacities[task_index]` so the codegen-emitted bounds
  // clamp at the float LCA-block claim site reads it back as the per-task capacity. Slot was reset to
  // UINT32_MAX by `publish_adstack_lazy_claim_buffers`; this overwrite tightens it to the real count.
  if (runtime_adstack_bound_row_capacities_field_ptr_ == nullptr || adstack_bound_row_capacities_alloc_ == nullptr) {
    return count;
  }
  void *bound_capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);
  // CPU only: write directly into the host-resident array.
  uint32_t *slots = static_cast<uint32_t *>(bound_capacities_dev_ptr);
  slots[task_index] = count;
  return count;
}

namespace {

// Encode the captured `BinaryOpType` into the 0-5 numeric range the LLVM device reducer's switch consumes.
// Mirrors the SPIR-V reducer's `encode_cmp_op` mapping at `quadrants/runtime/gfx/adstack_bound_reducer_launch.cpp`.
uint32_t encode_cmp_op_for_llvm_reducer(int captured_cmp_op) {
  switch (static_cast<BinaryOpType>(captured_cmp_op)) {
    case BinaryOpType::cmp_lt:
      return kLlvmReducerCmpLt;
    case BinaryOpType::cmp_le:
      return kLlvmReducerCmpLe;
    case BinaryOpType::cmp_gt:
      return kLlvmReducerCmpGt;
    case BinaryOpType::cmp_ge:
      return kLlvmReducerCmpGe;
    case BinaryOpType::cmp_eq:
      return kLlvmReducerCmpEq;
    case BinaryOpType::cmp_ne:
      return kLlvmReducerCmpNe;
    default:
      return std::numeric_limits<uint32_t>::max();
  }
}

}  // namespace

void LlvmRuntimeExecutor::publish_per_task_bound_count_device(std::size_t task_index,
                                                              const AdStackSizingInfo &ad_stack,
                                                              std::size_t length,
                                                              LaunchContextBuilder *ctx,
                                                              void *device_runtime_context_ptr) {
  // Only fires for CUDA / AMDGPU; CPU goes through `publish_per_task_bound_count_cpu`. Bail when the task did
  // not capture a bound_expr (no clamp needed - the slot stays at the UINT32_MAX default
  // `publish_adstack_lazy_claim_buffers` wrote) or when the field source isn't ndarray (SNode-backed gates
  // are not captured on the LLVM analysis path so they don't reach here, but the guard is cheap).
  if (config_.arch != Arch::cuda && config_.arch != Arch::amdgpu) {
    return;
  }
  if (!ad_stack.bound_expr.has_value()) {
    return;
  }
  const auto &be = ad_stack.bound_expr.value();
  if (be.field_source_kind != StaticAdStackBoundExpr::FieldSourceKind::NdArray) {
    return;
  }
  if (ctx == nullptr || ctx->args_type == nullptr) {
    return;
  }

  // Resolve the ndarray data pointer's word offset within the kernel arg buffer. Same path the SPIR-V reducer
  // and the CPU host-eval use; bytes -> words for the reducer's `arg_buffer_u32[arg_word_offset]` indexing.
  std::vector<int> indices = be.ndarray_arg_id;
  indices.push_back(TypeFactory::DATA_PTR_POS_IN_NDARRAY);
  std::size_t data_ptr_byte_off = ctx->args_type->get_element_offset(indices);
  if (data_ptr_byte_off % sizeof(uint32_t) != 0) {
    return;  // misaligned offset; the reducer's u32-word indexing would lose bits.
  }
  const uint32_t arg_word_offset = static_cast<uint32_t>(data_ptr_byte_off / sizeof(uint32_t));
  const uint32_t cmp_op_encoded = encode_cmp_op_for_llvm_reducer(be.cmp_op);
  if (cmp_op_encoded == std::numeric_limits<uint32_t>::max()) {
    return;  // unrecognised comparison op (the IR pattern matcher should have rejected it earlier)
  }

  // Fill the device-side params struct on the host. Threshold bits live as the same u32 the runtime function
  // bitcasts back; we copy whichever underlying integer or float value the analysis captured.
  LlvmAdStackBoundReducerDeviceParams params{};
  params.task_index = static_cast<uint32_t>(task_index);
  params.length = static_cast<uint32_t>(length);
  params.cmp_op = cmp_op_encoded;
  params.field_dtype_is_float = be.field_dtype_is_float ? 1u : 0u;
  params.polarity = be.polarity ? 1u : 0u;
  if (be.field_dtype_is_float) {
    std::memcpy(&params.threshold_bits, &be.literal_f32, sizeof(uint32_t));
  } else {
    params.threshold_bits = static_cast<uint32_t>(be.literal_i32);
  }
  params.arg_word_offset = arg_word_offset;
  params.padding = 0;

  // Lazy-allocate the device-side params scratch buffer the first time a bound_expr task fires; reuse for
  // subsequent tasks across kernels. Sized for one struct (the reducer is single-task per call); a future
  // optimisation could pack multiple tasks' params into one buffer and dispatch them in a single launch.
  const std::size_t needed_bytes = sizeof(LlvmAdStackBoundReducerDeviceParams);
  if (needed_bytes > adstack_bound_reducer_params_capacity_) {
    Device::AllocParams alloc_params{};
    alloc_params.size = std::max<std::size_t>(needed_bytes, 2 * adstack_bound_reducer_params_capacity_);
    alloc_params.host_read = false;
    alloc_params.host_write = true;
    alloc_params.export_sharing = false;
    alloc_params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(alloc_params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success,
                "Failed to allocate {} bytes for adstack bound reducer params buffer (err: {})", alloc_params.size,
                int(res));
    adstack_bound_reducer_params_alloc_ = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
    adstack_bound_reducer_params_capacity_ = alloc_params.size;
  }
  void *params_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_reducer_params_alloc_);

  // h2d the params struct into the device buffer.
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(params_dev_ptr, &params, needed_bytes);
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(params_dev_ptr, &params, needed_bytes);
#else
    QD_NOT_IMPLEMENTED;
#endif
  }

  // Dispatch the runtime reducer function: single-threaded device-side walk that reads `ctx->arg_buffer`
  // (the device-mirror the launcher staged) and writes the count into
  // `runtime->adstack_bound_row_capacities[task_index]`. Pass the device-side `RuntimeContext` pointer the
  // same way the size-expr sizer does so the function can deref `ctx->arg_buffer` on-device.
  auto *const runtime_jit = get_runtime_jit_module();
  void *runtime_context_ptr_for_reducer =
      device_runtime_context_ptr != nullptr ? device_runtime_context_ptr : static_cast<void *>(&ctx->get_context());
  runtime_jit->call<void *, void *, void *>("runtime_eval_static_bound_count", llvm_runtime_,
                                            runtime_context_ptr_for_reducer, params_dev_ptr);
}

void LlvmRuntimeExecutor::publish_adstack_lazy_claim_buffers(std::size_t num_tasks) {
  if (num_tasks == 0) {
    return;
  }
  // Cache the field-of-LLVMRuntime addresses for the row counter / bound row capacity array pointers. Resolved
  // once per program lifetime; subsequent grows write the new array pointers directly to the cached addresses.
  if (runtime_adstack_row_counters_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_lazy_claim_field_ptrs", llvm_runtime_);
    runtime_adstack_row_counters_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_bound_row_capacities_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
  }

  auto grow_to = [&](DeviceAllocationUnique &alloc, std::size_t capacity_u32) {
    Device::AllocParams params{};
    params.size = capacity_u32 * sizeof(uint32_t);
    params.host_read = false;
    params.host_write = false;
    params.export_sharing = false;
    params.usage = AllocUsage::Storage;
    DeviceAllocation new_alloc;
    RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
    QD_ERROR_IF(res != RhiResult::success, "Failed to allocate {} bytes for adstack lazy-claim array (err: {})",
                params.size, int(res));
    alloc = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));
  };

  bool grew = false;
  if (num_tasks > adstack_lazy_claim_capacity_) {
    std::size_t new_cap = std::max<std::size_t>(num_tasks, 2 * adstack_lazy_claim_capacity_);
    grow_to(adstack_row_counters_alloc_, new_cap);
    grow_to(adstack_bound_row_capacities_alloc_, new_cap);
    adstack_lazy_claim_capacity_ = new_cap;
    grew = true;
  }
  void *row_counters_dev_ptr = get_device_alloc_info_ptr(*adstack_row_counters_alloc_);
  void *bound_capacities_dev_ptr = get_device_alloc_info_ptr(*adstack_bound_row_capacities_alloc_);

  // After every grow, publish the new array pointers into the runtime so the codegen-emitted GEPs
  // (`runtime->adstack_row_counters[task_codegen_id]` and `runtime->adstack_bound_row_capacities[task_codegen_id]`)
  // resolve against the live allocations. Skipped between grows because the cached field address holds the same
  // pointer value.
  auto copy_h2d = [&](void *dst, const void *src, std::size_t bytes) {
    if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
      CUDADriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
      AMDGPUDriver::get_instance().memcpy_host_to_device(dst, const_cast<void *>(src), bytes);
#else
      QD_NOT_IMPLEMENTED;
#endif
    } else {
      std::memcpy(dst, src, bytes);
    }
  };
  if (grew) {
    copy_h2d(runtime_adstack_row_counters_field_ptr_, &row_counters_dev_ptr, sizeof(void *));
    copy_h2d(runtime_adstack_bound_row_capacities_field_ptr_, &bound_capacities_dev_ptr, sizeof(void *));
  }

  // Per-launch reset: zero the counter slots (each task's LCA-block atomic-rmw add starts from 0 and accumulates
  // its own claims) and write UINT32_MAX into the capacity slots so the codegen-emitted bounds clamp is inert
  // unless a follow-up reducer overrides slots with tighter counts. Memset rather than per-slot store: the host
  // pays one O(num_tasks) buffer fill per kernel-launch, regardless of arch.
  std::vector<uint32_t> zero_buf(num_tasks, 0u);
  std::vector<uint32_t> uint_max_buf(num_tasks, std::numeric_limits<uint32_t>::max());
  copy_h2d(row_counters_dev_ptr, zero_buf.data(), num_tasks * sizeof(uint32_t));
  copy_h2d(bound_capacities_dev_ptr, uint_max_buf.data(), num_tasks * sizeof(uint32_t));
}

void LlvmRuntimeExecutor::ensure_adstack_heap(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_) {
    return;
  }
  // Amortized doubling keeps the number of re-allocations across a run bounded by log(peak_size).
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack heap (err: {}). Consider lowering `ad_stack_size` or the "
              "per-kernel reverse-mode adstack count.",
              new_size, int(res));
  // `get_device_alloc_info_ptr` is the RHI-agnostic accessor that returns the raw host-visible
  // pointer on CPU and the device-visible pointer on CUDA / AMDGPU (`get_memory_addr` is only
  // implemented on the GPU devices, so we route through this helper instead).
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);

  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // Publish the new buffer pointer and size into the runtime struct. On CPU the runtime lives in host memory,
  // so plain stores through the cached field pointers are correct. On CUDA / AMDGPU the runtime lives in device
  // memory, so the host writes via the driver's host->device memcpy. The field-address query runs exactly once,
  // on the first grow, and caches the two device pointers; every subsequent grow is just two 8-byte memcpys.
  if (runtime_adstack_heap_buffer_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_field_ptr_, &new_ptr, sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_field_ptr_, &size_u64, sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_field_ptr_, &size_u64, sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_field_ptr_) = size_u64;
  }

  // Replace and release the old allocation. `DeviceAllocationGuard`'s destructor calls
  // `llvm_device()->dealloc_memory`. The new slab has already been handed to `new_guard` above, so the move-assignment
  // here is what destroys the *previous* guard - the new allocation is not the one being freed. Safety of the release
  // depends on the backend:
  //   - CPU: host `std::free`. No GPU involved, always safe.
  //   - CUDA: `CudaDevice::dealloc_memory` routes through `DeviceMemoryPool::release(release_raw=true)` ->
  //     `cuMemFree_v2`, which synchronizes with pending device work before returning.
  //   - AMDGPU: `AmdgpuDevice::dealloc_memory` routes through `DeviceMemoryPool::release(release_raw=false)` ->
  //     `CachingAllocator::release`, which pools the allocation *without* calling `hipFree` and *without*
  //     synchronizing. The physical memory stays mapped, so an in-flight kernel still holding the old base pointer
  //     keeps reading/writing valid storage. The cross-launch safety invariant for AMDGPU comes from
  //     `amdgpu::KernelLauncher::launch_llvm_kernel` ending with `hipFree(context_pointer)`, which synchronizes
  //     with all in-flight kernels launched during that call. By the time the *next* `launch_llvm_kernel` reaches
  //     `ensure_adstack_heap` and can destroy the previous guard, no GPU kernel from the prior call is still
  //     referencing the old slab. CUDA does not need this extra hop -- the `cuMemFree_v2` in the bullet above
  //     already syncs -- and the CUDA launcher correspondingly does not allocate a device-side `context_pointer`
  //     (it passes the `RuntimeContext` by host reference).
  adstack_heap_alloc_ = std::move(new_guard);
  adstack_heap_size_ = new_size;
  // DEBUG: combined-heap allocation tracker. Lets the user see the legacy combined-heap allocations alongside
  // the new split float-heap allocations the per-arch reducer drives. Revert before merging.
  fprintf(stderr, "[ADSTACK-HEAP-LLVM] alloc new_size=%zu (needed=%zu)\n", new_size, needed_bytes);
  fflush(stderr);
}

void LlvmRuntimeExecutor::ensure_adstack_heap_float(std::size_t needed_bytes) {
  if (needed_bytes == 0 || needed_bytes <= adstack_heap_size_float_) {
    return;
  }
  // Mirror `ensure_adstack_heap`'s amortised-doubling growth and grow-on-demand semantics. The float heap is
  // allocated independently from the combined heap so a kernel with bound_expr tasks can shrink the combined
  // slice to int-only while still backing float allocas at `row_id_var * stride_float + float_offset`.
  std::size_t new_size = std::max(needed_bytes, std::size_t(2) * adstack_heap_size_float_);

  Device::AllocParams params{};
  params.size = new_size;
  params.host_read = false;
  params.host_write = false;
  params.export_sharing = false;
  params.usage = AllocUsage::Storage;
  DeviceAllocation new_alloc;
  RhiResult res = llvm_device()->allocate_memory(params, &new_alloc);
  QD_ERROR_IF(res != RhiResult::success,
              "Failed to allocate {} bytes for the adstack float heap (err: {}). Consider lowering "
              "`ad_stack_size` or the per-kernel reverse-mode adstack count.",
              new_size, int(res));
  void *new_ptr = get_device_alloc_info_ptr(new_alloc);
  auto new_guard = std::make_unique<DeviceAllocationGuard>(std::move(new_alloc));

  // Resolve and cache the field-of-LLVMRuntime addresses for the split-heap fields on first grow. The
  // `runtime_get_adstack_split_heap_field_ptrs` helper returns four addresses in fixed slot order: float-buffer-
  // ptr, float-size, int-buffer-ptr, int-size. We only consume the float pair here; the int half is reserved for
  // a future symmetric `ensure_adstack_heap_int` if it becomes useful (today the int allocas in bound_expr tasks
  // ride the combined heap with a smaller stride).
  if (runtime_adstack_heap_buffer_float_field_ptr_ == nullptr) {
    auto *const runtime_jit = get_runtime_jit_module();
    runtime_jit->call<void *>("runtime_get_adstack_split_heap_field_ptrs", llvm_runtime_);
    runtime_adstack_heap_buffer_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id, result_buffer_cache_));
    runtime_adstack_heap_size_float_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 1, result_buffer_cache_));
    runtime_adstack_heap_buffer_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 2, result_buffer_cache_));
    runtime_adstack_heap_size_int_field_ptr_ = quadrants_union_cast_with_different_sizes<void *>(
        fetch_result_uint64(quadrants_result_buffer_ret_value_id + 3, result_buffer_cache_));
  }
  uint64 size_u64 = static_cast<uint64>(new_size);
  if (config_.arch == Arch::cuda) {
#if defined(QD_WITH_CUDA)
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                     sizeof(void *));
    CUDADriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                     sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else if (config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_AMDGPU)
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_buffer_float_field_ptr_, &new_ptr,
                                                       sizeof(void *));
    AMDGPUDriver::get_instance().memcpy_host_to_device(runtime_adstack_heap_size_float_field_ptr_, &size_u64,
                                                       sizeof(uint64));
#else
    QD_NOT_IMPLEMENTED;
#endif
  } else {
    *reinterpret_cast<void **>(runtime_adstack_heap_buffer_float_field_ptr_) = new_ptr;
    *reinterpret_cast<uint64 *>(runtime_adstack_heap_size_float_field_ptr_) = size_u64;
  }

  adstack_heap_alloc_float_ = std::move(new_guard);
  adstack_heap_size_float_ = new_size;
  // DEBUG: LLVM-side float-heap allocation tracker. Mirrors the SPIR-V `[ADSTACK-FHEAP]` print so the user can
  // validate reducer-driven heap sizing on CUDA / AMDGPU. Revert before merging.
  fprintf(stderr, "[ADSTACK-FHEAP-LLVM] alloc new_size=%zu (needed=%zu)\n", new_size, needed_bytes);
  fflush(stderr);
}

void LlvmRuntimeExecutor::preallocate_runtime_memory() {
  if (preallocated_runtime_memory_allocs_ != nullptr)
    return;

  std::size_t total_prealloc_size = 0;
  const auto total_mem = llvm_device()->get_total_memory();
  if (config_.device_memory_fraction == 0) {
    QD_ASSERT(config_.device_memory_GB > 0);
    total_prealloc_size = std::size_t(config_.device_memory_GB * (1UL << 30));
  } else {
    total_prealloc_size = std::size_t(config_.device_memory_fraction * total_mem);
  }
  QD_ASSERT(total_prealloc_size <= total_mem);

  void *runtime_memory_prealloc_buffer = preallocate_memory(total_prealloc_size, preallocated_runtime_memory_allocs_);

  QD_TRACE("Allocating device memory {:.2f} MB", 1.0 * total_prealloc_size / (1UL << 20));

  auto *const runtime_jit = get_runtime_jit_module();
  runtime_jit->call<void *, std::size_t, void *>("runtime_initialize_memory", llvm_runtime_, total_prealloc_size,
                                                 runtime_memory_prealloc_buffer);
}

void LlvmRuntimeExecutor::materialize_runtime(KernelProfilerBase *profiler, uint64 **result_buffer_ptr) {
  // Starting random state for the program calculated using the random seed.
  // The seed is multiplied by 1048391 so that two programs with different seeds
  // will not have overlapping random states in any thread.
  int starting_rand_state = config_.random_seed * 1048391;

  // Number of random states. One per CPU/CUDA thread.
  int num_rand_states = 0;

  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_CUDA) || defined(QD_WITH_AMDGPU)
    // It is important to make sure that every CUDA thread has its own random
    // state so that we do not need expensive per-state locks.
    num_rand_states = config_.saturating_grid_dim * config_.max_block_dim;
#else
    QD_NOT_IMPLEMENTED
#endif
  } else {
    num_rand_states = config_.cpu_max_num_threads;
  }

  // The result buffer allocated here is only used for the launches of
  // runtime JIT functions. To avoid memory leak, we use the head of
  // the preallocated device buffer as the result buffer in
  // CUDA and AMDGPU backends.
  // | ==================preallocated device buffer ========================== |
  // |<- reserved for return ->|<---- usable for allocators on the device ---->|
  auto *const runtime_jit = get_runtime_jit_module();

  size_t runtime_objects_prealloc_size = 0;
  void *runtime_objects_prealloc_buffer = nullptr;
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
#if defined(QD_WITH_CUDA) || defined(QD_WITH_AMDGPU)
    auto [temp_result_alloc, res] = llvm_device()->allocate_memory_unique({sizeof(uint64_t)});
    QD_ERROR_IF(res != RhiResult::success, "Failed to allocate memory for `runtime_get_memory_requirements`");
    void *temp_result_ptr = llvm_device()->get_memory_addr(*temp_result_alloc);

    runtime_jit->call<void *, int32_t, int32_t>("runtime_get_memory_requirements", temp_result_ptr, num_rand_states,
                                                /*use_preallocated_buffer=*/1);
    runtime_objects_prealloc_size = size_t(fetch_result<uint64_t>(0, (uint64_t *)temp_result_ptr));
    temp_result_alloc.reset();
    size_t result_buffer_size = sizeof(uint64) * quadrants_result_buffer_entries;

    QD_TRACE("Allocating device memory {:.2f} MB",
             1.0 * (runtime_objects_prealloc_size + result_buffer_size) / (1UL << 20));

    runtime_objects_prealloc_buffer =
        preallocate_memory(iroundup(runtime_objects_prealloc_size + result_buffer_size, quadrants_page_size),
                           preallocated_runtime_objects_allocs_);

    *result_buffer_ptr = (uint64_t *)((uint8_t *)runtime_objects_prealloc_buffer + runtime_objects_prealloc_size);
#else
    QD_NOT_IMPLEMENTED
#endif
  } else {
    *result_buffer_ptr =
        (uint64 *)HostMemoryPool::get_instance().allocate(sizeof(uint64) * quadrants_result_buffer_entries, 8);
  }

  QD_TRACE("Launching runtime_initialize");

  auto *host_memory_pool = &HostMemoryPool::get_instance();
  runtime_jit->call<void *, void *, std::size_t, void *, int, void *, void *, void *>(
      "runtime_initialize", *result_buffer_ptr, host_memory_pool, runtime_objects_prealloc_size,
      runtime_objects_prealloc_buffer, num_rand_states, (void *)&host_allocate_aligned, (void *)std::printf,
      (void *)std::vsnprintf);

  QD_TRACE("LLVMRuntime initialized (excluding `root`)");
  llvm_runtime_ = fetch_result<void *>(quadrants_result_buffer_ret_value_id, *result_buffer_ptr);
  result_buffer_cache_ = *result_buffer_ptr;
  QD_TRACE("LLVMRuntime pointer fetched");

  // Preallocate for runtime memory and update to LLVMRuntime
  if (config_.arch == Arch::cuda || config_.arch == Arch::amdgpu) {
    if (!use_device_memory_pool()) {
      preallocate_runtime_memory();
    }
  }

  if (config_.arch == Arch::cuda) {
    QD_TRACE("Initializing {} random states using CUDA", num_rand_states);
    runtime_jit->launch<void *, int>("runtime_initialize_rand_states_cuda", config_.saturating_grid_dim,
                                     config_.max_block_dim, 0, llvm_runtime_, starting_rand_state);
  } else {
    QD_TRACE("Initializing {} random states (serially)", num_rand_states);
    runtime_jit->call<void *, int>("runtime_initialize_rand_states_serial", llvm_runtime_, starting_rand_state);
  }

  if (arch_use_host_memory(config_.arch)) {
    runtime_jit->call<void *, void *, void *>("LLVMRuntime_initialize_thread_pool", llvm_runtime_, thread_pool_.get(),
                                              (void *)ThreadPool::static_run);

    runtime_jit->call<void *, void *>("LLVMRuntime_set_assert_failed", llvm_runtime_, (void *)assert_failed_host);
  }
  if (arch_is_cpu(config_.arch) && (profiler != nullptr)) {
    // Profiler functions can only be called on CPU kernels
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler", llvm_runtime_, profiler);
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler_start", llvm_runtime_,
                                      (void *)&KernelProfilerBase::profiler_start);
    runtime_jit->call<void *, void *>("LLVMRuntime_set_profiler_stop", llvm_runtime_,
                                      (void *)&KernelProfilerBase::profiler_stop);
  }
}

void LlvmRuntimeExecutor::destroy_snode_tree(SNodeTree *snode_tree) {
  get_llvm_context()->delete_snode_tree(snode_tree->id());
  snode_tree_buffer_manager_->destroy(snode_tree);
}

Device *LlvmRuntimeExecutor::get_compute_device() {
  return device_.get();
}

LLVMRuntime *LlvmRuntimeExecutor::get_llvm_runtime() {
  return static_cast<LLVMRuntime *>(llvm_runtime_);
}

void LlvmRuntimeExecutor::init_runtime_jit_module(std::unique_ptr<llvm::Module> module) {
  llvm_context_->init_runtime_module(module.get());
  runtime_jit_module_ = create_jit_module(std::move(module));
}

}  // namespace quadrants::lang
