#include "quadrants/runtime/amdgpu/kernel_launcher.h"
#include "quadrants/rhi/amdgpu/amdgpu_context.h"
#include "quadrants/program/launch_context_builder.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace quadrants::lang {
namespace amdgpu {

namespace exp12_diag {
static const bool diag_enabled = []() {
  const char *flag = std::getenv("QD_EXP12_DIAG_ON");
  return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}();

struct KernelBranchCounts {
  std::atomic<long> launches{0};
  std::atomic<long> kNone_on_device{0};
  std::atomic<long> kNone_host_copy{0};
  std::atomic<long> kNdarray_passthrough{0};
  std::atomic<long> skip{0};
};
static std::mutex stats_mutex;
static std::unordered_map<std::string, std::unique_ptr<KernelBranchCounts>>
    stats;
static std::atomic<bool> atexit_registered{false};

static void flush_to_side_log() {
  const char *out_path = std::getenv("QD_EXP12_DIAG_OUT");
  if (!out_path) {
    out_path = "/tmp/exp12_diag.csv";
  }
  std::ofstream out(out_path);
  if (!out.is_open()) {
    return;
  }
  out << "kernel_name,launches,kNone_on_device,kNone_host_copy,"
      << "kNdarray_passthrough,skip\n";
  std::lock_guard<std::mutex> lock(stats_mutex);
  for (const auto &kv : stats) {
    const auto &c = *kv.second;
    out << '"' << kv.first << '"' << ',' << c.launches.load() << ','
        << c.kNone_on_device.load() << ',' << c.kNone_host_copy.load() << ','
        << c.kNdarray_passthrough.load() << ',' << c.skip.load() << '\n';
  }
}

static KernelBranchCounts &get_or_create(const std::string &name) {
  std::lock_guard<std::mutex> lock(stats_mutex);
  if (!atexit_registered.exchange(true)) {
    std::atexit(flush_to_side_log);
  }
  auto it = stats.find(name);
  if (it == stats.end()) {
    it = stats.emplace(name, std::make_unique<KernelBranchCounts>()).first;
  }
  return *it->second;
}
}  // namespace exp12_diag

void KernelLauncher::launch_offloaded_tasks(
    LaunchContextBuilder &ctx,
    JITModule *amdgpu_module,
    const std::vector<OffloadedTask> &offloaded_tasks) {
  constexpr int kRuntimeContextArgSize = sizeof(RuntimeContext);
  for (const auto &task : offloaded_tasks) {
    QD_TRACE("Launching kernel {}<<<{}, {}>>>", task.name, task.grid_dim,
             task.block_dim);
    amdgpu_module->launch(task.name, task.grid_dim, task.block_dim,
                          task.dynamic_shared_array_bytes,
                          {&ctx.get_context()}, {kRuntimeContextArgSize});
  }
}

void KernelLauncher::launch_offloaded_tasks_with_do_while(
    LaunchContextBuilder &ctx,
    JITModule *amdgpu_module,
    const std::vector<OffloadedTask> &offloaded_tasks) {
  int32_t counter_val;
  do {
    launch_offloaded_tasks(ctx, amdgpu_module, offloaded_tasks);
    counter_val = 0;
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    AMDGPUDriver::get_instance().memcpy_device_to_host(
        &counter_val, ctx.graph_do_while_flag_dev_ptr, sizeof(int32_t));
  } while (counter_val != 0);
}

bool KernelLauncher::on_amdgpu_device(void *ptr) {
  unsigned int attr_val[8];
  // mem_get_attribute doesn't work well on ROCm
  uint32_t ret_code =
      AMDGPUDriver::get_instance().mem_get_attributes.call(attr_val, ptr);

  return ret_code == HIP_SUCCESS && attr_val[0] == HIP_MEMORYTYPE_DEVICE;
}

// Hot path. Uses cached per-handle device scratch buffers, async H2D/D2H,
// and lazy result-buffer materialisation to eliminate per-launch
// hipMallocAsync / hipFreeAsync overhead. See device_scratch_buffer.h
// for the full motivation.
void KernelLauncher::launch_llvm_kernel(Handle handle,
                                        LaunchContextBuilder &ctx) {
  QD_ASSERT(handle.get_launch_id() < contexts_.size());
  auto &launcher_ctx = contexts_[handle.get_launch_id()];
  auto *executor = get_runtime_executor();
  auto *amdgpu_module = launcher_ctx.jit_module;
  const auto &parameters = *launcher_ctx.parameters;
  const auto &offloaded_tasks = launcher_ctx.offloaded_tasks;

  AMDGPUContext::get_instance().make_current();
  ctx.get_context().runtime = executor->get_llvm_runtime();

  exp12_diag::KernelBranchCounts *branch_counts = nullptr;
  if (exp12_diag::diag_enabled && !offloaded_tasks.empty()) {
    branch_counts = &exp12_diag::get_or_create(offloaded_tasks.front().name);
    branch_counts->launches.fetch_add(1, std::memory_order_relaxed);
  }

  // Change from std::vector<int> to ArgArrayPtrKey
  std::unordered_map<ArgArrayPtrKey, std::pair<void *, DeviceAllocation>,
                     ArgArrayPtrKeyHasher>
      transfers;
  std::unordered_map<ArgArrayPtrKey, void *, ArgArrayPtrKeyHasher> device_ptrs;

  char *device_result_buffer = nullptr;

  for (int i = 0; i < (int)parameters.size(); i++) {
    const auto &kv = parameters[i];
    const auto &arg_id = kv.first;
    const auto &parameter = kv.second;
    if (parameter.is_array) {
      const auto arr_sz = ctx.array_runtime_sizes[arg_id];
      if (arr_sz == 0) {
        if (branch_counts) {
          branch_counts->skip.fetch_add(1, std::memory_order_relaxed);
        }
        continue;
      }

      ArgArrayPtrKey data_ptr_idx{arg_id, TypeFactory::DATA_PTR_POS_IN_NDARRAY};
      ArgArrayPtrKey grad_ptr_idx{arg_id, TypeFactory::GRAD_PTR_POS_IN_NDARRAY};
      auto data_ptr = ctx.array_ptrs[data_ptr_idx];

      if (ctx.device_allocation_type[arg_id] ==
          LaunchContextBuilder::DevAllocType::kNone) {
        if (on_amdgpu_device(data_ptr)) {
          if (branch_counts) {
            branch_counts->kNone_on_device.fetch_add(
                1, std::memory_order_relaxed);
          }
          device_ptrs[data_ptr_idx] = data_ptr;
        } else {
          if (branch_counts) {
            branch_counts->kNone_host_copy.fetch_add(
                1, std::memory_order_relaxed);
          }
          device_result_buffer = launcher_ctx.device_result_buffer.ensure(sizeof(uint64));
          DeviceAllocation devalloc = executor->allocate_memory_on_device(
              arr_sz, (uint64 *)device_result_buffer);
          device_ptrs[data_ptr_idx] =
              executor->get_device_alloc_info_ptr(devalloc);
          transfers[data_ptr_idx] = {data_ptr, devalloc};

          AMDGPUDriver::get_instance().memcpy_host_to_device(
              (void *)device_ptrs[data_ptr_idx], data_ptr, arr_sz);
        }
        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      } else if (arr_sz > 0) {  // why use arr_sz constrain?
        if (branch_counts) {
          branch_counts->kNdarray_passthrough.fetch_add(
              1, std::memory_order_relaxed);
        }
        // Ndarray
        DeviceAllocation *ptr = static_cast<DeviceAllocation *>(data_ptr);
        // Unwrapped raw ptr on device
        device_ptrs[data_ptr_idx] = executor->get_device_alloc_info_ptr(*ptr);

        ctx.set_ndarray_ptrs(arg_id, (uint64)device_ptrs[data_ptr_idx],
                             (uint64)ctx.array_ptrs[grad_ptr_idx]);
        if (arg_id == ctx.graph_do_while_arg_id) {
          ctx.graph_do_while_flag_dev_ptr = device_ptrs[data_ptr_idx];
        }
      }
    }
  }
  // No pre-kernel sync needed: transfer H2Ds above are synchronous, and
  // the arg-buffer async H2D below is stream-ordered with the kernel.
  char *host_result_buffer = (char *)ctx.get_context().result_buffer;
  if (ctx.result_buffer_size > 0) {
    device_result_buffer = launcher_ctx.device_result_buffer.ensure(std::max(ctx.result_buffer_size, sizeof(uint64)));
    ctx.get_context().result_buffer = (uint64 *)device_result_buffer;
  }
  if (ctx.arg_buffer_size > 0) {
    char *device_arg_buffer =
        launcher_ctx.device_arg_buffer.ensure(ctx.arg_buffer_size);
    AMDGPUDriver::get_instance().memcpy_host_to_device_async(
        device_arg_buffer, ctx.get_context().arg_buffer, ctx.arg_buffer_size,
        nullptr);
    ctx.get_context().arg_buffer = device_arg_buffer;
  }

  if (ctx.graph_do_while_arg_id >= 0) {
    QD_ASSERT(ctx.graph_do_while_flag_dev_ptr);
    launch_offloaded_tasks_with_do_while(ctx, amdgpu_module, offloaded_tasks);
  } else {
    launch_offloaded_tasks(ctx, amdgpu_module, offloaded_tasks);
  }
  QD_TRACE("Launching kernel");
  bool needs_sync = false;
  if (ctx.result_buffer_size > 0) {
    AMDGPUDriver::get_instance().memcpy_device_to_host_async(
        host_result_buffer, device_result_buffer, ctx.result_buffer_size,
        nullptr);
    // Caller reads result_buffer via get_ret() with no further sync;
    // hipMemcpyDtoHAsync may return before the host sees the copy, so
    // force a sync below regardless of pageable/pinned destination.
    needs_sync = true;
  }
  if (!transfers.empty() || needs_sync) {
    AMDGPUDriver::get_instance().stream_synchronize(nullptr);
    for (auto itr = transfers.begin(); itr != transfers.end(); itr++) {
      auto &idx = itr->first;
      auto arg_id = idx.arg_id;
      AMDGPUDriver::get_instance().memcpy_device_to_host(
          itr->second.first, (void *)device_ptrs[idx],
          ctx.array_runtime_sizes[arg_id]);
      executor->deallocate_memory_on_device(itr->second.second);
    }
  }
}

KernelLauncher::Handle KernelLauncher::register_llvm_kernel(
    const LLVM::CompiledKernelData &compiled) {
  QD_ASSERT(compiled.arch() == Arch::amdgpu);

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

}  // namespace amdgpu
}  // namespace quadrants::lang

