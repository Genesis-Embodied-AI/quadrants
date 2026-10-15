#include "quadrants/runtime/gfx/runtime.h"
#include "quadrants/program/program.h"
#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang::gfx {
std::vector<PerTaskAdStackRuntime> GfxRuntime::publish_adstack_metadata_spirv(
    LaunchContextBuilder &host_ctx, const std::vector<spirv::TaskAttributes> &task_attribs) {
  std::vector<PerTaskAdStackRuntime> per_task_ad_stack(task_attribs.size());
  std::vector<size_t> adstack_task_indices;
  for (size_t i = 0; i < task_attribs.size(); ++i)
    if (!task_attribs[i].ad_stack.allocas.empty())
      adstack_task_indices.push_back(i);
  if (adstack_task_indices.empty())
    return per_task_ad_stack;
  QD_ERROR_IF(!host_ctx.get_context().adstack_capacity, "Stack-based backward requires a counted forward inside qd.adstack");

    const uint64_t capacity = host_ctx.get_context().adstack_capacity;
    QD_ERROR_IF(capacity > INT32_MAX, "Measured SPIR-V adstack capacity exceeds the supported limit");
    for (auto ti : adstack_task_indices) {
      const auto &allocas = task_attribs[ti].ad_stack.allocas;
      auto &rt = per_task_ad_stack[ti];
      rt.metadata.assign(3 + 2 * allocas.size(), 0);
      uint64_t float_offset = 0, int_offset = 0;
      for (size_t i = 0; i < allocas.size(); ++i) {
        bool is_float = allocas[i].heap_kind == spirv::TaskAttributes::AdStackAllocaAttribs::HeapKind::Float;
        auto &offset = is_float ? float_offset : int_offset;
        QD_ERROR_IF(offset + capacity * (is_float ? 2 : 1) > UINT32_MAX, "Measured SPIR-V adstack stride overflow");
        rt.metadata[2 + 2 * i] = static_cast<uint32_t>(offset);
        rt.metadata[3 + 2 * i] = static_cast<uint32_t>(capacity);
        offset += capacity * (is_float ? 2 : 1);
      }
      rt.stride_float = rt.metadata[0] = static_cast<uint32_t>(float_offset);
      rt.stride_int = rt.metadata[1] = static_cast<uint32_t>(int_offset);
    }
  return per_task_ad_stack;
}
}  // namespace quadrants::lang::gfx
