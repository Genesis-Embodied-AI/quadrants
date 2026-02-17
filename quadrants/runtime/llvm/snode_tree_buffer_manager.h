#pragma once
#include "quadrants/inc/constants.h"
#include "quadrants/struct/snode_tree.h"
#include "quadrants/rhi/public_device.h"
#define QD_RUNTIME_HOST

#include <set>

using Ptr = uint8_t *;

namespace quadrants::lang {

class JITModule;
class LlvmRuntimeExecutor;

class SNodeTreeBufferManager {
 public:
  explicit SNodeTreeBufferManager(LlvmRuntimeExecutor *runtime_exec);

  Ptr allocate(std::size_t size,
               const int snode_tree_id,
               uint64 *result_buffer);

  void destroy(SNodeTree *snode_tree);

 private:
  LlvmRuntimeExecutor *runtime_exec_;
  std::map<int, DeviceAllocation> snode_tree_id_to_device_alloc_;
};

}  // namespace quadrants::lang
