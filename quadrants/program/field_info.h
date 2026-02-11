#pragma once
#include "quadrants/ir/type_utils.h"
#include "quadrants/ir/snode.h"
#include "quadrants/rhi/device.h"
#include "quadrants/program/program.h"

namespace quadrants {

namespace ui {

enum class FieldSource : int {
  QuadrantsNDarray = 0,
  HostMappedPtr = 1,
};

#define DEFINE_PROPERTY(Type, name)       \
  Type name;                              \
  void set_##name(const Type &new_name) { \
    name = new_name;                      \
  }                                       \
  Type get_##name() {                     \
    return name;                          \
  }

struct FieldInfo {
  DEFINE_PROPERTY(bool, valid)
  DEFINE_PROPERTY(std::vector<int>, shape);
  DEFINE_PROPERTY(uint64_t, num_elements);
  DEFINE_PROPERTY(FieldSource, field_source);
  DEFINE_PROPERTY(quadrants::lang::DataType, dtype);
  DEFINE_PROPERTY(quadrants::lang::DeviceAllocation, dev_alloc);

  FieldInfo() {
    valid = false;
  }
};

quadrants::lang::DevicePtr get_device_ptr(quadrants::lang::Program *program,
                                          quadrants::lang::SNode *snode);

}  // namespace ui

}  // namespace quadrants
