#include <functional>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include "quadrants/common/interface.h"
#include "quadrants/common/task.h"
#include "quadrants/system/benchmark.h"

namespace quadrants {

#define QD_INTERFACE_DEF_WITH_NANOBIND(class_name, base_alias)                                                 \
                                                                                                               \
  class InterfaceInjector_##class_name {                                                                       \
   public:                                                                                                     \
    explicit InterfaceInjector_##class_name(const std::string &name) {                                         \
      InterfaceHolder::get_instance()->register_registration_method(base_alias, [&](void *m) {                 \
        ((nanobind::module_ *)m)                                                                               \
            ->def("create_" base_alias, static_cast<std::shared_ptr<class_name> (*)(const std::string &name)>( \
                                            &create_instance<class_name>));                                    \
        ((nanobind::module_ *)m)                                                                               \
            ->def("create_initialized_" base_alias,                                                            \
                  static_cast<std::shared_ptr<class_name> (*)(const std::string &name, const Config &config)>( \
                      &create_instance<class_name>));                                                          \
      });                                                                                                      \
      InterfaceHolder::get_instance()->register_interface(                                                     \
          base_alias, (ImplementationHolderBase *)get_implementation_holder_instance_##class_name());          \
    }                                                                                                          \
  } ImplementationInjector_##base_class_name##class_name##instance(base_alias);

QD_INTERFACE_DEF_WITH_NANOBIND(Benchmark, "benchmark")
QD_INTERFACE_DEF_WITH_NANOBIND(Task, "task")

}  // namespace quadrants
