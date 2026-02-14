#include "quadrants/ir/offloaded_task_type.h"

namespace quadrants::lang {

std::string offloaded_task_type_name(OffloadedTaskType tt) {
  if (false) {
  }
#define PER_TASK_TYPE(x) else if (tt == OffloadedTaskType::x) return #x;
#include "quadrants/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
  else
    QD_NOT_IMPLEMENTED
}

}  // namespace quadrants::lang
