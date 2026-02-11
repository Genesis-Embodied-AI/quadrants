#pragma once

#include "quadrants/common/core.h"

#include <string>

namespace quadrants::lang {

enum class OffloadedTaskType : int {
#define PER_TASK_TYPE(x) x,
#include "quadrants/inc/offloaded_task_type.inc.h"
#undef PER_TASK_TYPE
};

std::string offloaded_task_type_name(OffloadedTaskType tt);

}  // namespace quadrants::lang
