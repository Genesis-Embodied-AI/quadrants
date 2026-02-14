// Definitions of utility functions and enums
#pragma once

#include "quadrants/util/io.h"
#include "quadrants/common/core.h"
#include "quadrants/system/profiler.h"
#include "quadrants/common/exceptions.h"
#include "quadrants/ir/stmt_op_types.h"
#include "quadrants/ir/type.h"
#include "quadrants/ir/type_utils.h"
#include "quadrants/ir/type_factory.h"

namespace quadrants::lang {

std::string get_custom_cuda_library_path();

real get_cpu_frequency();

extern real default_measurement_time;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second = default_measurement_time);

class LaunchContextBuilder;

using FunctionType = std::function<void(LaunchContextBuilder &)>;

inline std::string make_list(const std::vector<std::string> &data,
                             std::string bracket = "") {
  std::string ret = bracket;
  for (int i = 0; i < (int)data.size(); i++) {
    ret += data[i];
    if (i + 1 < (int)data.size()) {
      ret += ", ";
    }
  }
  if (bracket == "<") {
    ret += ">";
  } else if (bracket == "{") {
    ret += "}";
  } else if (bracket == "[") {
    ret += "]";
  } else if (bracket == "(") {
    ret += ")";
  } else if (bracket != "") {
    QD_P(bracket);
    QD_NOT_IMPLEMENTED
  }
  return ret;
}

template <typename T>
std::string make_list(const std::vector<T> &data,
                      std::function<std::string(const T &t)> func,
                      std::string bracket = "") {
  std::vector<std::string> ret(data.size());
  for (int i = 0; i < (int)data.size(); i++) {
    ret[i] = func(data[i]);
  }
  return make_list(ret, bracket);
}

extern std::string compiled_lib_dir;
extern std::string runtime_tmp_dir;
std::string runtime_lib_dir();

bool command_exist(const std::string &command);

}  // namespace quadrants::lang

namespace quadrants {
void initialize_benchmark();

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(std::function<T(Args...)>) {
  return nullptr;
}

template <typename T, typename... Args, typename FP = T (*)(Args...)>
FP function_pointer_helper(T (*)(Args...)) {
  return nullptr;
}

template <typename T>
using function_pointer_type =
    decltype(function_pointer_helper(std::declval<T>()));

}  // namespace quadrants
