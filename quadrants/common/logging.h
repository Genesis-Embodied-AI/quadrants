#pragma once

#include <functional>
#include <cstring>

// This is necessary for QD_UNREACHABLE
#include "quadrants/common/platform_macros.h"

// Must include "spdlog/common.h" to define SPDLOG_HEADER_ONLY
// before including "spdlog/fmt/fmt.h"
#include "spdlog/common.h"
#include "spdlog/fmt/fmt.h"
namespace spdlog {
class logger;
}

#ifdef _WIN64
#define __FILENAME__ \
  (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__ \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define SPD_AUGMENTED_LOG(X, ...)                                        \
  quadrants::Logger::get_instance().X(                                   \
      fmt::format("[{}:{}@{}] ", __FILENAME__, __FUNCTION__, __LINE__) + \
      fmt::format(__VA_ARGS__))

#if defined(QD_PLATFORM_WINDOWS)
#define QD_UNREACHABLE __assume(0);
#else
#define QD_UNREACHABLE __builtin_unreachable();
#endif

#define QD_TRACE(...) SPD_AUGMENTED_LOG(trace, __VA_ARGS__)
#define QD_DEBUG(...) SPD_AUGMENTED_LOG(debug, __VA_ARGS__)
#define QD_INFO(...) SPD_AUGMENTED_LOG(info, __VA_ARGS__)
#define QD_WARN(...) SPD_AUGMENTED_LOG(warn, __VA_ARGS__)
#define QD_ERROR(...)                      \
  {                                        \
    SPD_AUGMENTED_LOG(error, __VA_ARGS__); \
    QD_UNREACHABLE;                        \
  }
#define QD_CRITICAL(...)                      \
  {                                           \
    SPD_AUGMENTED_LOG(critical, __VA_ARGS__); \
    QD_UNREACHABLE;                           \
  }

#define QD_TRACE_IF(condition, ...) \
  if (condition) {                  \
    QD_TRACE(__VA_ARGS__);          \
  }
#define QD_TRACE_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    QD_TRACE(__VA_ARGS__);              \
  }
#define QD_DEBUG_IF(condition, ...) \
  if (condition) {                  \
    QD_DEBUG(__VA_ARGS__);          \
  }
#define QD_DEBUG_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    QD_DEBUG(__VA_ARGS__);              \
  }
#define QD_INFO_IF(condition, ...) \
  if (condition) {                 \
    QD_INFO(__VA_ARGS__);          \
  }
#define QD_INFO_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    QD_INFO(__VA_ARGS__);              \
  }
#define QD_WARN_IF(condition, ...) \
  if (condition) {                 \
    QD_WARN(__VA_ARGS__);          \
  }
#define QD_WARN_UNLESS(condition, ...) \
  if (!(condition)) {                  \
    QD_WARN(__VA_ARGS__);              \
  }
#define QD_ERROR_IF(condition, ...) \
  if (condition) {                  \
    QD_ERROR(__VA_ARGS__);          \
  }
#define QD_ERROR_UNLESS(condition, ...) \
  if (!(condition)) {                   \
    QD_ERROR(__VA_ARGS__);              \
  }
#define QD_CRITICAL_IF(condition, ...) \
  if (condition) {                     \
    QD_CRITICAL(__VA_ARGS__);          \
  }
#define QD_CRITICAL_UNLESS(condition, ...) \
  if (!(condition)) {                      \
    QD_CRITICAL(__VA_ARGS__);              \
  }

#define QD_ASSERT(x) QD_ASSERT_INFO((x), "Assertion failure: " #x)
#define QD_ASSERT_INFO(x, ...)             \
  {                                        \
    bool ___ret___ = static_cast<bool>(x); \
    if (!___ret___) {                      \
      QD_ERROR(__VA_ARGS__);               \
    }                                      \
  }
#define QD_NOT_IMPLEMENTED QD_ERROR("Not supported.");

#define QD_STOP QD_ERROR("Stopping here")
#define QD_TAG QD_INFO("Tagging here")

#define QD_LOG_SET_PATTERN(x) spdlog::set_pattern(x);

#define QD_FLUSH_LOGGER                        \
  {                                            \
    quadrants::Logger::get_instance().flush(); \
  };

#define QD_P(x)                                                   \
  {                                                               \
    QD_INFO("{}", quadrants::TextSerializer::serialize(#x, (x))); \
  }

namespace quadrants {

class QD_DLL_EXPORT Logger {
 private:
  std::shared_ptr<spdlog::logger> console_;
  int level_;
  std::function<void()> print_stacktrace_fn_;

  Logger();

 public:
  void trace(const std::string &s);
  void debug(const std::string &s);
  void info(const std::string &s);
  void warn(const std::string &s);
  void error(const std::string &s, bool raise_exception = true);
  void critical(const std::string &s);
  void flush();
  void set_level(const std::string &level);
  bool is_level_effective(const std::string &level_name);
  int get_level();
  static int level_enum_from_string(const std::string &level);
  void set_level_default();

  // This is mostly to decouple the implementation.
  void set_print_stacktrace_func(std::function<void()> print_fn);

  static Logger &get_instance();
};

}  // namespace quadrants
