#pragma once

#include <exception>
#include <string>
#include <string_view>
#include "quadrants/common/logging.h"

namespace quadrants::lang {

class IRModified {};

struct Location {
  int line_number = 0;
  std::string var_name = "";
};

struct DebugInfo {
  Location src_loc;
  std::string tb;

  explicit DebugInfo() = default;

  explicit DebugInfo(std::string tb_) : tb(tb_) {
  }

  explicit DebugInfo(const char *tb_) : tb(tb_) {
  }

  std::string get_last_tb() const {
    return tb;
  }

  std::string const &get_tb() const {
    return tb;
  }

  void set_tb(std::string const &tb) {
    this->tb = tb;
  }
};

class QuadrantsExceptionImpl : public std::exception {
  friend struct ErrorEmitter;

 protected:
  std::string msg_;

 public:
  // Add default constructor to allow passing Exception to ErrorEmitter
  // TODO: remove this and find a better way
  explicit QuadrantsExceptionImpl() = default;
  explicit QuadrantsExceptionImpl(const std::string msg) : msg_(msg) {
  }
  const char *what() const noexcept override {
    return msg_.c_str();
  }
};

class QuadrantsError : public QuadrantsExceptionImpl {
  using QuadrantsExceptionImpl::QuadrantsExceptionImpl;
};

class QuadrantsWarning : public QuadrantsExceptionImpl {
  using QuadrantsExceptionImpl::QuadrantsExceptionImpl;

 protected:
  static constexpr std::string_view name_ = "QuadrantsWarning";

 public:
  void emit() {
    quadrants::Logger::get_instance().warn(std::string(name_) + "\n" + msg_);
  }
};

class QuadrantsTypeError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsSyntaxError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsIndexError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsRuntimeError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsAssertionError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsIrError : public QuadrantsError {
  using QuadrantsError::QuadrantsError;
};

class QuadrantsCastWarning : public QuadrantsWarning {
  using QuadrantsWarning::QuadrantsWarning;
  static constexpr std::string_view name_ = "QuadrantsCastWarning";
};

class QuadrantsTypeWarning : public QuadrantsWarning {
  using QuadrantsWarning::QuadrantsWarning;
  static constexpr std::string_view name_ = "QuadrantsTypeWarning";
};

class QuadrantsIrWarning : public QuadrantsWarning {
  using QuadrantsWarning::QuadrantsWarning;
  static constexpr std::string_view name_ = "QuadrantsIrWarning";
};

class QuadrantsIndexWarning : public QuadrantsWarning {
  using QuadrantsWarning::QuadrantsWarning;
  static constexpr std::string_view name_ = "QuadrantsIndexWarning";
};

class QuadrantsRuntimeWarning : public QuadrantsWarning {
  using QuadrantsWarning::QuadrantsWarning;
  static constexpr std::string_view name_ = "QuadrantsRuntimeWarning";
};

struct ErrorEmitter {
  ErrorEmitter() = delete;
  ErrorEmitter(ErrorEmitter &) = delete;
  ErrorEmitter(ErrorEmitter &&) = delete;

  // Emit an error on stmt with error message
  template <typename E,
            typename = std::enable_if_t<
                std::is_base_of_v<QuadrantsExceptionImpl, std::decay_t<E>>>,
            // The expected type for T is `Stmt`, `Expression`, or `DebugInfo`.
            // These types have a member function named get_tb() that returns
            // trace back information as a `std::string`.
            typename T,
            typename = std::enable_if_t<std::is_same_v<
                std::decay_t<decltype(std::declval<T>()->get_tb())>,
                std::string>>>
  ErrorEmitter(E &&error, T p_dbg_info, std::string &&error_msg) {
    if constexpr ((std::is_same_v<std::decay_t<T>, DebugInfo *> ||
                   std::is_same_v<std::decay_t<T>, const DebugInfo *>) &&
                  std::is_base_of_v<QuadrantsError, std::decay_t<E>>) {
      // Indicates a failed C++ API call from Python side, we should not print
      // tb here
      error.msg_ = error_msg;
    } else {
      error.msg_ = p_dbg_info->get_last_tb() + error_msg;
    }

    if constexpr (std::is_base_of_v<QuadrantsWarning, std::decay_t<E>>) {
      error.emit();
    } else if constexpr (std::is_base_of_v<QuadrantsError, std::decay_t<E>>) {
      throw std::move(error);
    } else {
      TI_NOT_IMPLEMENTED;
    }
  }
};

}  // namespace quadrants::lang
