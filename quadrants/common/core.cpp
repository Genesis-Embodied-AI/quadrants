/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#include "quadrants/common/core.h"
#include "quadrants/common/version.h"
#include "quadrants/common/commit_hash.h"

#include <spdlog/fmt/fmt.h>
#include <cstdlib>
#include "quadrants/common/logging.h"

#if defined(QD_PLATFORM_WINDOWS)
#include "quadrants/platform/windows/windows.h"
#else
// Mac and Linux
#include <unistd.h>
#endif

namespace quadrants {

std::string python_package_dir;

std::string get_python_package_dir() {
  return python_package_dir;
}

void set_python_package_dir(const std::string &dir) {
  python_package_dir = dir;
}

std::string get_repo_dir() {
#if defined(QD_PLATFORM_WINDOWS)
  return "C:/quadrants_cache/";
#else
  auto xdg_cache = std::getenv("XDG_CACHE_HOME");

  std::string xdg_cache_str;
  if (xdg_cache != nullptr) {
    xdg_cache_str = xdg_cache;
  } else {
    // XDG_CACHE_HOME is not defined, defaulting to ~/.cache
    auto home = std::getenv("HOME");
    QD_ASSERT(home != nullptr);
    xdg_cache_str = home;
    xdg_cache_str += "/.cache";
  }

  return xdg_cache_str + "/quadrants/";
#endif
}

CoreState &CoreState::get_instance() {
  static CoreState state;
  return state;
}

int __trash__;

std::string get_version_string() {
  return fmt::format("{}.{}.{}", get_version_major(), get_version_minor(),
                     get_version_patch());
}

int get_version_major() {
  return QD_VERSION_MAJOR;
}

int get_version_minor() {
  return QD_VERSION_MINOR;
}

int get_version_patch() {
  return QD_VERSION_PATCH;
}

std::string get_commit_hash() {
  return QD_COMMIT_HASH;
}

int PID::get_pid() {
#if defined(QD_PLATFORM_WINDOWS)
  return (int)GetCurrentProcessId();
#else
  return (int)getpid();
#endif
}

int PID::get_parent_pid() {
#if defined(QD_PLATFORM_WINDOWS)
  QD_NOT_IMPLEMENTED
  return -1;
#else
  return (int)getppid();
#endif
}

}  // namespace quadrants
