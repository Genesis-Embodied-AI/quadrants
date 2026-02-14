/*******************************************************************************
    Copyright (c) The Quadrants Authors (2016- ). All Rights Reserved.
    The use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "quadrants/common/core.h"
#include "quadrants/system/timer.h"

namespace quadrants {

class ProfilerRecords;

// Captures running time between the construction and destruction of the
// profiler instance
class ScopedProfiler {
 public:
  explicit ScopedProfiler(std::string name, uint64 elements = -1);

  void stop();

  static void enable();

  static void disable();

  ~ScopedProfiler();

 private:
  std::string name_;
  float64 start_time_;
  uint64 elements_;
  bool stopped_;
};

// A profiling system for multithreaded applications
class Profiling {
 public:
  void print_profile_info();
  void clear_profile_info();
  ProfilerRecords *get_this_thread_profiler();
  static Profiling &get_instance();

 private:
  std::mutex mut_;
  std::unordered_map<std::thread::id, ProfilerRecords *> profilers_;
};

#define QD_PROFILER(name) quadrants::ScopedProfiler _profiler_##__LINE__(name);

#define QD_AUTO_PROF QD_PROFILER(__FUNCTION__)

}  // namespace quadrants
