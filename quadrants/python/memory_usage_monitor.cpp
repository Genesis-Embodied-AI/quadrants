#include "quadrants/python/memory_usage_monitor.h"

#include <fstream>
#include <string>

#include "quadrants/common/core.h"
#include "quadrants/common/task.h"
#include "quadrants/math/scalar.h"
#include "quadrants/system/threading.h"
#include "quadrants/system/timer.h"

#if defined(__linux__)
#include <unistd.h>
#endif

namespace quadrants {

float64 bytes_to_GB(float64 bytes) {
  return float64(bytes) * pow<3>(1.0_f64 / 1024.0_f64);
}

namespace {
// Resident set size in bytes for the given pid. Read directly from /proc on Linux. nanobind does not
// support embedding a Python interpreter (which the previous pybind11 implementation used via psutil),
// so the RSS is now sourced from the kernel. Returns 0 where unavailable (e.g. non-Linux platforms).
uint64 read_rss_bytes(int pid) {
#if defined(__linux__)
  std::ifstream f("/proc/" + std::to_string(pid) + "/statm");
  if (!f.good()) {
    return 0;
  }
  uint64 size_pages = 0, resident_pages = 0;
  f >> size_pages >> resident_pages;
  if (!f) {
    return 0;
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  return resident_pages * static_cast<uint64>(page_size);
#else
  (void)pid;
  return 0;
#endif
}
}  // namespace

float64 get_memory_usage_gb(int pid) {
  return bytes_to_GB(get_memory_usage(pid));
}

uint64 get_memory_usage(int pid) {
  if (pid == -1) {
    pid = PID::get_pid();
  }
  return read_rss_bytes(pid);
}

MemoryMonitor::MemoryMonitor(int pid, std::string output_fn) {
  log_.open(output_fn, std::ios_base::out);
  pid_ = pid;
}

MemoryMonitor::~MemoryMonitor() {
}

uint64 MemoryMonitor::get_usage() const {
  return read_rss_bytes(pid_);
}

void MemoryMonitor::append_sample() {
  auto t = std::chrono::system_clock::now();
  log_ << fmt::format("{:.5f} {}\n", (t.time_since_epoch() / std::chrono::nanoseconds(1)) / 1e9_f64, get_usage());
  log_.flush();
}

void start_memory_monitoring(std::string output_fn, int pid, real interval) {
  if (pid == -1) {
    pid = PID::get_pid();
  }
  QD_P(pid);
  std::thread th([=]() {
    MemoryMonitor monitor(pid, output_fn);
    while (true) {
      monitor.append_sample();
      Time::sleep(interval);
    }
  });
  th.detach();
}

class MemoryTest : public Task {
 public:
  std::string run(const std::vector<std::string> &parameters) override {
    QD_P(get_memory_usage());
    Time::sleep(3);
    std::vector<uint8> a(1024ul * 1024 * 1024 * 10, 3);
    QD_P(get_memory_usage());
    Time::sleep(3);
    return "";
  }
};

class MemoryTest2 : public Task {
 public:
  std::string run(const std::vector<std::string> &parameters) override {
    start_memory_monitoring("test.txt");
    std::vector<uint8> a;
    for (int i = 0; i < 10; i++) {
      a.resize(1024ul * 1024 * 1024 * i / 2);
      std::fill(std::begin(a), std::end(a), 3);
      Time::sleep(0.5);
    }
    return "";
  }
};

QD_IMPLEMENTATION(Task, MemoryTest, "mem_test");
QD_IMPLEMENTATION(Task, MemoryTest2, "mem_test2");

}  // namespace quadrants
