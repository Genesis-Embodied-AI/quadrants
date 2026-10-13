// Standalone process for the native Metal crash controls: no Python, Vulkan or Quadrants runtime.
#include "native_metal_heap.mm"
#include <cstdlib>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 3)
    return 64;
  int mode = std::atoi(argv[1]);
  int cycles = std::atoi(argv[2]);
  // 2: buffers and addresses, no queue. 3: empty heap. 4: buffers, no addresses or queue.
  // 5: buffers, addresses, queue, no command. 6: commit an empty command buffer and wait.
  // 7: blit encoder with no fills, commit, and wait.
  if ((mode < 2 || mode > 7) || cycles < 1 || cycles > 2000)
    return 64;
  if (native_install_crash_trace() != 0)
    return 65;
  std::printf("pid=%d mode=%d cycles=%d allocation_bytes=%llu delay_us=50000\n", getpid(), mode, cycles,
              static_cast<unsigned long long>(128ull << 20));
  std::fflush(stdout);
  for (int i = 0; i < cycles; ++i) {
    ProbeStats stats{};
    int result = native_metal_cycle(128ull << 20, mode, &stats);
    uint64_t expected_status = (mode == 6 || mode == 7) ? 4 : 0;
    if (result != 0 || stats.allocation_bytes < (128ull << 20) || stats.command_status != expected_status) {
      std::fprintf(stderr, "failed cycle=%d result=%d size=%llu command_status=%llu\n", i, result,
                   static_cast<unsigned long long>(stats.allocation_bytes),
                   static_cast<unsigned long long>(stats.command_status));
      return 66;
    }
    if (i % 25 == 0 || i == cycles - 1) {
      std::printf("completed=%d\n", i + 1);
      std::fflush(stdout);
    }
    // Pace allocation churn without touching a Python graph or collecting garbage.
    usleep(50000);
  }
  std::puts("completed all cycles");
  return 0;
}
