// Best-effort diagnostic only: native unwinding is not guaranteed async-signal-safe.
// Prewarm the unwinder before the workload; never attempt to recover from a fatal signal.
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

static void native_crash_trace(int signal_number) {
  constexpr char message[] = "Native Metal probe fatal signal; native stack follows:\n";
  (void)write(STDERR_FILENO, message, sizeof(message) - 1);
  void *frames[96];
  int count = backtrace(frames, 96);
  backtrace_symbols_fd(frames, count, STDERR_FILENO);
  _exit(128 + signal_number);
}

extern "C" int native_install_crash_trace() {
  void *frames[8];
  (void)backtrace(frames, 8);
  struct sigaction action{};
  action.sa_handler = native_crash_trace;
  sigemptyset(&action.sa_mask);
  action.sa_flags = SA_RESETHAND;
  for (int signal_number : {SIGSEGV, SIGBUS, SIGABRT}) {
    if (sigaction(signal_number, &action, nullptr) != 0)
      return 1;
  }
  return 0;
}
