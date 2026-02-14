#include <csignal>

#include "quadrants/common/logging.h"
#include "quadrants/system/hacked_signal_handler.h"
#include "quadrants/system/threading.h"
#include "quadrants/system/traceback.h"

namespace quadrants {
namespace {

std::string signal_name(int sig) {
#if !defined(_WIN64)
  return strsignal(sig);
#else
  if (sig == SIGABRT) {
    return "SIGABRT";
  } else if (sig == SIGFPE) {
    return "SIGFPE";
  } else if (sig == SIGILL) {
    return "SIGFPE";
  } else if (sig == SIGSEGV) {
    return "SIGSEGV";
  } else if (sig == SIGTERM) {
    return "SIGTERM";
  } else {
    return "SIGNAL-Unknown";
  }
#endif
}

void signal_handler(int signo) {
  // It seems that there's no way to pass exception to Python in signal
  // handlers?
  // @archibate found that in fact there are such solution:
  // https://docs.python.org/3/library/faulthandler.html#module-faulthandler
  auto sig_name = signal_name(signo);
  Logger::get_instance().error(
      fmt::format("Received signal {} ({})", signo, sig_name), false);
  exit(-1);
  QD_UNREACHABLE;
}

}  // namespace

HackedSignalRegister::HackedSignalRegister() {
#define QD_REGISTER_SIGNAL_HANDLER(name, handler)                   \
  {                                                                 \
    if (std::signal(name, handler) == SIG_ERR)                      \
      std::printf("Cannot register signal handler for" #name "\n"); \
  }

  QD_REGISTER_SIGNAL_HANDLER(SIGSEGV, signal_handler);
  QD_REGISTER_SIGNAL_HANDLER(SIGABRT, signal_handler);
#if !defined(_WIN64)
  QD_REGISTER_SIGNAL_HANDLER(SIGBUS, signal_handler);
#endif
  QD_REGISTER_SIGNAL_HANDLER(SIGFPE, signal_handler);

#undef QD_REGISTER_SIGNAL_HANDLER

  Logger::get_instance().set_print_stacktrace_func(print_traceback);
  QD_TRACE("Quadrants signal handlers registered. Thread ID = {}",
           PID::get_pid());
}

HackedSignalRegister::~HackedSignalRegister() {
#define QD_UNREGISTER_SIGNAL_HANDLER(name)                            \
  {                                                                   \
    if (std::signal(name, SIG_DFL) == SIG_ERR)                        \
      std::printf("Cannot unregister signal handler for" #name "\n"); \
  }

  QD_UNREGISTER_SIGNAL_HANDLER(SIGSEGV);
  QD_UNREGISTER_SIGNAL_HANDLER(SIGABRT);
#if !defined(_WIN64)
  QD_UNREGISTER_SIGNAL_HANDLER(SIGBUS);
#endif
  QD_UNREGISTER_SIGNAL_HANDLER(SIGFPE);

#undef QD_UNREGISTER_SIGNAL_HANDLER
  QD_TRACE("Quadrants signal handlers unregistered. Thread ID = {}",
           PID::get_pid());
}

}  // namespace quadrants
