#include <cstdlib>

inline bool is_ci() {
  char *res = std::getenv("QD_CI");
  if (res == nullptr)
    return false;
  return std::stoi(res);
}
