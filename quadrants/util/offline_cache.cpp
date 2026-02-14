#include "quadrants/util/offline_cache.h"

namespace quadrants::lang::offline_cache {

constexpr std::size_t offline_cache_key_length = 65;
constexpr std::size_t min_mangled_name_length = offline_cache_key_length + 2;

std::string mangle_name(const std::string &primal_name,
                        const std::string &key) {
  // Result: {primal_name}{key: char[65]}_{(checksum(primal_name)) ^
  // checksum(key)}
  if (key.size() != offline_cache_key_length) {
    return primal_name;
  }
  std::size_t checksum1{0}, checksum2{0};
  for (auto &e : primal_name) {
    checksum1 += std::size_t(e);
  }
  for (auto &e : key) {
    checksum2 += std::size_t(e);
  }
  return fmt::format("{}{}_{}", primal_name, key, checksum1 ^ checksum2);
}

bool try_demangle_name(const std::string &mangled_name,
                       std::string &primal_name,
                       std::string &key) {
  if (mangled_name.size() < min_mangled_name_length) {
    return false;
  }

  std::size_t checksum{0}, checksum1{0}, checksum2{0};
  auto pos = mangled_name.find_last_of('_');
  if (pos == std::string::npos) {
    return false;
  }
  try {
    checksum = std::stoull(mangled_name.substr(pos + 1));
  } catch (const std::exception &) {
    return false;
  }

  std::size_t i = 0, primal_len = pos - offline_cache_key_length;
  for (i = 0; i < primal_len; ++i) {
    checksum1 += (int)mangled_name[i];
  }
  for (; i < pos; ++i) {
    checksum2 += (int)mangled_name[i];
  }
  if ((checksum1 ^ checksum2) != checksum) {
    return false;
  }

  primal_name = mangled_name.substr(0, primal_len);
  key = mangled_name.substr(primal_len, offline_cache_key_length);
  QD_ASSERT(key.size() == offline_cache_key_length);
  QD_ASSERT(primal_name.size() + key.size() == pos);
  return true;
}

}  // namespace quadrants::lang::offline_cache
