#pragma once

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace quadrants::lang {

// Read the complete file as bytes, preserving zeros; throw if opening or reading fails.
inline std::string read_binary_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  if (file.bad()) {
    throw std::runtime_error("Failed reading file: " + path);
  }
  return bytes;
}

}  // namespace quadrants::lang
