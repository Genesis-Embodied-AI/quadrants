#pragma once

#include <cstdint>
#include <vector>

namespace quadrants::lang {

class Program;
class Ndarray;

// Read the two-word replay measurement without scalar-access staging allocations on CPU/CUDA.
std::vector<int64_t> read_adstack_replay_measurement(Program *program, Ndarray *buffer);

}  // namespace quadrants::lang
