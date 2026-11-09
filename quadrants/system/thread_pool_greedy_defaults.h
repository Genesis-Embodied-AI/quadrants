#pragma once

namespace quadrants {

// Experimental branch defaults: bit 0 static tasks, bit 1 caller polling, bit 2 atomic completion.
inline constexpr int greedy_default_features = 1;
inline constexpr int greedy_default_worker_spin_us = 200;

}  // namespace quadrants
