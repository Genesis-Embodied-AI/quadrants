#pragma once

#include <functional>
#include <vector>

#include "quadrants/program/launch_context_builder.h"

namespace quadrants::lang {

// Shared host-side driver for (possibly nested) graph_do_while loops, used by every backend that does not have native
// GPU conditional-graph support (CPU, CUDA pre-SM 9.0, AMDGPU, GFX). It reconstructs the loop nesting from the flat
// offloaded-task list using each task's `graph_do_while_level_id` plus the launch-context level table (parent
// pointers), and drives the do-while loops on the host.
//
// The backend supplies two callbacks:
//   launch_task(i)        : launch offloaded task `i` (synchronously or enqueued). Return false to
//                           abort the whole kernel (e.g. a device-side assert fired); the driver then
//                           unwinds without running further tasks or iterations.
//   continue_level(level) : called after one pass over a loop level's body. The backend should make
//                           that level's writes visible (e.g. stream sync) and read the level's flag
//                           ndarray; return true to run the body again, false to exit the loop.
//
// `task_level_ids[i]` is task i's innermost level id (-1 if outside all graph_do_while loops).
namespace graph_do_while_detail {

inline bool is_descendant_or_self(int level, int ancestor, const std::vector<GraphDoWhileLevel> &levels) {
  for (int c = level; c != -1; c = levels[c].parent_id) {
    if (c == ancestor) {
      return true;
    }
  }
  return false;
}

inline int child_of(int parent_id, int descendant, const std::vector<GraphDoWhileLevel> &levels) {
  int c = descendant;
  while (levels[c].parent_id != parent_id) {
    c = levels[c].parent_id;
  }
  return c;
}

// Returns false if aborted (launch_task returned false), true otherwise.
inline bool process_level(int parent_id,
                          int begin,
                          int end,
                          const std::vector<int> &task_level_ids,
                          const std::vector<GraphDoWhileLevel> &levels,
                          const std::function<bool(int)> &launch_task,
                          const std::function<bool(int)> &continue_level) {
  bool keep_going = true;
  do {
    int cursor = begin;
    while (cursor < end) {
      const int task_level = task_level_ids[cursor];
      if (task_level == parent_id) {
        if (!launch_task(cursor)) {
          return false;
        }
        cursor++;
      } else {
        const int child = child_of(parent_id, task_level, levels);
        int run_end = cursor;
        while (run_end < end && is_descendant_or_self(task_level_ids[run_end], child, levels)) {
          run_end++;
        }
        if (!process_level(child, cursor, run_end, task_level_ids, levels, launch_task, continue_level)) {
          return false;
        }
        cursor = run_end;
      }
    }
    // The kernel top level (parent_id == -1) runs exactly once; a real loop level repeats while its flag stays
    // non-zero.
    keep_going = (parent_id >= 0) && continue_level(parent_id);
  } while (keep_going);
  return true;
}

}  // namespace graph_do_while_detail

// Drives the whole task list. `launch_task` / `continue_level` as described above.
inline void run_graph_do_while(int num_tasks,
                               const std::vector<int> &task_level_ids,
                               const std::vector<GraphDoWhileLevel> &levels,
                               const std::function<bool(int)> &launch_task,
                               const std::function<bool(int)> &continue_level) {
  graph_do_while_detail::process_level(/*parent_id=*/-1, 0, num_tasks, task_level_ids, levels, launch_task,
                                       continue_level);
}

}  // namespace quadrants::lang
