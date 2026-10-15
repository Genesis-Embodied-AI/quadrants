# Measure stack storage during backward

The experimental `qd.init(ad_stack_replay_count=True, ad_stack_experimental_enabled=True)` option measures stack requirements immediately before each `.grad()` call or Tape backward. Forward calls require no context manager or recording handle. Backward-only kernels and repeated backwards use their current arguments and state.

Backward first executes a count-only version of its reduced primal replay, allocates measured storage, then runs the normal recording/reverse pass. Counting does not write output values or gradients. This adds a reduced replay and synchronization to each backward; it does not cache sizes by argument contents.

CPU (x64), CUDA, Vulkan, and Metal code paths are implemented. Stack accesses remain bounded if recording exceeds the measurement, and the host raises after execution; discard gradients from a failed backward. The prototype rejects residual writes to non-differentiable buffers rather than execute them twice. Other initial restrictions include random/internal operations, while/continue/return, sparse fields, graphs/checkpoints, and custom streams. Vector stacks require `force_scalarize_matrix=True`. These restrictions are being evaluated against Genesis.

The default remains the legacy sizing path, allowing comparisons. The replay-count option bypasses legacy size estimation for measured variants. Persistent caches distinguish counting, measured backward, and ordinary kernels.
