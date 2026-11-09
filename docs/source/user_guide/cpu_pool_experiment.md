# Experimental CPU pool comparison

This branch provides opt-in CPU thread-pool prototypes for benchmarking short parallel loops. The normal pool remains the default. Set these environment variables before importing and initializing Quadrants:

- `QD_CPU_POOL_MODE=baseline`: use the existing pool.
- `QD_CPU_POOL_MODE=dynamic`: let the caller execute tasks alongside workers, with dynamic task claiming.
- `QD_CPU_POOL_MODE=static`: use caller participation and assign task `i` to execution thread `i % active_threads` consistently across launches.
- `QD_CPU_POOL_SPIN_US=200`: let idle workers poll for up to this many microseconds before sleeping. This applies to `dynamic` and `static`; zero disables idle spinning. The default is 200 microseconds.

In both experimental modes, the configured CPU thread limit includes the caller. Launches requiring only one execution thread run directly on the caller. After finishing its tasks, the caller spins until the participating workers complete. Worker spinning between launches is bounded; caller completion spinning lasts until the work finishes.

These controls are experimental and do not change solver settings or simulation accuracy options. Static assignment can affect load balance, and changing thread assignment can change random-number streams or floating-point reduction order. Validate numerical results as well as timing for your workload.

## Controlled mechanism comparisons

`QD_CPU_POOL_MODE=ablation` selects a separate common implementation for experiments that change one mechanism at a time.
It always waits for every selected execution slot to finish a launch; the original pool allows late workers to skip
finished launches. Therefore its all-condition-variable configuration is a controlled reference, not an identical copy
of the original pool. Compare it with `baseline` separately before interpreting individual changes.

Options are read when the pool is constructed:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `QD_ABLATION_CALLER_WORK` | `0` | `1` assigns execution slot zero to the caller instead of a worker. |
| `QD_ABLATION_INLINE_SINGLE` | `0` | `1` bypasses launch/completion when only the caller is needed. |
| `QD_ABLATION_ATOMIC_COMPLETION` | `0` | `1` replaces mutex-protected completion counting with an atomic countdown. |
| `QD_ABLATION_CALLER_WAIT` | `cv` | Completion waiting: `cv`, `tight`, `pause`, or `sleep1`. Polling requires atomic completion. |
| `QD_ABLATION_WORKER_WAIT` | `cv` | Between-launch waiting: `cv`, `tight`, `pause`, or `sleep1`. |
| `QD_ABLATION_SPIN_US` | `0` | Worker polling budget: `0` sleeps immediately; `-1` polls indefinitely; positive values bound polling. |
| `QD_ABLATION_FAST_LAUNCH` | `0` | `1` skips the launch mutex/notification when no worker has registered for sleep. |
| `QD_ABLATION_STATIC_TASKS` | `0` | `1` replaces atomic task claiming with stable assignment. |

`pause` uses the CPU pause/yield instruction; `tight` adds no instruction between atomic checks. `sleep1` requests a
one-microsecond sleep between checks. The operating system can sleep longer than requested; a bounded polling period
can overshoot its budget by that sleep. Caller polling has no time bound. All execution-thread limits include the caller
when caller work is enabled. The existing `dynamic` and `static` prototypes retain their earlier behavior and controls.
