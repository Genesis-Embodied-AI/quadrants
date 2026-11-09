# Experimental CPU pool comparison

This branch provides opt-in CPU thread-pool prototypes for benchmarking short parallel loops. The normal pool remains the default. Set these environment variables before importing and initializing Quadrants:

- `QD_CPU_POOL_MODE=baseline`: use the existing pool.
- `QD_CPU_POOL_MODE=dynamic`: let the caller execute tasks alongside workers, with dynamic task claiming.
- `QD_CPU_POOL_MODE=static`: use caller participation and assign task `i` to execution thread `i % active_threads` consistently across launches.
- `QD_CPU_POOL_SPIN_US=200`: let idle workers poll for up to this many microseconds before sleeping. This applies to `dynamic` and `static`; zero disables idle spinning. The default is 200 microseconds.

In both experimental modes, the configured CPU thread limit includes the caller. Launches requiring only one execution thread run directly on the caller. After finishing its tasks, the caller spins until the participating workers complete. Worker spinning between launches is bounded; caller completion spinning lasts until the work finishes.

These controls are experimental and do not change solver settings or simulation accuracy options. Static assignment can affect load balance, and changing thread assignment can change random-number streams or floating-point reduction order. Validate numerical results as well as timing for your workload.
