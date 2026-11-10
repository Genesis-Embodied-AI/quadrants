# Experimental CPU pool comparison

This branch provides opt-in CPU thread-pool prototypes for benchmarking short parallel loops. The normal pool remains the default. Set these environment variables before importing and initializing Quadrants:

- `QD_CPU_POOL_MODE=baseline`: use the existing pool.
- `QD_CPU_POOL_MODE=dynamic`: let the caller execute tasks alongside workers, with dynamic task claiming.
- `QD_CPU_POOL_MODE=static`: use caller participation and assign task `i` to execution thread `i % active_threads` consistently across launches.
- `QD_CPU_POOL_SPIN_US=200`: let idle workers poll for approximately this many microseconds before sleeping. This applies to `dynamic` and `static`; zero disables idle spinning. The default is 200 microseconds.

In both experimental modes, the configured CPU thread limit includes the caller. Launches requiring only one execution thread run directly on the caller. After finishing its tasks, the caller spins until the participating workers complete. Worker spinning between launches is bounded; caller completion spinning lasts until the work finishes.

These controls are experimental and do not change solver settings or simulation accuracy options. Static assignment can affect load balance, and changing thread assignment can change random-number streams or floating-point reduction order. Validate numerical results as well as timing for your workload.

The polling budget is approximate: clock reads are amortized, and a descheduled worker can resume after the budget expires. Once workers are sleeping, an idle pool does not intentionally keep polling. CPU time measures processor occupancy, not electrical power consumption.

For Genesis CPU experiments, select the pool before launching the process, for example `QD_CPU_POOL_MODE=static QD_CPU_POOL_SPIN_US=200 python benchmark.py`. Keep the scene, numerical mode, batch size, physical-core affinity, and thread count fixed when comparing pools. Compare against the fastest correct serial configuration and report CPU consumption alongside elapsed time.

A timing improvement does not establish serial/parallel numerical equivalence. In the current dumplings field-mode validation, the optimized pool matches the original parallel pool exactly in exported final states, while the existing serial and parallel execution paths differ. Preserve this distinction when validating application accuracy.
