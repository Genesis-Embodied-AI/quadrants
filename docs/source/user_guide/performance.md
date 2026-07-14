# Performance

This page covers the execution model that the other performance guides build on: what happens when you launch a GPU kernel, where the time goes, and when reducing that time actually speeds up your program. The other pages in this section - graphs, streams, fastcache, and performance dispatch - are specific tools for reducing that cost.

## What a GPU kernel launch is

When you call a `@qd.kernel` from Python, the numerical work does not run inline in the calling thread. Instead Quadrants *launches* the work onto the GPU: it hands a compiled kernel plus its arguments to the GPU driver, the driver enqueues it onto a stream (an ordered queue of GPU work), and the launch call returns to the host before the GPU has finished - usually before it has even started. Host and GPU run asynchronously.

Each top-level `for`-loop in a kernel is a separate *offloaded task*, and each offloaded task becomes one GPU kernel. So a kernel with three top-level for-loops launches three GPU kernels every time you call it. Every launch carries a fixed overhead that is independent of how much data the kernel processes - that per-launch overhead is what we call *kernel launch latency*. When a kernel crunches a lot of data the overhead is negligible; when it does little work, or when you relaunch small kernels many times in a loop, launch latency can dominate the total runtime.

## Where kernel launch latency comes from

It helps to split the latency of a single launch into three parts:

- **Python side.** Everything that happens in the Python interpreter before control reaches Quadrants' C++ runtime. This is the most expensive of the three per call, and it is paid once per *Python* call - so a Python `while`/`for` loop that calls a kernel N times pays it N times.
- **C++ side.** Work inside the Quadrants runtime (C++) once control has crossed the boundary. Cheaper than the Python side, but non-zero, and paid per offloaded task.
- **GPU / driver side.** The GPU driver's own cost to place the kernel on a stream, plus the fixed scheduling latency on the GPU itself before the kernel begins running on the hardware.

## Latency hiding

Because host and GPU run asynchronously, kernel launches overlap with kernel execution: the host can be issuing the next launch while earlier kernels are still running on the GPU. As long as the host issues launches at least as fast as the GPU drains them, the queue never empties, the GPU stays busy, and the launch latency is fully *hidden* behind execution.

So reducing launch latency does not always improve overall throughput. When each kernel runs long enough that its execution time exceeds the launch overhead, the GPU is the bottleneck and shaving launch latency changes nothing - the launches were already hidden behind execution.

Kernel launch latency matters when kernels are relatively small. Then the GPU can finish a kernel before the host has issued the next one, so the GPU sits idle waiting for the next launch and the launch latency is *exposed* on the critical path. In that regime, cutting launch latency can directly increase throughput.

## Reducing launch latency

[Graphs](graph.md) are the main tool for reducing launch latency. On backends with hardware graph support, the kernel's whole sequence of GPU tasks is issued as a single graph launch rather than one launch per task, collapsing the per-task C++ and GPU/driver launch overhead. The per-call Python overhead is still paid on every call; to cut that too, move the loop itself into the graph with `graph_do_while` so that many iterations become a single call.

## Other performance tools

The rest of this section covers tools that improve performance in other ways, not by reducing launch latency:

- [Streams](streams.md) let independent top-level for-loops run concurrently instead of serializing on the default stream.
- [Fastcache](fastcache.md) reduces the time to load cached kernels when a new Python process starts.
- [Performance dispatch](perf_dispatch.md) benchmarks multiple implementations of an operation at runtime and picks the fastest.
