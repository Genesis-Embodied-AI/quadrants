# GPU execution model

This page defines the core GPU execution-model terms used across the Quadrants user guide: kernel, thread, subgroup, block, lane, and shared memory. Other pages link here the first time they use one of these words. If you already know them from CUDA, HIP, or another GPU framework you can skip this page; the only Quadrants-specific content is the API name attached to each concept.

Quadrants compiles the same Python kernel to several GPU backends (CUDA, AMD, Vulkan, and Metal), and each vendor uses a different word for the same piece of hardware. This page picks one name per concept and lists the vendor synonyms, so the rest of the guide can stay vendor-neutral.

## Kernel

"Kernel" means two related but distinct things, and this guide keeps them apart:

- A Quadrants kernel is the Python function you write and decorate with `@qd.kernel`. It is the program, expressed in Python, that describes the work to do. This guide writes it as `@qd.kernel` whenever the distinction matters.
- A hardware kernel is a compiled device program that Quadrants generates for a specific backend (CUDA, AMD, Vulkan, Metal, or CPU) and launches on the device.

The two are not one-to-one: Quadrants splits each `@qd.kernel` into one or more hardware kernels. Every top-level `for` loop becomes its own hardware kernel whose iterations run in parallel across [threads](#thread), and every contiguous run of serial statements outside any top-level `for` loop becomes its own hardware kernel as well. Running a `@qd.kernel` launches these hardware kernels in order; the `@qd.kernel` itself does not run on the device (this compilation model holds on GPU and CPU backends alike).

## Thread

When the device launches a [hardware kernel](#kernel) it spawns one or more threads; a thread is one such execution of the kernel's code, with its own registers and its own local variables. The hardware schedules and runs these threads. Each thread reads an index identifying it so it can decide which data to work on. This is the ordinary "thread" you already use; the terms below just describe how threads are grouped and how they cooperate.

## Subgroup

A subgroup (a warp on NVIDIA, a wavefront or wave on AMD, and a subgroup on Vulkan and Metal) is a fixed-size group of threads that the hardware runs together, advancing through the same instruction at the same time. Its size is fixed by the hardware and backend, not something you choose: it is typically 32 threads (NVIDIA, and AMD in wave32 mode) or 64 threads (AMD in wave64 mode). Because the threads of a subgroup run together like this, they can exchange values held in registers directly, without going through [shared memory](#shared-memory), using [subgroup operations](subgroup.md).

## Block

A block (also called a thread block, a CUDA cooperative thread array or CTA, or a workgroup on AMD, Vulkan, and Metal) is a set of threads that launch together and can cooperate closely: they can wait for each other at a barrier (a point every thread in the block must reach before any thread moves past it) and share fast on-chip [shared memory](#shared-memory). The threads of a block are divided into [subgroups](#subgroup). You set how many threads a block has with `qd.loop_config(block_dim=N)`. A block's shared memory and its block barrier only reach the threads of that one block; coordinating across blocks needs coarser tools such as a device-scope memory fence or a separate kernel launch (see [grid](grid.md)).

## Lane

A lane is the position of a thread within its subgroup, numbered from 0 to the subgroup size minus 1. Lane numbers are per subgroup, not per block: two threads in the same block but in different subgroups can carry the same lane number. A global thread index tells you which thread you are among all the launched threads; the lane index tells you where you sit inside your own subgroup. Subgroup operations that move a value from one thread to another (a shuffle, for example) name the source or destination thread by its lane. In Quadrants you read the calling thread's lane with `qd.simt.subgroup.invocation_id()`.

## Shared memory

Shared memory is fast on-chip memory that every thread in one [block](#block) can read and write, used to pass data between threads that would otherwise only see their own registers and the slower global memory. On the memory-speed spectrum it sits between the two: slower than a thread's own registers, but much faster than global memory. It is small in size, private to a single block, and gone once the block finishes. Vendors call it shared memory (CUDA), local data share or LDS (AMD), or threadgroup memory (Metal). In Quadrants you allocate it with `qd.simt.block.SharedArray(shape, dtype)` and coordinate access to it with the block barrier `qd.simt.block.sync()`; see [block primitives](block.md).
