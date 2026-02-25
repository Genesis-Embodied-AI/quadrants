import time

from pytest import approx

import quadrants as qd

# TODO: make this a real benchmark and set up regression
# TODO: merge this file into benchmark_reduction.py
qd.init(
    arch=qd.gpu,
    print_ir=True,
    print_kernel_llvm_ir=True,
    kernel_profiler=True,
    print_kernel_llvm_ir_optimized=True,
)

N = 1024 * 1024 * 128

a = qd.field(qd.f32, shape=N)


@qd.kernel
def fill():
    qd.block_dim(128)
    for i in a:
        a[i] = 1.0


@qd.kernel
def reduce() -> qd.f32:
    s = 0.0
    qd.block_dim(1024)
    for i in a:
        s += a[i]
    return s


fill()

num_runs = 10
# Invoke it here to get the kernel compiled
reduce()

start = time.time()
got = 0.0
for i in range(num_runs):
    got += reduce()
duration = time.time() - start
print(f"duration={duration:.2e}s average={(duration / num_runs):.2e}s")

ground_truth = float(N * num_runs)
assert got == approx(ground_truth, 1e-4)
