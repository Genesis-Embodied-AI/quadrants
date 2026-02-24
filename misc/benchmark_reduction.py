import quadrants as qd

# TODO: make this a real benchmark and set up regression

qd.init(arch=qd.gpu)

N = 1024 * 1024 * 1024

a = qd.field(qd.i32, shape=N)
tot = qd.field(qd.i32, shape=())


@qd.kernel
def fill():
    qd.block_dim(128)
    for i in a:
        a[i] = i


@qd.kernel
def reduce():
    qd.block_dim(1024)
    for i in a:
        tot[None] += a[i]


fill()
fill()

for i in range(10):
    reduce()

ground_truth = 10 * N * (N - 1) / 2 % 2**32
assert tot[None] % 2**32 == ground_truth
qd.print_kernel_profile_info()
