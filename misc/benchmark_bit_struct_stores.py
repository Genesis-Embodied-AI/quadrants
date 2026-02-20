import quadrants as qd

qd.init(arch=qd.cpu, kernel_profiler=True, print_ir=True)

quant = True

n = 1024 * 1024 * 256

if quant:
    qi16 = qd.types.quant.int(16, True)

    x = qd.field(dtype=qi16)
    y = qd.field(dtype=qi16)

    qd.root.dense(qd.i, n).bit_struct(num_bits=32).place(x, y)
else:
    x = qd.field(dtype=qd.i16)
    y = qd.field(dtype=qd.i16)

    qd.root.dense(qd.i, n).place(x, y)


@qd.kernel
def foo():
    for i in range(n):
        x[i] = i & 1023
        y[i] = i & 15


for i in range(10):
    foo()

qd.print_kernel_profile_info()
