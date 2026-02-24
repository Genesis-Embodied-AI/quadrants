import ctypes
import math
import time

import quadrants as qd

libm = ctypes.CDLL("libm.so.6")

x, y = qd.field(qd.f32), qd.field(qd.f32)


@qd.kernel
def laplace():
    for i, j in x:
        y[i, j] = 4.0 * x[i, j] - x[i - 1, j] - x[i + 1, j] - x[i, j - 1] - x[i, j + 1]


qd.root.dense(qd.ij, (16, 16)).place(x).place(y)

laplace()

t = time.time()
N = 1000000
for i in range(N):
    x[i & 7, i & 7] = 1.0
print((time.time() - t) / N * 1e9, "ns")

t = time.time()
N = 1000000
a = 0
for i in range(N):
    a += x[i, i]
print((time.time() - t) / N * 1e9, "ns")

t = time.time()
N = 1000000
a = 0
sin = getattr(libm, "sin")
for i in range(N):
    a += sin(i)
print((time.time() - t) / N * 1e9, "ns")

t = time.time()
N = 1000000
a = 0
for i in range(N):
    a += math.sin(i)
print((time.time() - t) / N * 1e9, "ns")
