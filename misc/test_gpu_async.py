import time

import quadrants as qd

qd.init(arch=qd.cuda)

a = qd.field(qd.f32, shape=(1024 * 1024 * 1024))


@qd.kernel
def fill(x: qd.f32):
    for i in a:
        a[i] = x


for i in range(100):
    t = time.time()
    fill(i)
    print(time.time() - t)

print(a[0])
