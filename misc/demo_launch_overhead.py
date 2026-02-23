import time

import quadrants as qd

qd.init()


@qd.kernel
def compute_div(a: qd.i32):
    pass


compute_div(0)
print("starting...")
t = time.time()
for i in range(100000):
    compute_div(0)
print((time.time() - t) * 10, "us")
exit(0)
