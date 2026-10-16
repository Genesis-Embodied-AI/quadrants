"""Exercise process teardown after a caught replay overflow."""

import sys

import quadrants as qd
from quadrants.lang import adstack_replay

arch, force_sync = sys.argv[1:]
qd.init(arch=getattr(qd, arch), enable_fallback=False, offline_cache=False)
x = qd.ndarray(qd.f32, shape=17, needs_grad=True)
y = qd.ndarray(qd.f32, shape=17, needs_grad=True)
n = qd.ndarray(qd.i32, shape=1)
x.fill(0.2)
y.grad.fill(1)
n[0] = 2


@qd.kernel
def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
    for i in range(x.shape[0]):
        value = x[i]
        for j in range(n[0]):
            value = qd.sin(value)
        y[i] = value


original_variant = adstack_replay._variant


def grow_after_measurement(kernel, mode):
    if mode == 2:
        n[0] = 64
    return original_variant(kernel, mode)


adstack_replay._variant = grow_after_measurement
try:
    compute.grad(x, y, n)
except RuntimeError as error:
    assert "Adstack capacity exceeded" in str(error), str(error)
else:
    raise AssertionError("Expected an overflow after increasing depth beyond the measurement")
finally:
    adstack_replay._variant = original_variant
if int(force_sync):
    qd.sync()
qd.reset()
print("OVERFLOW_TEARDOWN_OK")
