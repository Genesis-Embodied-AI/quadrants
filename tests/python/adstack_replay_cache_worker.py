"""Cross-process cache regression worker; invoked by test_adstack_replay."""

import math
import sys

import quadrants as qd

arch, path, fast, warm = sys.argv[1:]
fast, warm = bool(int(fast)), bool(int(warm))
qd.init(
    arch=getattr(qd, arch),
    offline_cache=True,
    offline_cache_file_path=path,
    ad_stack_experimental_enabled=True,
    ad_stack_replay_count=True,
)


@qd.kernel(fastcache=fast)
def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
    for i in range(x.shape[0]):
        value = x[i]
        for j in range(n[0]):
            value = qd.sin(value)
        y[i] = value


x = qd.ndarray(qd.f32, shape=4, needs_grad=True)
y = qd.ndarray(qd.f32, shape=4, needs_grad=True)
x.fill(0.2)
depth = 5 if warm else 2
n = qd.ndarray(qd.i32, shape=1)
n[0] = depth
y.grad.fill(1)
x.grad.fill(0)
compute.grad(x, y, n)
value, derivative = 0.2, 1.0
for _ in range(depth):
    derivative *= math.cos(value)
    value = math.sin(value)
assert abs(x.grad[0] - derivative) < 2e-5
assert y[0] == 0
for variant in (
    compute._adjoint._adstack_count_variant,
    compute._adjoint._adstack_reverse_variant,
):
    if fast:
        assert variant.src_ll_cache_observations.cache_loaded == warm, variant.src_ll_cache_observations
    else:
        assert variant.fe_ll_cache_observations.cache_hit == warm, variant.fe_ll_cache_observations
# The ordinary forward still behaves normally after using the counted variants.
n[0] = 1
compute(x, y, n)
assert abs(y[0] - math.sin(0.2)) < 2e-5
qd.reset()
print(f"CACHE_OK fast={fast} warm={warm}")
