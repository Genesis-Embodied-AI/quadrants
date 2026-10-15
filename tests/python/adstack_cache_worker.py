"""Cross-process cache regression worker; invoked by test_adstack_compatibility."""

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
)


@qd.kernel(fastcache=fast)
def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.i32):
    for i in range(x.shape[0]):
        value = x[i]
        for j in range(n):
            value = qd.sin(value)
        y[i] = value


x = qd.ndarray(qd.f32, shape=4, needs_grad=True)
y = qd.ndarray(qd.f32, shape=4, needs_grad=True)
x.fill(0.2)
n = 5 if warm else 2
compute(x, y, n)
y.grad.fill(1)
compute.grad(x, y, n)
x.grad.fill(0)
with qd.adstack as scope:
    compute(x, y, n)
assert scope.capacities[0] > 1
scope.records[0].backward()
value, derivative = 0.2, 1.0
for _ in range(n):
    derivative *= math.cos(value)
    value = math.sin(value)
assert abs(x.grad[0] - derivative) < 2e-5
assert abs(y[0] - value) < 2e-5
for variant in (
    compute._primal,
    compute._primal._adstack_count_variant,
    compute._adjoint._adstack_reverse_variant,
):
    if fast:
        assert (
            variant.src_ll_cache_observations.cache_loaded == warm
        ), variant.src_ll_cache_observations
    else:
        assert (
            variant.fe_ll_cache_observations.cache_hit == warm
        ), variant.fe_ll_cache_observations
# The ordinary forward still behaves normally after using the counted variants.
compute(x, y, 1)
assert abs(y[0] - math.sin(0.2)) < 2e-5
qd.reset()
print(f"CACHE_OK fast={fast} warm={warm}")
