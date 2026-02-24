# https://github.com/taichi-dev/quadrants/pull/839#issuecomment-626217806
import quadrants as qd

qd.init(print_ir=True)
# qd.core.toggle_advanced_optimization(False)


@qd.kernel
def calc_pi() -> qd.f32:
    term = 1.0
    sum = 0.0
    divisor = 1
    for i in qd.static(range(10)):
        sum += term / divisor
        term *= -1 / 3
        divisor += 2
    return sum * qd.sqrt(12.0)


print(calc_pi())
