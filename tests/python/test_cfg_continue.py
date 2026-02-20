import quadrants as qd

from tests import test_utils


@test_utils.test(exclude=[qd.amdgpu])
def test_cfg_continue():
    x = qd.field(dtype=int, shape=1)
    state = qd.field(dtype=int, shape=1)

    @qd.kernel
    def foo():
        for p in range(1):
            if state[p] == 0:
                x[p] = 1
                continue
            if state[p] != 0:
                print("test")

    foo()
    assert x[0] == 1
