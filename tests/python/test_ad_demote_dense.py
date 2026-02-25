import quadrants as qd

from tests import test_utils


@test_utils.test(exclude=[qd.metal])
def test_ad_demote_dense():
    a = qd.field(qd.f32, shape=(7, 3, 19))

    @qd.kernel
    def inc():
        for i, j, k in a:
            a[i, j, k] += 1

    inc.grad()
