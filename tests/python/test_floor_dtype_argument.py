import quadrants as qd

from tests import test_utils


def _test_floor_ceil_round(dt):
    @qd.kernel
    def make_tests():
        x = 1.5
        v = qd.math.vec3(1.1, 2.2, 3.3)

        assert qd.floor(x) == 1
        assert qd.floor(x, dt) == 1.0
        assert qd.floor(x, int) == 1

        assert all(qd.floor(v) == [1, 2, 3])
        assert all(qd.floor(v, dt) == [1.0, 2.0, 3.0])
        assert all(qd.floor(v, int) == [1, 2, 3])

        assert qd.ceil(x) == 2
        assert qd.ceil(x, dt) == 2.0
        assert qd.ceil(x, int) == 2

        assert all(qd.ceil(v) == [2, 3, 4])
        assert all(qd.ceil(v, dt) == [2.0, 3.0, 4.0])
        assert all(qd.ceil(v, int) == [2, 3, 4])

        assert qd.round(x) == 2
        assert qd.round(x, dt) == 2.0
        assert qd.round(x, int) == 2

        assert all(qd.round(v) == [1, 2, 3])
        assert all(qd.round(v, dt) == [1.0, 2.0, 3.0])
        assert all(qd.round(v, int) == [1, 2, 3])

    make_tests()


@test_utils.test(default_fp=qd.f32)
def test_floor_ceil_round_f32():
    _test_floor_ceil_round(qd.f32)


@test_utils.test(default_fp=qd.f64, require=qd.extension.data64)
def test_floor_ceil_round_f64():
    _test_floor_ceil_round(qd.f64)
