import pytest

import quadrants as qd
from quadrants.math import inf, isinf, isnan, nan, pi, vdir

from tests import test_utils


def _test_inf_nan(dt):
    @qd.kernel
    def make_tests():
        assert isnan(nan) == isnan(-nan) == True
        x = -1.0
        assert isnan(qd.sqrt(x)) == True
        assert isnan(inf) == isnan(1.0) == isnan(-1) == False
        assert isinf(inf) == isinf(-inf) == True
        assert isinf(nan) == isinf(1.0) == isinf(-1) == False

        v = qd.math.vec4(inf, -inf, 1.0, nan)
        assert all(isinf(v) == [1, 1, 0, 0])

        v = qd.math.vec4(nan, -nan, 1, inf)
        assert all(isnan(v) == [1, 1, 0, 0])

    make_tests()


@qd.func
def check_epsilon_equal(mat_cal, mat_ref, epsilon) -> int:
    assert mat_cal.n == mat_ref.n and mat_cal.m == mat_ref.m
    err = 0
    for i in qd.static(range(mat_cal.n)):
        for j in qd.static(range(mat_cal.m)):
            err = qd.abs(mat_cal[i, j] - mat_ref[i, j]) > epsilon
    return err


@pytest.mark.parametrize("dt", [qd.f32, qd.f64])
@test_utils.test()
def test_inf_nan_f32(dt):
    _test_inf_nan(dt)


@test_utils.test()
def test_vdir():
    @qd.kernel
    def make_test():
        assert all(vdir(pi / 2) == [0, 1])

    make_test()


@test_utils.test(default_fp=qd.f32, debug=True)
def test_vector_types_f32():
    @qd.dataclass
    class Ray:
        pos: qd.math.vec3
        uv: qd.math.vec2
        mat: qd.math.mat3
        _id: qd.math.uvec2

    @qd.kernel
    def test():
        ray = Ray(qd.math.vec3(pi), qd.math.vec2(0.5, 0.5), qd.math.mat3(1))

    test()


@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, default_ip=qd.i64, debug=True)
def test_vector_types_f64():
    @qd.dataclass
    class Ray:
        pos: qd.math.vec3
        uv: qd.math.vec2
        mat: qd.math.mat3
        id: qd.math.uvec2

    @qd.kernel
    def test():
        pi = 3.14159265358
        N = qd.u64(2**63 - 1)
        ray = Ray(qd.math.vec3(pi), qd.math.vec2(pi), id=qd.math.uvec2(N))

        assert abs(ray.pos.x - pi) < 1e-10
        assert ray.id.x == N

    test()


@test_utils.test(debug=True)
@qd.kernel
def test_translate():
    error = 0
    translate_vec = qd.math.vec3(1.0, 2.0, 3.0)
    translate_mat = qd.math.translate(translate_vec[0], translate_vec[1], translate_vec[2])
    translate_ref = qd.math.mat4(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    error += check_epsilon_equal(translate_mat, translate_ref, 0.00001)
    assert error == 0


@test_utils.test(debug=True)
@qd.kernel
def test_scale():
    error = 0
    scale_vec = qd.math.vec3(1.0, 2.0, 3.0)
    scale_mat = qd.math.scale(scale_vec[0], scale_vec[1], scale_vec[2])
    scale_ref = qd.math.mat4(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    error += check_epsilon_equal(scale_mat, scale_ref, 0.00001)
    assert error == 0


@test_utils.test(debug=True)
@qd.kernel
def test_rotation2d():
    error = 0
    rotationTest = qd.math.rotation2d(qd.math.radians(30))
    rotationRef = qd.math.mat2([[0.866025, -0.500000], [0.500000, 0.866025]])
    error += check_epsilon_equal(rotationRef, rotationTest, 0.00001)
    assert error == 0


@test_utils.test(debug=True)
@qd.kernel
def test_rotation3d():
    error = 0

    first = 1.046
    second = 0.52
    third = -0.785
    axisX = qd.math.vec3(1.0, 0.0, 0.0)
    axisY = qd.math.vec3(0.0, 1.0, 0.0)
    axisZ = qd.math.vec3(0.0, 0.0, 1.0)

    rotationEuler = qd.math.rot_yaw_pitch_roll(first, second, third)
    rotationInvertedY = (
        qd.math.rot_by_axis(axisZ, third) @ qd.math.rot_by_axis(axisX, second) @ qd.math.rot_by_axis(axisY, -first)
    )
    rotationDumb = qd.Matrix.zero(qd.f32, 4, 4)
    rotationDumb = qd.math.rot_by_axis(axisY, first) @ rotationDumb
    rotationDumb = qd.math.rot_by_axis(axisX, second) @ rotationDumb
    rotationDumb = qd.math.rot_by_axis(axisZ, third) @ rotationDumb
    rotationTest = qd.math.rotation3d(second, third, first)

    dif0 = rotationEuler - rotationDumb
    dif1 = rotationEuler - rotationInvertedY

    difRef0 = qd.math.mat4(
        [
            [0.05048351, -0.61339645, -0.78816002, 0.0],
            [0.65833154, 0.61388511, -0.4355969, 0.0],
            [0.75103329, -0.49688014, 0.4348093, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    difRef1 = qd.math.mat4(
        [
            [-0.60788802, 0.0, -1.22438441, 0.0],
            [0.60837229, 0.0, -1.22340979, 0.0],
            [1.50206658, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    error += check_epsilon_equal(dif0, difRef0, 0.00001)
    error += check_epsilon_equal(dif1, difRef1, 0.00001)
    error += check_epsilon_equal(rotationEuler, rotationTest, 0.00001)

    assert error == 0
