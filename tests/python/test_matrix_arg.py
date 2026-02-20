import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_matrix_arg():
    mat1 = qd.Matrix([[1, 2, 3], [4, 5, 6]])

    @qd.kernel
    def foo(mat: qd.types.matrix(2, 3, qd.i32)) -> qd.i32:
        return mat[0, 0] + mat[1, 2]

    assert foo(mat1) == 7

    mat3 = qd.Matrix([[1, 2], [3, 4], [5, 6]])

    @qd.kernel
    def foo2(var: qd.i32, mat: qd.types.matrix(3, 2, qd.i32)) -> qd.i32:
        res = mat
        for i in qd.static(range(3)):
            for j in qd.static(range(2)):
                res[i, j] += var
        return res[2, 1]

    assert foo2(3, mat3) == 9


@test_utils.test()
def test_vector_arg():
    vec1 = qd.Vector([1, 2, 3])

    @qd.kernel
    def foo(vec: qd.types.vector(3, qd.i32)) -> int:
        return vec[0] + vec[1] + vec[2]

    assert foo(vec1) == 6


@test_utils.test()
def test_matrix_fancy_arg():
    from quadrants.math import mat3, vec3

    mat4x3 = qd.types.matrix(4, 3, float)
    mat2x6 = qd.types.matrix(2, 6, float)

    a = np.random.random(3)
    b = np.random.random((3, 3))

    v = vec3(0, 1, 2)
    v = vec3([0, 1, 2])

    M = mat3(a, a, a)
    M = mat3(b)

    m = mat4x3([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])

    m = mat4x3([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    m = mat4x3(vec3(1, 2, 3), vec3(4, 5, 6), vec3(7, 8, 9), vec3(10, 11, 12))

    m = mat4x3([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    m = mat4x3(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    m = mat4x3(1)

    m = mat4x3(m)

    k = mat2x6(m)


@test_utils.test()
def test_matrix_arg_insertion_pos():
    rgba8 = qd.types.vector(4, qd.u8)

    @qd.kernel
    def _render(
        color_attm: qd.types.ndarray(rgba8, ndim=2),
        camera_pos: qd.math.vec3,
        camera_up: qd.math.vec3,
    ):
        up = qd.math.normalize(camera_up)

        for x, y in color_attm:
            o = camera_pos

    color_attm = qd.Vector.ndarray(4, dtype=qd.u8, shape=(512, 512))
    camera_pos = qd.math.vec3(0, 0, 0)
    camera_up = qd.math.vec3(0, 1, 0)

    _render(color_attm, camera_pos, camera_up)
