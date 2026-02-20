import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("static_value", [False, True])
@pytest.mark.parametrize("is_inner", [False, True])
@pytest.mark.parametrize("use_field", [False, True])
@test_utils.test()
def test_for_static_if_iter_runs(use_field: bool, is_inner: bool, static_value: bool) -> None:
    # Note that we currently dont have a way to turn static range on/off using some kind of variable/parameter.
    # So, for now, we'll have one side as static range, and one side as non-static range.
    # Since the code itself treats either side identically (same code path except for choosing one or the other side),
    # whilst the test isn't ideal, it should give identical coverage to something more rigorous.
    # We can think about approaches to parametrizing the static range in the future (nop function, macro,
    # parametrizablle qd.static, parametrizable qd.range, etc...).

    B = 2
    N_right = 5

    V = qd.field if use_field else qd.ndarray
    V_ANNOT = qd.Template if use_field else qd.types.NDArray[qd.i32, 2]

    if is_inner:

        @qd.kernel
        def k1(a: V_ANNOT, n_left: qd.i32) -> None:
            for b in range(B):
                for i in range(n_left) if qd.static(static_value) else qd.static(range(N_right)):
                    a[b, i] = 1

    else:

        @qd.kernel
        def k1(a: V_ANNOT, n_left: qd.i32) -> None:
            for i in range(n_left) if qd.static(static_value) else qd.static(range(N_right)):
                a[0, i] = 1

    def create_expected(n_left: int):
        a_expected = np.zeros(dtype=np.int32, shape=(B, 6))
        for b in range(B) if is_inner else range(1):
            for i in range(n_left) if static_value else range(N_right):
                a_expected[b, i] = 1
        return a_expected

    a = V(qd.i32, (B, 6))
    k1(a, n_left=2)
    assert np.all(create_expected(n_left=2) == a.to_numpy())

    a = V(qd.i32, (B, 6))
    k1(a, n_left=3)
    assert np.all(create_expected(n_left=3) == a.to_numpy())


@pytest.mark.parametrize("is_static", [False, True])
@test_utils.test()
def test_for_static_if_iter_static_ranges(is_static: bool) -> None:
    # See comments on test_for_static_if_iter_runs for discussion of testing static vs non static ranges.

    # In this test, we verify that the static side is really static, and that the non-static side is
    # really non-static, by adding a conditional break to each, and seeing if that causes compilation to fail.

    # Note that break is only valid in inner loops, so we only test the inner loop case.
    B = 2
    N_left = 3
    N_right = 5

    @qd.kernel
    def k1(break_threshold: qd.i32, n_right: qd.i32) -> None:
        for b in range(B):
            for i in qd.static(range(N_left)) if qd.static(is_static) else range(n_right):
                if i >= break_threshold:
                    break

    if is_static:
        with pytest.raises(qd.QuadrantsCompilationError, match="You are trying to `break` a static `for` loop"):
            k1(0, N_right)
    else:
        # Dynamic break is ok, since not static for range.
        k1(0, N_right)


@pytest.mark.parametrize("use_field", [False, True])
@test_utils.test()
def test_for_static_if_forwards_backwards(use_field: bool) -> None:
    """
    Test a forwards/backwards requirement for rigid body differentiation.
    """
    MAX_LINKs = 3
    BATCH_SIZE = 1

    V = qd.field if use_field else qd.ndarray
    V_ANNOT = qd.Template if use_field else qd.types.NDArray[qd.i32, 1]
    V_ANNOT2 = qd.Template if use_field else qd.types.NDArray[qd.i32, 2]

    field_a = V(qd.i32, shape=(BATCH_SIZE))
    field_a.from_numpy(np.array([1]))

    field_target = V(qd.i32, (BATCH_SIZE, MAX_LINKs))

    @qd.kernel
    def k1(is_backward: qd.template(), field_a: V_ANNOT, field_target: V_ANNOT2):
        for i_b in range(BATCH_SIZE):
            for j in qd.static(range(MAX_LINKs)) if qd.static(is_backward) else range(field_a[i_b]):
                print("is_backward", is_backward, j)
                field_target[i_b, j] = 1

    k1(is_backward=False, field_a=field_a, field_target=field_target)
    k1(is_backward=True, field_a=field_a, field_target=field_target)


@test_utils.test()
def test_for_static_if_no_ad1():
    @qd.kernel
    def k1():
        for b in range(2):
            for i in range(2):
                ...

    k1()

    if qd.is_extension_enabled(qd.extension.adstack):
        k1.grad()
    else:
        with pytest.raises(qd.QuadrantsCompilationError):
            k1.grad()


@test_utils.test()
def test_for_static_if_no_ad2():
    B = 10
    x = qd.field(qd.f32, shape=(3, B), needs_grad=True)
    y = qd.field(qd.math.vec3, shape=(3, B), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    x[0, 0] = 1.0
    y[0, 0].fill(2.0)

    @qd.kernel
    def k1(use_static: qd.template()):
        for i_b in qd.ndrange(B):
            z = qd.Vector.zero(qd.f32, 3)

            # Non-static inner loop is not supported in backward
            for i_3 in qd.static(range(3)) if qd.static(use_static) else range(3):
                z += x[i_3, i_b] * y[i_3, i_b]

            loss[None] += z.x + z.y + z.z

    use_static = False

    loss.fill(0.0)
    k1(use_static)

    loss.grad[None] = 1.0
    x.grad.fill(0.0)
    y.grad.fill(0.0)

    if qd.is_extension_enabled(qd.extension.adstack):
        k1.grad(use_static)
    else:
        with pytest.raises(qd.QuadrantsCompilationError):
            k1.grad(use_static)
