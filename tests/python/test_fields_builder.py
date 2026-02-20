import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeError

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal])
def test_fields_with_shape():
    shape = 5
    x = qd.field(qd.f32, shape=shape)

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i

    assign_field_single()
    for i in range(shape):
        assert x[i] == i

    y = qd.field(qd.f32, shape=shape)

    @qd.kernel
    def assign_field_multiple():
        for i in range(shape):
            y[i] = i * 2
        for i in range(shape):
            x[i] = i * 3

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 3
        assert y[i] == i * 2

    assign_field_single()
    for i in range(shape):
        assert x[i] == i


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal])
def test_fields_builder_dense():
    shape = 5
    fb1 = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    fb1.dense(qd.i, shape).place(x)
    fb1.finalize()

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = qd.FieldsBuilder()
    y = qd.field(qd.f32)
    fb2.dense(qd.i, shape).place(y)
    z = qd.field(qd.f32)
    fb2.dense(qd.i, shape).place(z)
    fb2.finalize()

    @qd.kernel
    def assign_field_multiple():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_fields_builder_pointer():
    shape = 5
    fb1 = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    fb1.pointer(qd.i, shape).place(x)
    fb1.finalize()

    @qd.kernel
    def assign_field_single():
        for i in range(shape):
            x[i] = i * 3

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3

    fb2 = qd.FieldsBuilder()
    y = qd.field(qd.f32)
    fb2.pointer(qd.i, shape).place(y)
    z = qd.field(qd.f32)
    fb2.pointer(qd.i, shape).place(z)
    fb2.finalize()

    @qd.kernel
    def assign_field_multiple_range_for():
        for i in range(shape):
            x[i] = i * 2
        for i in range(shape):
            y[i] = i + 5
        for i in range(shape):
            z[i] = i + 10

    assign_field_multiple_range_for()
    for i in range(shape):
        assert x[i] == i * 2
        assert y[i] == i + 5
        assert z[i] == i + 10

    @qd.kernel
    def assign_field_multiple_struct_for():
        for i in y:
            y[i] += 5
        for i in z:
            z[i] -= 5

    assign_field_multiple_struct_for()
    for i in range(shape):
        assert y[i] == i + 10
        assert z[i] == i + 5

    assign_field_single()
    for i in range(shape):
        assert x[i] == i * 3


# We currently only consider data types that all platforms support.
# See https://docs.taichi-lang.org/docs/type#primitive-types for more details.
@pytest.mark.parametrize("test_1d_size", [1, 10, 100])
@pytest.mark.parametrize("field_type", [qd.f32, qd.i32])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal])
def test_fields_builder_destroy(test_1d_size, field_type):
    def test_for_single_destroy_multi_fields():
        fb = qd.FieldsBuilder()
        for create_field_idx in range(10):
            field = qd.field(field_type)
            fb.dense(qd.i, test_1d_size).place(field)
        fb_snode_tree = fb.finalize()
        fb_snode_tree.destroy()

    def test_for_multi_destroy_multi_fields():
        fb0 = qd.FieldsBuilder()
        fb1 = qd.FieldsBuilder()

        for create_field_idx in range(10):
            field0 = qd.field(field_type)
            field1 = qd.field(field_type)

            fb0.dense(qd.i, test_1d_size).place(field0)
            fb1.pointer(qd.i, test_1d_size).place(field1)

        fb0_snode_tree = fb0.finalize()
        fb1_snode_tree = fb1.finalize()

        fb0_snode_tree.destroy()
        fb1_snode_tree.destroy()

    def test_for_raise_destroy_twice():
        fb = qd.FieldsBuilder()
        a = qd.field(qd.f32)
        fb.dense(qd.i, test_1d_size).place(a)
        c = fb.finalize()

        with pytest.raises(QuadrantsRuntimeError):
            c.destroy()
            c.destroy()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan])
def test_field_initialize_zero():
    fb0 = qd.FieldsBuilder()
    a = qd.field(qd.i32)
    fb0.dense(qd.i, 1).place(a)
    c = fb0.finalize()
    a[0] = 5
    c.destroy()
    fb1 = qd.FieldsBuilder()
    b = qd.field(qd.i32)
    fb1.dense(qd.i, 1).place(b)
    d = fb1.finalize()
    assert b[0] == 0


@test_utils.test()
def test_field_builder_place_grad():
    @qd.kernel
    def mul(arr: qd.template(), out: qd.template()):
        for i in arr:
            out[i] = arr[i] * 2.0

    @qd.kernel
    def calc_loss(arr: qd.template(), loss: qd.template()):
        for i in arr:
            loss[None] += arr[i]

    arr = qd.field(qd.f32, needs_grad=True)
    fb0 = qd.FieldsBuilder()
    fb0.dense(qd.i, 10).place(arr, arr.grad)
    snode0 = fb0.finalize()
    out = qd.field(qd.f32)
    fb1 = qd.FieldsBuilder()
    fb1.dense(qd.i, 10).place(out, out.grad)
    snode1 = fb1.finalize()
    loss = qd.field(qd.f32)
    fb2 = qd.FieldsBuilder()
    fb2.place(loss, loss.grad)
    snode2 = fb2.finalize()
    arr.fill(1.0)
    mul(arr, out)
    calc_loss(out, loss)
    loss.grad[None] = 1.0
    calc_loss.grad(out, loss)
    mul.grad(arr, out)
    for i in range(10):
        assert arr.grad[i] == 2.0


@test_utils.test(arch=qd.cpu)
def test_fields_builder_numpy_dimension():
    shape = np.int32(5)
    fb = qd.FieldsBuilder()
    x = qd.field(qd.f32)
    y = qd.field(qd.i32)
    fb.dense(qd.i, shape).place(x)
    fb.pointer(qd.j, shape).place(y)
    fb.finalize()
