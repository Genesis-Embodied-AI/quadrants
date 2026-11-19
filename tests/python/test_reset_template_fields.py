import dataclasses
import gc
import weakref

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("field_kind", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("offline_cache", [False, True])
@test_utils.test()
def test_reset_releases_template_fields(field_kind, nested, offline_cache, monkeypatch):
    arch = qd.cfg.arch
    monkeypatch.setenv("QD_OFFLINE_CACHE", str(int(offline_cache)))

    @qd.kernel
    def fill(x: qd.Template, value: qd.i32):
        if qd.static(nested):
            for i in x[0]:
                x[0][i] += value
        else:
            for i in x:
                x[i] += value

    refs = []
    for value in (1, 2, 3):
        qd.init(arch=arch)
        if field_kind == "scalar":
            x = qd.field(qd.i32, shape=4)
        elif field_kind == "vector":
            x = qd.Vector.field(2, qd.i32, shape=4)
        else:
            x = qd.Matrix.field(2, 2, qd.i32, shape=4)
        fill((x,) if nested else x, value)
        np.testing.assert_array_equal(x.to_numpy(), value)
        refs.append(weakref.ref(x))
        del x
        qd.reset()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not fill._primal.mapper.mapping
        assert not fill._primal.mapper._mapping_cache
        assert not fill._primal.mapper._mapping_cache_tracker


@test_utils.test()
def test_reset_releases_gradient_template_fields():
    arch = qd.cfg.arch

    @qd.kernel
    def square(x: qd.Template, loss: qd.Template):
        for i in x:
            loss[None] += x[i] * x[i]

    refs = []
    for _ in range(3):
        qd.init(arch=arch)
        x = qd.field(qd.f32, shape=4, needs_grad=True)
        loss = qd.field(qd.f32, shape=(), needs_grad=True)
        x.fill(2)
        square(x, loss)
        loss.grad[None] = 1
        square.grad(x, loss)
        np.testing.assert_array_equal(x.grad.to_numpy(), 4)
        refs.extend((weakref.ref(x), weakref.ref(loss)))
        del x, loss
        qd.reset()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not square._primal.mapper.mapping
        assert not square._adjoint.mapper.mapping


@test_utils.test()
def test_reset_reuses_specialization_ids_with_new_member_paths():
    arch = qd.cfg.arch

    @qd.data_oriented(template_primitives=False)
    class Left:
        def __init__(self):
            self.left = qd.ndarray(qd.i32, shape=1)
            self.offset = 3

    @qd.data_oriented(template_primitives=False)
    class Right:
        def __init__(self):
            self.right = qd.ndarray(qd.i32, shape=1)
            self.offset = 7

    @qd.kernel
    def read(state: qd.Template) -> qd.i32:
        value = 0
        if qd.static(hasattr(state, "left")):
            value = state.left[0]
        else:
            value = state.right[0]
        return value + state.offset

    for cls, member in ((Left, "left"), (Right, "right"), (Left, "left")):
        qd.init(arch=arch)
        state = cls()
        getattr(state, member)[0] = 10
        assert read(state) == 10 + state.offset
        assert list(read._primal.mapper.mapping.values()) == [0]
        replacement = qd.ndarray(qd.i32, shape=1)
        replacement[0] = 20
        setattr(state, member, replacement)
        state.offset += 1
        assert read(state) == 20 + state.offset
        qd.reset()
        assert not read._primal.used_py_dataclass_parameters_by_key_enforcing
        assert not read._primal._struct_ndarray_launch_info_by_key
        assert not read._primal._runtime_scalar_launch_infos_by_compiled_kernel_key


@test_utils.test()
def test_reset_releases_runtime_dataclass_fields():
    arch = qd.cfg.arch

    @dataclasses.dataclass(frozen=True)
    class State:
        x: qd.Tensor

    @qd.kernel
    def read(state: State) -> qd.i32:
        return state.x[0]

    refs = []
    for value in (1, 2, 3):
        qd.init(arch=arch)
        x = qd.field(qd.i32, shape=1)
        x[0] = value
        state = State(x)
        assert read(state) == value
        refs.extend((weakref.ref(x), weakref.ref(state)))
        del x, state
        qd.reset()
        gc.collect()
        assert all(ref() is None for ref in refs)
        assert not read._primal.mapper.mapping
