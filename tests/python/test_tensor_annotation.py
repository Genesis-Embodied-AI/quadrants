"""Tests for ``qd.tensor_annotation``."""

import pytest

import quadrants as qd

from tests import test_utils


def test_tensor_annotation_field_returns_template_instance():
    ann = qd.tensor_annotation(qd.Backend.FIELD)
    assert isinstance(ann, qd.types.annotations.template)


def test_tensor_annotation_ndarray_returns_ndarray_type():
    ann = qd.tensor_annotation(qd.Backend.NDARRAY)
    direct = qd.types.ndarray()
    assert type(ann) is type(direct)


def test_tensor_annotation_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.tensor_annotation(99)


def test_tensor_annotation_int_value_accepted():
    """``qd.tensor_annotation(0)`` and ``(1)`` work via IntEnum coercion."""
    ann_field = qd.tensor_annotation(0)
    ann_ndarray = qd.tensor_annotation(1)
    assert isinstance(ann_field, qd.types.annotations.template)
    assert type(ann_ndarray) is type(qd.types.ndarray())


@test_utils.test(arch=qd.cpu)
def test_tensor_annotation_field_drives_kernel():
    V_ANNOTATION = qd.tensor_annotation(qd.Backend.FIELD)

    @qd.kernel
    def fill(x: V_ANNOTATION):
        for i in range(4):
            x[i] = i + 10

    a = qd.tensor(qd.i32, shape=(4,))
    fill(a)
    assert list(a.to_numpy()) == [10, 11, 12, 13]


@test_utils.test(arch=qd.cpu)
def test_tensor_annotation_ndarray_drives_kernel():
    V_ANNOTATION = qd.tensor_annotation(qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: V_ANNOTATION):
        for i in range(4):
            x[i] = i + 100

    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    fill(a)
    assert list(a.to_numpy()) == [100, 101, 102, 103]
