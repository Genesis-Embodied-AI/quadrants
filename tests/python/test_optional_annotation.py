"""Tests for the ``T | None`` optional kernel-argument spelling (design.md W4).

W4 makes ``T | None`` parse uniformly for the three annotation families (``qd.Tensor``, ``qd.types.Template``,
``qd.types.NDArray[...]``) and records the slot as optional on its ``ArgMetadata``. ``qd.Tensor`` and
``qd.types.Template`` slots already accept ``None`` at runtime, so for those two the union spelling works end to end
here. ``qd.types.NDArray[...] | None`` parses and still works when a value is present; making ``None`` actually launch
against an ndarray slot is W5, and the boundary is pinned below.
"""

import typing

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeTypeError, QuadrantsSyntaxError
from quadrants.types.ndarray_type import NdarrayType

from tests import test_utils

# ----------------------------------------------------------------------------
# Parse + normalization: the union spelling is accepted and the slot is marked optional.
# These run at @qd.kernel decoration time and need no qd.init().
# ----------------------------------------------------------------------------


def _arg0(kernel):
    return kernel._primal.arg_metas[0]


def test_tensor_union_parses_and_marks_optional():
    @qd.kernel
    def k(x: qd.Tensor | None):
        pass

    meta = _arg0(k)
    assert meta.optional is True
    # None is stripped; the stored annotation is the bare qd.Tensor class, so downstream dispatch is unchanged.
    assert meta.annotation is qd.Tensor


def test_template_union_parses_and_marks_optional():
    @qd.kernel
    def k(x: qd.types.Template | None):
        pass

    meta = _arg0(k)
    assert meta.optional is True
    assert meta.annotation is qd.types.Template


def test_ndarray_union_parses_and_preserves_spec():
    @qd.kernel
    def k(x: qd.types.NDArray[qd.f32, 1] | None):
        pass

    meta = _arg0(k)
    assert meta.optional is True
    # The dtype/ndim carried by the subscripted instance must survive the __or__ shim unwrap.
    assert isinstance(meta.annotation, NdarrayType)
    assert meta.annotation.ndim == 1
    assert meta.annotation.dtype == qd.f32


def test_optional_typing_form_parses_for_class_families():
    """``typing.Optional[T]`` is equivalent to ``T | None`` for the two class-style families (design.md D1: the
    ndarray family stays ``|``-only until issue 831, because ``Optional[instance]`` fails ``typing._type_check``)."""

    @qd.kernel
    def kt(x: typing.Optional[qd.Tensor]):
        pass

    @qd.kernel
    def ktemplate(x: typing.Optional[qd.types.Template]):
        pass

    assert _arg0(kt).optional is True
    assert _arg0(kt).annotation is qd.Tensor
    assert _arg0(ktemplate).optional is True
    assert _arg0(ktemplate).annotation is qd.types.Template


def test_non_optional_annotation_stays_non_optional():
    @qd.kernel
    def k(x: qd.Tensor):
        pass

    assert _arg0(k).optional is False
    assert _arg0(k).annotation is qd.Tensor


def test_union_without_none_is_rejected():
    """Only ``T | None`` is a supported union; any other union must raise a clear syntax error."""
    with pytest.raises(QuadrantsSyntaxError, match=r"T \| None"):

        @qd.kernel
        def k(x: int | float):
            pass


def test_multi_member_optional_union_is_rejected():
    with pytest.raises(QuadrantsSyntaxError, match=r"T \| None"):

        @qd.kernel
        def k(x: qd.Tensor | int | None):
            pass


def test_optional_unsupported_family_is_rejected():
    """`T | None` is only for qd.Tensor / qd.types.Template / qd.types.NDArray. A family with no absent-value path
    (here a dataclass) must be rejected at registration, not silently recorded as optional and left to crash with an
    internal AttributeError when `None` reaches the dataclass launch branch."""
    import dataclasses

    @dataclasses.dataclass
    class State:
        a: qd.Tensor

    with pytest.raises(QuadrantsSyntaxError, match="only supported for qd.Tensor"):

        @qd.kernel
        def k(s: State | None):
            pass


# ----------------------------------------------------------------------------
# Runtime: qd.Tensor | None works end to end (both branches), just like the bare spelling.
# ----------------------------------------------------------------------------


@test_utils.test()
def test_tensor_union_none_and_present_run():
    out = qd.ndarray(qd.f32, shape=(4,))
    bias = qd.ndarray(qd.f32, shape=(4,))
    bias.from_numpy(np.full(4, 100.0, dtype=np.float32))

    @qd.kernel
    def add_bias(out: qd.types.NDArray[qd.f32, 1], bias: qd.Tensor | None):
        for i in range(out.shape[0]):
            if qd.static(bias is not None):
                out[i] = qd.f32(i) + bias[i]
            else:
                out[i] = qd.f32(i)

    add_bias(out, None)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(4, dtype=np.float32))

    add_bias(out, qd.wrap(bias))
    np.testing.assert_array_equal(out.to_numpy(), np.arange(4, dtype=np.float32) + 100.0)

    # Present and absent branches specialize separately.
    assert len(add_bias._primal.mapper.mapping) == 2


@test_utils.test()
def test_template_union_none_and_present_run():
    out = qd.ndarray(qd.i32, shape=(1,))

    @qd.kernel
    def pick(out: qd.types.NDArray[qd.i32, 1], flag: qd.types.Template | None):
        if qd.static(flag is not None):
            out[0] = flag
        else:
            out[0] = -1

    pick(out, None)
    assert out.to_numpy()[0] == -1

    pick(out, 7)
    assert out.to_numpy()[0] == 7


# ----------------------------------------------------------------------------
# Runtime: qd.types.NDArray[...] | None. Present value works now; None is the W5 boundary.
# ----------------------------------------------------------------------------


@test_utils.test()
def test_ndarray_union_present_value_runs():
    a = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def fill(x: qd.types.NDArray[qd.f32, 1] | None):
        for i in range(x.shape[0]):
            x[i] = qd.f32(i) * 10.0

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), np.array([0, 10, 20, 30], dtype=np.float32))


@test_utils.test()
def test_ndarray_union_none_not_yet_supported():
    """W4/W5 boundary: an optional ndarray slot parses, but launching it with ``None`` still fails until W5 makes the
    slot dual-nature. W5 will flip this test from raises to passes."""
    a = qd.ndarray(qd.f32, shape=(4,))

    @qd.kernel
    def maybe_fill(x: qd.types.NDArray[qd.f32, 1] | None, out: qd.types.NDArray[qd.f32, 1]):
        for i in range(out.shape[0]):
            out[i] = qd.f32(i)

    with pytest.raises(QuadrantsRuntimeTypeError):
        maybe_fill(None, a)
