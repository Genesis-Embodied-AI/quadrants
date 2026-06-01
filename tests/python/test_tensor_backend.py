"""Tests for ``qd.Backend``.

Scope: this PR only ships the enum. No factory, no layout, no kernel integration. Tests cover symbol export, value,
name, ordering, and IntEnum semantics so that downstream PRs can rely on stable behaviour.
"""

from enum import IntEnum

import pytest

import quadrants as qd


def test_backend_is_exported():
    assert hasattr(qd, "Backend")


def test_backend_is_intenum():
    assert issubclass(qd.Backend, IntEnum)


def test_backend_values():
    assert int(qd.Backend.FIELD) == 0
    assert int(qd.Backend.NDARRAY) == 1


def test_backend_names():
    assert qd.Backend.FIELD.name == "FIELD"
    assert qd.Backend.NDARRAY.name == "NDARRAY"


def test_backend_lookup_by_name():
    assert qd.Backend["FIELD"] is qd.Backend.FIELD
    assert qd.Backend["NDARRAY"] is qd.Backend.NDARRAY


def test_backend_lookup_by_value():
    assert qd.Backend(0) is qd.Backend.FIELD
    assert qd.Backend(1) is qd.Backend.NDARRAY


def test_backend_int_compare():
    assert qd.Backend.FIELD == 0
    assert qd.Backend.NDARRAY == 1
    assert qd.Backend.FIELD < qd.Backend.NDARRAY


def test_backend_members_are_distinct():
    members = list(qd.Backend)
    assert len(members) == 2
    assert qd.Backend.FIELD in members
    assert qd.Backend.NDARRAY in members


def test_backend_invalid_value_raises():
    with pytest.raises(ValueError):
        qd.Backend(2)
