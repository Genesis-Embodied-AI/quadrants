from enum import Enum
from types import SimpleNamespace

import pytest

import quadrants as qd
from quadrants.lang.misc import _EnvironmentConfigurator


class ExampleMode(Enum):
    FIRST = 1
    SECOND = 2


@pytest.mark.parametrize("enum_type", [ExampleMode, qd.Layout])
def test_environment_configurator_enum(enum_type, monkeypatch):
    first, second = list(enum_type)[:2]
    base_cfg = SimpleNamespace(option=first)
    # An environment variable containing an enum member name selects that member.
    monkeypatch.setenv("QD_OPTION", second.name)
    _EnvironmentConfigurator({}, base_cfg).add("option")
    assert base_cfg.option is second

    # A keyword argument overrides even an invalid environment value and is removed from kwargs.
    monkeypatch.setenv("QD_OPTION", "invalid")
    kwargs = {"option": first}
    _EnvironmentConfigurator(kwargs, base_cfg).add("option")
    assert base_cfg.option is first
    assert kwargs == {}

    # Without a keyword override, an invalid enum name in the environment raises an error.
    with pytest.raises(ValueError, match="QD_OPTION must be"):
        _EnvironmentConfigurator({}, base_cfg).add("option")

    # An empty environment variable is ignored, leaving the configured value unchanged.
    monkeypatch.setenv("QD_OPTION", "")
    _EnvironmentConfigurator({}, base_cfg).add("option")
    assert base_cfg.option is first


@pytest.mark.parametrize("value", ["FIRST", 1, True, None, qd.Layout.AOS])
def test_environment_configurator_rejects_nonmember(value, monkeypatch):
    monkeypatch.delenv("QD_OPTION", raising=False)
    cfg = SimpleNamespace(option=ExampleMode.FIRST)
    with pytest.raises(TypeError, match="option must be a ExampleMode member"):
        _EnvironmentConfigurator({"option": value}, cfg).add("option")
    assert cfg.option is ExampleMode.FIRST


@pytest.mark.parametrize(
    "default, text, expected", [(False, "0", False), (False, "1", True), (32, "64", 64), ("a", "b", "b")]
)
def test_environment_configurator_infers_parser(default, text, expected, monkeypatch):
    cfg = SimpleNamespace(option=default)
    monkeypatch.setenv("QD_OPTION", text)
    _EnvironmentConfigurator({}, cfg).add("option")
    assert cfg.option == expected
    assert type(cfg.option) is type(expected)
