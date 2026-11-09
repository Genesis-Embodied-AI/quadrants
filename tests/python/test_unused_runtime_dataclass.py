from dataclasses import dataclass

import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("unused", ["application notes", {1, 2}, object(), [1, 2]])
@test_utils.test()
def test_unused_runtime_member(unused):
    @dataclass
    class State:
        value: str
        unused: int

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    assert read(State(7, unused)) == 7


@pytest.mark.parametrize("keyword", [False, True])
@test_utils.test()
def test_unused_runtime_member_through_nested_func(keyword):
    @dataclass
    class Inner:
        value: object
        unused: object

    @dataclass(frozen=True)
    class Outer:
        inner: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.value

    @qd.func
    def forward(container: qd.RuntimeTyped):
        if qd.static(keyword):
            return read(renamed=container.inner)
        else:
            return read(container.inner)

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return forward(state)

    child = Inner(7, object())
    state = Outer(child)
    assert run(state) == 7
    child.value = 9
    child.unused = {1, 2}
    assert run(state) == 9


@test_utils.test()
def test_used_unsupported_runtime_member_reports_original_path():
    @dataclass
    class State:
        value: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.value

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return read(renamed=state)

    with pytest.raises(Exception, match=r"state.value.*unsupported runtime type set"):
        run(State({1, 2}))


@test_utils.test()
def test_unused_runtime_member_through_bound_method():
    @dataclass
    class State:
        value: object
        unused: object

    @qd.data_oriented
    class Reader:
        @qd.func
        def read(self, renamed: qd.RuntimeTyped):
            return renamed.value

    reader = Reader()

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return reader.read(renamed=state)

    assert run(State(9, object())) == 9


@test_utils.test()
def test_runtime_deferred_member_type_transitions():
    @dataclass
    class State:
        value: object
        unused: object

    @qd.kernel
    def read(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    state = State(7, 1)
    for ignored in (object(), object(), 2, {1, 2}, {3}, 3):
        state.unused = ignored
        state.value += 1
        assert read(state) == state.value
    state.value = object()
    with pytest.raises(Exception, match=r"state.value.*unsupported runtime type object"):
        read(state)
    state.value = 42
    assert read(state) == 42


@test_utils.test()
def test_runtime_deferred_static_and_runtime_branches():
    from typing import Final

    @dataclass(frozen=True)
    class State:
        enabled: Final[bool]
        unused: object

    @qd.func
    def conditional(renamed: qd.RuntimeTyped):
        result = 7
        if qd.static(renamed.enabled):
            result = renamed.unused
        return result

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return conditional(state)

    assert run(State(False, object())) == 7
    with pytest.raises(Exception, match=r"state.unused.*unsupported runtime type object"):
        run(State(True, object()))
    assert run(State(False, object())) == 7

    @qd.kernel
    def runtime_branch(state: qd.RuntimeTyped, enabled: qd.i32) -> qd.i32:
        result = 7
        if enabled:
            result = state.unused
        return result

    with pytest.raises(Exception, match=r"state.unused.*unsupported runtime type object"):
        runtime_branch(State(False, object()), 0)


@pytest.mark.parametrize("invalid_annotation", [False, True])
@test_utils.test()
def test_unused_final_member_preserves_valid_sibling(invalid_annotation):
    from typing import Final

    @dataclass(frozen=True)
    class Config:
        value: Final[int]
        unused: Final[list] if invalid_annotation else Final[int]

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return qd.static(renamed.value)

    @qd.kernel
    def run(config: qd.RuntimeTyped) -> qd.i32:
        return read(renamed=config)

    assert run(Config(7, [1, 2])) == 7
    assert run(Config(9, [3])) == 9

    @qd.kernel
    def use_invalid(config: qd.RuntimeTyped) -> qd.i32:
        return qd.static(config.unused[0])

    with pytest.raises(Exception, match=r"config.unused.*Final"):
        use_invalid(Config(7, [1, 2]))


@test_utils.test()
def test_unused_final_below_mutable_ancestor():
    from typing import Final

    @dataclass(frozen=True)
    class Inner:
        value: object
        unused: Final[int]

    @dataclass
    class Outer:
        child: object

    @qd.func
    def read(renamed: qd.RuntimeTyped):
        return renamed.value

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return read(state.child)

    state = Outer(Inner(7, 1))
    assert run(state) == 7
    state.child = Inner(9, 2)
    assert run(state) == 9

    @qd.kernel
    def use_invalid(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.child.unused)

    with pytest.raises(Exception, match=r"state.child.unused.*not frozen but reaches"):
        use_invalid(state)


@test_utils.test()
def test_unused_final_on_mutable_carrier():
    from typing import Final

    @dataclass
    class State:
        value: object
        unused: Final[int]

    @qd.kernel
    def run(state: qd.RuntimeTyped) -> qd.i32:
        return state.value

    state = State(7, 1)
    assert run(state) == 7
    state.unused = 2
    assert run(state) == 7

    @qd.kernel
    def use_invalid(state: qd.RuntimeTyped) -> qd.i32:
        return qd.static(state.unused)

    with pytest.raises(Exception, match=r"state.unused.*not frozen"):
        use_invalid(state)


@test_utils.test()
def test_runtime_deferred_members_disable_offline_cache(tmp_path):
    from typing import Final

    @dataclass(frozen=True)
    class State:
        value: Final[int]
        output: object
        unused: object

    arch = qd.lang.impl.current_cfg().arch
    for bias in (7, 9):
        qd.init(arch=arch, offline_cache=True, offline_cache_file_path=str(tmp_path))

        @qd.kernel(fastcache=True)
        def run(state: qd.RuntimeTyped):
            value = qd.static(state.value)
            state.output[0] = value

        output = qd.ndarray(qd.i32, shape=1)
        run(State(bias, output, 1))
        assert output[0] == bias
        assert run._primal.src_ll_cache_observations.cache_key_generated
        for unused in (object(), object(), 2, object()):
            # Observations accumulate across launches; clear them to inspect this particular call.
            observed = run._primal.src_ll_cache_observations
            observed.cache_loaded = False
            observed.cache_stored = False
            observed.cache_key_generated = False
            run(State(bias, output, unused))
            assert output[0] == bias
            observed = run._primal.src_ll_cache_observations
            if type(unused) is object:
                assert not observed.cache_loaded
                assert not observed.cache_stored
                assert not observed.cache_key_generated
        assert run._primal._runtime_has_deferred


@test_utils.test()
def test_unused_member_in_captured_dataclass():
    @dataclass(frozen=True)
    class State:
        value: object
        unused: object

    @qd.func
    def read(state: qd.RuntimeTyped):
        return state.value[0]

    @qd.data_oriented
    class Reader:
        def __init__(self, state):
            self.state = state

        @qd.kernel
        def run(self) -> qd.i32:
            return read(self.state)

    values = qd.ndarray(qd.i32, shape=1)
    values[0] = 9
    assert Reader(State(values, object())).run() == 9


@pytest.mark.parametrize("debug_settings", [(False, True, False), (True, False, True)])
@pytest.mark.parametrize("member_kind", ["ordinary", "final_value", "final_container"])
@test_utils.test(arch=[qd.cpu])
def test_deferred_runtime_type_debug_cache_isolation(debug_settings, member_kind):
    from typing import Final

    from quadrants.lang._runtime_dataclass import infer_dataclass_type

    @dataclass(frozen=member_kind != "final_container")
    class Inner:
        value: Final if member_kind != "ordinary" else object

    @dataclass(frozen=True)
    class Outer:
        child: object
        valid: object

    value = 1 if member_kind == "final_container" else {1, 2}
    state = Outer(Inner(value), 7)
    schemas = {}
    for debug in debug_settings:
        qd.reset()
        qd.init(arch=qd.cpu, debug=debug, offline_cache=False)
        schema = infer_dataclass_type(state, "state")
        if debug in schemas:
            assert schema is schemas[debug]
        elif schemas:
            assert schema is not next(iter(schemas.values()))
        schemas[debug] = schema

        @qd.kernel
        def read_valid(state: qd.RuntimeTyped) -> qd.i32:
            return state.valid

        @qd.kernel
        def read_invalid(state: qd.RuntimeTyped) -> qd.i32:
            return state.child.value

        # An unused invalid member remains harmless, including after cached launches and runtime resets.
        assert read_valid(state) == 7
        assert read_valid(state) == 7
        with pytest.raises(Exception, match=r"state.child.value") as error:
            read_invalid(state)
        message = str(error.value)
        expected_type = "builtins.int" if member_kind == "final_container" else "builtins.set"
        assert (f"(runtime type: {expected_type})" in message) == debug
        expected_error = {
            "ordinary": "unsupported runtime type set",
            "final_value": "Final",
            "final_container": "not frozen",
        }[member_kind]
        assert expected_error in message


@test_utils.test(arch=[qd.cpu], debug=True)
def test_deferred_debug_type_names_do_not_retain_application_class():
    import gc
    import weakref

    from quadrants.lang._runtime_dataclass import infer_dataclass_type

    @dataclass(frozen=True)
    class State:
        value: object

    class Unsupported:
        pass

    value = Unsupported()
    class_ref = weakref.ref(Unsupported)
    value_ref = weakref.ref(value)
    schema = infer_dataclass_type(State(value), "state")
    description = schema.__dataclass_fields__["value"].type
    expected_name = f"{Unsupported.__module__}.{Unsupported.__qualname__}"
    del value, Unsupported
    gc.collect()
    assert value_ref() is None
    assert class_ref() is None
    with pytest.raises(Exception) as error:
        description.bind("__qd_state__qd_value").reject()
    assert f"(runtime type: {expected_name})" in str(error.value)
