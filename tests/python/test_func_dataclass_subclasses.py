from dataclasses import dataclass

import pytest

import quadrants as qd
from quadrants.lang._kernel_types import KernelBatchedArgType

from tests.helpers import test_utils


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("use_field", [False, True])
@test_utils.test()
def test_func_dataclass_subclass(frozen, keyword, use_field):
    @dataclass(frozen=frozen)
    class Base:
        value: qd.Tensor
        unused: qd.Tensor

    @dataclass(frozen=frozen)
    class Child(Base):
        extra: qd.Tensor

    @qd.func
    def add(before: int, state: Base, after: int):
        state.value[0] += before + after

    @qd.data_oriented
    class Writer:
        @qd.func
        def forward(self, before: int, state: Base, after: int):
            if qd.static(keyword):
                add(after=after, state=state, before=before)
            else:
                add(before, state, after)

    writer = Writer()

    @qd.kernel
    def run(state: Child):
        if qd.static(keyword):
            writer.forward(after=3, state=state, before=2)
            add(7, after=11, state=state)
        else:
            writer.forward(2, state, 3)
            add(7, state, 11)
        # A subclass-only field can still be used by the caller after narrowing at the call boundary.
        state.extra[0] += 13

    make = qd.field if use_field else qd.ndarray
    arrays = [make(qd.i32, shape=(1,)) for _ in range(3)]
    for array in arrays:
        array.fill(0)
    state = Child(*arrays)
    run(state)
    run(state)
    assert arrays[0][0] == 46
    assert arrays[1][0] == 0
    assert arrays[2][0] == 26
    assert run._primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == (0 if use_field else 2)


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("from_instance", [False, True])
@test_utils.test()
def test_func_dataclass_subclass_nested(keyword, from_instance):
    @dataclass(frozen=True)
    class Leaf:
        value: qd.Tensor

    @dataclass(frozen=True)
    class ExtraLeaf(Leaf):
        unused: qd.Tensor

    @dataclass(frozen=True)
    class Base:
        leaf: Leaf

    @dataclass(frozen=True)
    class Child(Base):
        leaf: ExtraLeaf
        unused: qd.Tensor

    @qd.func
    def add_leaf(leaf: Leaf):
        leaf.value[0] += 1

    @qd.func
    def add(state: Base):
        add_leaf(state.leaf)

    @qd.data_oriented
    class Runner:
        def __init__(self, state):
            self.state = state

        @qd.func
        def forward(self, state: Base):
            add(state)

        @qd.kernel
        def run(self):
            if qd.static(keyword):
                self.forward(state=self.state)
                add_leaf(leaf=self.state.leaf)
            else:
                self.forward(self.state)
                add_leaf(self.state.leaf)

    @qd.kernel
    def run(state: Child):
        if qd.static(keyword):
            add(state=state)
            add_leaf(leaf=state.leaf)
        else:
            add(state)
            add_leaf(state.leaf)

    arrays = [qd.ndarray(qd.i32, shape=(1,)) for _ in range(3)]
    for array in arrays:
        array.fill(0)
    state = Child(ExtraLeaf(arrays[0], arrays[1]), arrays[2])
    if from_instance:
        runner = Runner(state)
        runner.run()
        runner.run()
    else:
        run(state)
        run(state)
        assert run._primal.launch_stats.kernel_args_count_by_type[KernelBatchedArgType.QD_ARRAY] == 1
    assert arrays[0][0] == 4
    assert arrays[1][0] == arrays[2][0] == 0


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("from_instance", [False, True])
@test_utils.test()
def test_func_dataclass_subclass_rejects_unrelated(keyword, from_instance):
    @dataclass
    class Base:
        value: qd.Tensor

    @dataclass
    class Unrelated:
        value: qd.Tensor

    @qd.func
    def add(state: Base):
        state.value[0] += 1

    @qd.kernel
    def run(state: Unrelated):
        if qd.static(keyword):
            add(state=state)
        else:
            add(state)

    @qd.data_oriented
    class Runner:
        def __init__(self, state):
            self.state = state

        @qd.kernel
        def run(self):
            if qd.static(keyword):
                add(state=self.state)
            else:
                add(self.state)

    array = qd.ndarray(qd.i32, shape=(1,))
    array.fill(0)
    state = Unrelated(array)
    with pytest.raises(qd.QuadrantsTypeError, match="Expected dataclass Base or a subclass, got Unrelated"):
        if from_instance:
            Runner(state).run()
        else:
            run(state)
    assert array[0] == 0
