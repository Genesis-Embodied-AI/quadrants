"""Tests for ``@qd.data_oriented(template_primitives=False)``.

By default the primitive (``int`` / ``float`` / ``bool``) members of a ``@qd.data_oriented`` object reached through a
``qd.template()`` kernel argument are baked into the compiled kernel as compile-time constants: mutating them does not
take effect without re-specialisation. Decorating the class ``@qd.data_oriented(template_primitives=False)`` instead
lifts every primitive the kernel actually accesses into a runtime scalar kernel argument, read fresh on every launch.

This file pins the runtime-lifting behaviour, the pruning (only accessed primitives become kernel args), the
``qd.static`` hard error, and the opt-in boundary (default stays baked). See
``perso_hugh/doc/data_oriented_template_primitives.md`` for the design.
"""

import dataclasses

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

# ---------------------------------------------------------------------------
# 1. Int + float members are runtime args: mutate between launches, values take effect without recompiling.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_int_float_mutation():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.n = 3
            self.scale = 2.0
            self.x = qd.field(qd.f32, shape=8)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        for i in range(s.n):
            s.x[i] += s.scale

    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [2, 2, 2, 0, 0, 0, 0, 0])

    sim.n = 6
    sim.scale = 10.0
    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [12, 12, 12, 10, 10, 10, 0, 0])


# ---------------------------------------------------------------------------
# 2. Mutating a lifted primitive does NOT trigger re-specialisation (same compiled kernel is reused).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_no_respecialization(monkeypatch):
    compile_count = [0]
    materialize_orig = qd.lang.kernel_impl.Kernel.materialize

    def counting_materialize(self, *a, **k):
        before = len(self.materialized_kernels)
        ret = materialize_orig(self, *a, **k)
        if self.func.__name__ == "step" and len(self.materialized_kernels) > before:
            compile_count[0] += 1
        return ret

    monkeypatch.setattr("quadrants.lang.kernel_impl.Kernel.materialize", counting_materialize)

    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.k = 1
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.k * 100

    for k in (1, 2, 7, 42):
        sim.k = k
        step(sim)
        assert sim.out.to_numpy()[0] == k * 100
    assert compile_count[0] == 1, f"expected a single compile, got {compile_count[0]}"


# ---------------------------------------------------------------------------
# 3. Bool member.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_bool():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.enabled = True
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        if s.enabled:
            s.out[0] = 1
        else:
            s.out[0] = 0

    step(sim)
    assert sim.out.to_numpy()[0] == 1
    sim.enabled = False
    step(sim)
    assert sim.out.to_numpy()[0] == 0


# ---------------------------------------------------------------------------
# 4. Negative / signed int value.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_signed_int():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.delta = -5
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.delta

    step(sim)
    assert sim.out.to_numpy()[0] == -5
    sim.delta = -123
    step(sim)
    assert sim.out.to_numpy()[0] == -123


# ---------------------------------------------------------------------------
# 5. Primitive inside a nested ``@dataclasses.dataclass`` member is lifted too.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_nested_dataclass():
    @dataclasses.dataclass
    class Params:
        nt: int
        gain: float

    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.params = Params(nt=2, gain=3.0)
            self.x = qd.field(qd.f32, shape=8)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        for i in range(s.params.nt):
            s.x[i] += s.params.gain

    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [3, 3, 0, 0, 0, 0, 0, 0])
    sim.params.nt = 5
    sim.params.gain = 1.0
    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [4, 4, 1, 1, 1, 0, 0, 0])


# ---------------------------------------------------------------------------
# 6. Primitive inside a nested ``@qd.data_oriented`` member is lifted too.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_nested_data_oriented():
    @qd.data_oriented
    class Inner:
        def __init__(self):
            self.count = 2

    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.inner = Inner()
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.inner.count

    step(sim)
    assert sim.out.to_numpy()[0] == 2
    sim.inner.count = 9
    step(sim)
    assert sim.out.to_numpy()[0] == 9


# ---------------------------------------------------------------------------
# 7. Member (``self``) kernel: same lifting applies to the implicit data_oriented ``self`` template arg.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_self_method_kernel():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.amt = 1
            self.out = qd.field(qd.i32, shape=1)

        @qd.kernel
        def step(self):
            self.out[0] = self.amt + 1

    sim = Sim()
    sim.step()
    assert sim.out.to_numpy()[0] == 2
    sim.amt = 40
    sim.step()
    assert sim.out.to_numpy()[0] == 41


# ---------------------------------------------------------------------------
# 8. Pruning: only the primitives the kernel actually accesses become kernel args. A class with far more primitive
#    members than MAX_ARG_NUM (512) compiles fine because the unused ones are never declared.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_pruning_unused_not_declared():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            for i in range(600):
                setattr(self, f"unused_{i}", i)
            self.used = 7
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.used

    # Declaring all 600+ reachable primitives would exceed MAX_ARG_NUM; pruning keeps only ``used``.
    step(sim)
    assert sim.out.to_numpy()[0] == 7
    sim.used = 11
    step(sim)
    assert sim.out.to_numpy()[0] == 11


# ---------------------------------------------------------------------------
# 9. Using a lifted primitive inside ``qd.static`` is a hard error (it requires a compile-time constant).
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_in_static_raises():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.n = 4
            self.out = qd.field(qd.i32, shape=8)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        for i in range(qd.static(s.n)):
            s.out[i] = i

    with pytest.raises(Exception, match="runtime kernel argument"):
        step(sim)


# ---------------------------------------------------------------------------
# 10. Opt-in boundary: the default ``@qd.data_oriented`` (template_primitives=True) still bakes primitives, so a
#     mutation has no effect until re-specialisation. This pins that the new behaviour is strictly opt-in.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_default_template_primitives_still_bakes():
    @qd.data_oriented
    class Sim:
        def __init__(self):
            self.n = 3
            self.x = qd.field(qd.f32, shape=8)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        for i in range(s.n):
            s.x[i] += 1.0

    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [1, 1, 1, 0, 0, 0, 0, 0])
    # n is baked as 3; mutating it has no effect on the already-compiled kernel (same spec key -> no recompile).
    sim.n = 6
    step(sim)
    np.testing.assert_array_equal(sim.x.to_numpy(), [2, 2, 2, 0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# 11. fastcache=True / pure kernel: the used-primitive subset survives the two-pass / fastcache machinery.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_fastcache_pure():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.k = 2
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel(fastcache=True)
    def step(s: qd.template()):
        s.out[0] = s.k * 10

    step(sim)
    assert sim.out.to_numpy()[0] == 20
    sim.k = 5
    step(sim)
    assert sim.out.to_numpy()[0] == 50


# ---------------------------------------------------------------------------
# 12. Two distinct instances of the same flagged class each read their own live values.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_distinct_instances():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self, k):
            self.k = k
            self.out = qd.field(qd.i32, shape=1)

    a = Sim(3)
    b = Sim(8)

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.k

    step(a)
    step(b)
    assert a.out.to_numpy()[0] == 3
    assert b.out.to_numpy()[0] == 8
    a.k = 100
    step(a)
    assert a.out.to_numpy()[0] == 100
    assert b.out.to_numpy()[0] == 8


# ---------------------------------------------------------------------------
# 13. dtype is frozen at first compile: a member lifted as an integer that is later mutated to a float is rejected at
#     launch (coercing it would silently truncate), rather than corrupting the value.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_int_to_float_mutation_raises():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.k = 2  # int at first compile -> lifted as an integer kernel argument
            self.out = qd.field(qd.i32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.k

    step(sim)
    assert sim.out.to_numpy()[0] == 2

    # Reassigning the same member to a float would silently truncate against the frozen integer dtype (int(1.5) == 1),
    # so the launch path rejects it instead of corrupting the value.
    sim.k = 1.5
    with pytest.raises(TypeError, match="truncate"):
        step(sim)


# ---------------------------------------------------------------------------
# 14. The lossless directions still coerce (they are not rejected): int -> a float-lifted member, and int -> a
#     bool-lifted (integer-kind) member.
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_runtime_primitive_lossless_coercions_still_bind():
    @qd.data_oriented(template_primitives=False)
    class Sim:
        def __init__(self):
            self.scale = 2.0  # float at first compile -> lifted as a float argument
            self.flag = True  # bool at first compile -> lifted as an integer argument
            self.out = qd.field(qd.f32, shape=1)

    sim = Sim()

    @qd.kernel
    def step(s: qd.template()):
        s.out[0] = s.scale * s.flag

    step(sim)
    assert sim.out.to_numpy()[0] == 2.0

    # int -> float-lifted arg and int -> integer-kind arg are both lossless, so they bind (coerce) rather than raise.
    sim.scale = 3
    sim.flag = 2
    step(sim)
    assert sim.out.to_numpy()[0] == 6.0
