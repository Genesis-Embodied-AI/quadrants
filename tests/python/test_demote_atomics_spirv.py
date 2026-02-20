"""Tests for atomic demotion correctness on SPIR-V backends.

Regression test for a Metal shader compiler bug where demoting atomics to
load-op-store in serial tasks, combined with the cache_loop_invariant_global_vars
optimisation pass, produces incorrect results.  The caching pass may remove a
nearby device-memory write from the loop body, leaving a lone load-increment-store
that the Metal compiler incorrectly optimises (caching/hoisting the load across
loop iterations).

The fix: on SPIR-V backends (Metal/Vulkan), global atomics in serial tasks are
kept as real atomic operations so that cache_loop_invariant_global_vars cannot
interfere with them.
"""

import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_serial_atomic_counter_with_nested_loops():
    """Atomic counter inside nested loops with data-dependent bounds.

    Two atomic_add calls share the innermost loop body.  When one is
    cache-eligible (loop-invariant pointer, written but never read outside
    the conditional), the caching pass may hoist it to a local variable.
    On Metal this caused the *other* atomic's demoted load-op-store to be
    miscompiled, producing a stale counter value.
    """
    n_links = 9
    n_joints = 8
    n_dofs = 12
    n_batches = 1
    max_constraints = 1500

    link_jnt_start = qd.field(dtype=qd.i32, shape=(n_links,))
    link_jnt_end = qd.field(dtype=qd.i32, shape=(n_links,))
    jnt_dof_start = qd.field(dtype=qd.i32, shape=(n_joints,))
    jnt_dof_end = qd.field(dtype=qd.i32, shape=(n_joints,))
    frictionloss = qd.field(dtype=qd.f32, shape=(n_dofs,))

    n_constraints = qd.field(dtype=qd.i32, shape=(n_batches,))
    n_constraints_fl = qd.field(dtype=qd.i32, shape=(n_batches,))
    efc_frictionloss = qd.field(dtype=qd.f32, shape=(max_constraints, n_batches))

    link_jnt_start.from_numpy(np.array([0, 1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32))
    link_jnt_end.from_numpy(np.array([1, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32))
    jnt_dof_start.from_numpy(np.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32))
    jnt_dof_end.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 12], dtype=np.int32))
    frictionloss.from_numpy(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    n_constraints.fill(0)
    n_constraints_fl.fill(0)
    efc_frictionloss.fill(0.0)

    @qd.kernel
    def add_inequality_constraints():
        qd.loop_config(serialize=True)
        for i_b in range(n_batches):
            n_constraints_fl[i_b] = 0
            for i_l in range(n_links):
                for i_j in range(link_jnt_start[i_l], link_jnt_end[i_l]):
                    for i_d in range(jnt_dof_start[i_j], jnt_dof_end[i_j]):
                        if frictionloss[i_d] > 0.0:
                            i_con = qd.atomic_add(n_constraints[i_b], 1)
                            qd.atomic_add(n_constraints_fl[i_b], 1)
                            efc_frictionloss[i_con, i_b] = frictionloss[i_d]

    add_inequality_constraints()

    assert n_constraints[0] == 6
    assert n_constraints_fl[0] == 6
    np.testing.assert_allclose(
        efc_frictionloss.to_numpy()[:12, 0],
        frictionloss.to_numpy(),
    )


@test_utils.test()
def test_serial_atomic_counter_simple():
    """Simplified variant: single level of data-dependent looping.

    A counter is atomically incremented inside a conditional, and a second
    atomic (to a different field) is also present.  The caching pass may
    cache the second atomic's target, leaving the first as a lone
    device-memory RMW — which Metal miscompiles.
    """
    n = 16
    data = qd.field(dtype=qd.f32, shape=(n,))
    counter = qd.field(dtype=qd.i32, shape=())
    counter2 = qd.field(dtype=qd.i32, shape=())
    output = qd.field(dtype=qd.f32, shape=(n,))

    data.from_numpy(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32))
    counter.fill(0)
    counter2.fill(0)
    output.fill(0.0)

    @qd.kernel
    def count_nonzero():
        qd.loop_config(serialize=True)
        for _ in range(1):
            for i in range(n):
                if data[i] > 0.0:
                    idx = qd.atomic_add(counter[None], 1)
                    qd.atomic_add(counter2[None], 1)
                    output[idx] = data[i]

    count_nonzero()

    expected_count = 8
    assert counter[None] == expected_count
    assert counter2[None] == expected_count
    for i in range(expected_count):
        assert output[i] == 1.0
