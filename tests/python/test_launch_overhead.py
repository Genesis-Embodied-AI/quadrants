#!/usr/bin/env python3
"""
Quadrants kernel-launch overhead benchmarks.

These measure the Python-side cost of launching kernels with various parameter patterns,
independent of kernel computation. The kernels themselves do trivial work — we only care
about the launch path (argument processing, caching, struct traversal).

Scenarios modeled after Genesis rigid-body simulation patterns:
- Many frozen-dataclass struct parameters (mimics forward_kinematics, constraint_solver)
- Mix of cacheable (struct) and uncacheable (torch.Tensor) args (mimics set_dofs_position)
- Template annotations vs struct annotations (before/after always-fastcache migration)

Run with: pytest benchmarks/launch_overhead/bench_launch_overhead.py -v
Filter:   pytest benchmarks/launch_overhead/bench_launch_overhead.py -k "cache_miss"
"""
import dataclasses
import time

import pytest
import torch

import quadrants as qd

N_WARMUP = 50
N_STEPS = 2000
N_TRIALS = 5

pytestmark = [pytest.mark.benchmarks]


# ---------------------------------------------------------------------------
# Struct definitions — mimics Genesis array_class patterns (10-15 fields each)
# ---------------------------------------------------------------------------


def _make_struct_class(name, n_fields):
    """Dynamically create a frozen dataclass with n_fields, each typed as qd.Tensor."""
    fields = [(f"f{i}", qd.Tensor, dataclasses.field(default=None)) for i in range(n_fields)]
    return dataclasses.make_dataclass(name, fields, frozen=True)


StructA = _make_struct_class("StructA", 10)
StructB = _make_struct_class("StructB", 12)
StructC = _make_struct_class("StructC", 8)
StructD = _make_struct_class("StructD", 14)
StructE = _make_struct_class("StructE", 6)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qd_init():
    qd.init(arch=qd.cpu)


@pytest.fixture(scope="module")
def structs(qd_init):
    def _make(cls):
        fields = dataclasses.fields(cls)
        return cls(**{f.name: qd.field(qd.f32, shape=(4,)) for f in fields})

    return _make(StructA), _make(StructB), _make(StructC), _make(StructD), _make(StructE)


@pytest.fixture(scope="module")
def out_field(qd_init):
    return qd.field(qd.i32, shape=())


@pytest.fixture(scope="module")
def kernel_many_structs(out_field):
    @qd.kernel
    def _k(s1: StructA, s2: StructB, s3: StructC, s4: StructD, s5: StructE):
        out_field[None] = 1
    return _k


@pytest.fixture(scope="module")
def kernel_structs_plus_tensor(out_field):
    @qd.kernel
    def _k(s1: StructA, s2: StructB, s3: StructC, s4: StructD, s5: StructE, t: qd.types.ndarray()):
        out_field[None] = 1
    return _k


@pytest.fixture(scope="module")
def kernel_template_annotations(out_field):
    @qd.kernel
    def _k(s1: qd.template(), s2: qd.template(), s3: qd.template(), s4: qd.template(), s5: qd.template()):
        out_field[None] = 1
    return _k


@pytest.fixture(scope="module")
def kernel_template_plus_tensor(out_field):
    @qd.kernel
    def _k(
        s1: qd.template(), s2: qd.template(), s3: qd.template(), s4: qd.template(), s5: qd.template(),
        t: qd.types.ndarray(),
    ):
        out_field[None] = 1
    return _k


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _measure(step_fn, n_warmup=N_WARMUP, n_steps=N_STEPS, n_trials=N_TRIALS):
    for _ in range(n_warmup):
        step_fn()

    results = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        elapsed = time.perf_counter() - t0
        results.append(n_steps / elapsed)

    return sorted(results)[len(results) // 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_many_structs_cached(structs, kernel_many_structs):
    """5 all-Field struct args, cache hits after warmup (best case)."""
    sa, sb, sc, sd, se = structs
    fps = _measure(lambda: kernel_many_structs(sa, sb, sc, sd, se))
    print(f"\nmany_structs_cached: {fps:,.0f} launches/s")


def test_many_structs_cache_miss(structs, kernel_structs_plus_tensor):
    """5 all-Field struct args + torch.Tensor that changes id each call (cache miss)."""
    sa, sb, sc, sd, se = structs

    def step():
        t = torch.zeros(4, dtype=torch.float32)
        kernel_structs_plus_tensor(sa, sb, sc, sd, se, t)

    fps = _measure(step)
    print(f"\nmany_structs_cache_miss: {fps:,.0f} launches/s")


def test_template_cached(structs, kernel_template_annotations):
    """5 template args, cache hits (baseline — no struct traversal)."""
    sa, sb, sc, sd, se = structs
    fps = _measure(lambda: kernel_template_annotations(sa, sb, sc, sd, se))
    print(f"\ntemplate_cached: {fps:,.0f} launches/s")


def test_template_cache_miss(structs, kernel_template_plus_tensor):
    """5 template args + torch.Tensor (baseline with cache miss)."""
    sa, sb, sc, sd, se = structs

    def step():
        t = torch.zeros(4, dtype=torch.float32)
        kernel_template_plus_tensor(sa, sb, sc, sd, se, t)

    fps = _measure(step)
    print(f"\ntemplate_cache_miss: {fps:,.0f} launches/s")
