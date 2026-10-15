"""Time three unchanged Genesis MPM kernels with legacy sizing and initial-forward counting.

Run with Genesis and this Quadrants checkout installed, GS_ENABLE_NDARRAY=1:
    python scripts/benchmark_adstack_genesis_forward_count.py --envs 2 --repeats 30 --output results.json

Restore primal state and seed gradients outside timing. Alternate warmed legacy and replay samples; include
Python dispatch and synchronization in timings. This measures individual kernel calls, not whole scene steps.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", choices=("compute_F_tmp", "p2g", "g2p"))
    args = parser.parse_args()
    os.environ["GS_ENABLE_NDARRAY"] = "1"
    import genesis as gs

    import quadrants as qd

    gs.init(backend=gs.gpu, precision="32", seed=0, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-4, substeps=1, substeps_local=2, requires_grad=True),
        mpm_options=gs.options.MPMOptions(lower_bound=(0, 0, 0), upper_bound=(0.6, 0.6, 0.6)),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(lower=(0.2, 0.2, 0.2), upper=(0.4, 0.4, 0.3)),
        material=gs.materials.MPM.Elastic(rho=500),
    )
    scene.build(n_envs=args.envs)
    entity.set_velocity(gs.tensor([0.1, -0.2, 0.05], requires_grad=True))
    scene.step()
    solver = scene.mpm_solver
    rigid = scene.sim.coupler.rigid_solver
    calls = {
        "compute_F_tmp": (solver.compute_F_tmp, (0,)),
        "p2g": (
            solver.p2g,
            (
                0,
                rigid.dyn_state.geoms,
                rigid.dyn_info.geoms,
                rigid.dyn_state.links,
                rigid.rigid_info,
                rigid.collider._sdf._sdf_info,
                rigid.collider.collider_config,
            ),
        ),
        "g2p": (solver.g2p, (0, rigid.dyn_info.geoms, rigid.dyn_state.links, rigid.rigid_info)),
    }
    particle_state = solver.particles.to_numpy()
    grid_state = solver.grid.to_numpy()
    active_state = solver.particles_ng.to_numpy()
    result = {
        "quadrants_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "genesis_commit": subprocess.check_output(
            ["git", "-C", str(Path(gs.__file__).parents[1]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "envs": args.envs,
        "particles_per_env": int(solver.n_particles),
        "grid_resolution": [int(n) for n in solver._grid_res],
        "precision": "32",
        "repeats": args.repeats,
        "kernels": {},
    }
    for name, (kernel, kernel_args) in calls.items():
        if args.kernel and name != args.kernel:
            continue
        print(f"BENCHMARK {name}", flush=True)
        samples = {False: [], True: []}
        reference = None
        for iteration in range(args.repeats + 3):
            for counted in (False, True) if iteration % 2 == 0 else (True, False):
                solver.particles.from_numpy(particle_state)
                solver.particles_ng.from_numpy(active_state)
                solver.grid.from_numpy(grid_state)
                if name == "p2g":
                    solver.grid.mass.fill(0)
                    solver.grid.vel_in.fill(0)
                solver.particles.grad.fill(0.001)
                solver.grid.grad.fill(0.001)
                qd.sync()
                start = time.perf_counter()
                if counted:
                    with qd.adstack as scope:
                        kernel(*kernel_args)
                else:
                    kernel(*kernel_args)
                qd.sync()
                forward_done = time.perf_counter()
                kernel.grad(*kernel_args)
                qd.sync()
                done = time.perf_counter()
                if counted:
                    for invocation in scope.records:
                        invocation.release()
                    # The old dispatcher retains an empty pending-record marker to reject unmatched backwards.
                    # Reset that routing marker outside timing so this same callable can also measure legacy AD.
                    assert not kernel._adjoint._adstack_pending
                    del kernel._adjoint._adstack_pending
                if iteration >= 3:
                    samples[counted].append(
                        [(forward_done - start) * 1000, (done - forward_done) * 1000, (done - start) * 1000]
                    )
                if iteration == 0:
                    values = {}
                    for prefix, field in (("particles", solver.particles), ("grid", solver.grid)):
                        for kind, data in (("primal", field.to_numpy()), ("grad", field.grad.to_numpy())):
                            values.update({f"{prefix}.{kind}.{key}": value for key, value in data.items()})
                    if not counted:
                        reference = values
                    else:
                        for key, value in values.items():
                            np.testing.assert_allclose(value, reference[key], rtol=2e-4, atol=2e-5, err_msg=key)
                        assert any(
                            np.any(np.abs(value - 0.001) > 1e-5) for key, value in values.items() if ".grad." in key
                        ), "Backward did not change any seeded gradient"
        stats = {}
        for counted, rows in samples.items():
            stats["forward_count" if counted else "legacy"] = {
                label: statistics.median(row[i] for row in rows)
                for i, label in enumerate(("forward_ms", "backward_ms", "pair_ms"))
            }
            stats["forward_count" if counted else "legacy"]["samples_ms"] = rows
        stats["backward_ratio"] = stats["forward_count"]["backward_ms"] / stats["legacy"]["backward_ms"]
        stats["pair_ratio"] = stats["forward_count"]["pair_ms"] / stats["legacy"]["pair_ms"]
        stats["primal_and_gradient_check"] = "passed"
        result["kernels"][name] = stats
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({name: {k: v for k, v in stats.items() if k not in ("legacy", "forward_count")}}), flush=True)
    print(json.dumps(result, indent=2), flush=True)
    gs.destroy()


if __name__ == "__main__":
    main()
