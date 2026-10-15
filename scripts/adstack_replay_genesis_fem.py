"""Exercise explicit FEM backward with replay counting in a Genesis environment.

Run with Genesis installed alongside this Quadrants checkout. Use --legacy for the existing sizing baseline.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    os.environ["QD_AD_STACK_REPLAY_COUNT"] = "0" if args.legacy else "1"

    import genesis as gs
    import numpy as np

    import quadrants as qd

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    dt, steps = 1e-4, 3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=1, requires_grad=True, gravity=(0, 0, 0)),
        fem_options=gs.options.FEMOptions(use_implicit_solver=False),
        show_viewer=False,
    )
    entity = scene.add_entity(
        morph=gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0, 0, 0.5)),
        material=gs.materials.FEM.Elastic(E=1e4, nu=0.3, rho=1000),
    )
    scene.build()
    velocity = gs.tensor([0.1, 0.0, 0.0], requires_grad=True)
    entity.set_velocity(velocity)
    for _ in range(steps):
        scene.step()
    solver = scene.fem_solver
    # Genesis currently lacks FEMEntity.collect_output_grads, even on the legacy path. Seed the solver fields
    # directly to exercise its four normal autodiff kernels without relying on that missing integration hook.
    qd.ad.clear_all_gradients()

    @qd.kernel
    def seed():
        for i in range(solver.n_vertices):
            solver.elements_v.grad[steps, i, 0].pos = qd.Vector([1.0 / solver.n_vertices, 0.0, 0.0])

    seed()
    for frame in reversed(range(steps)):
        solver.substep_post_coupling_grad(frame)
        solver.substep_pre_coupling_grad(frame)
    gradient = solver.elements_v.vel.grad.to_numpy()[0, :, 0, :].sum(axis=0)
    # Uniform translation with no external forces advances the mean x coordinate by steps * dt * velocity.x.
    np.testing.assert_allclose(gradient, [steps * dt, 0, 0], rtol=1e-5, atol=1e-9)
    print({"legacy": args.legacy, "velocity_gradient": gradient.tolist()})
    gs.destroy()


if __name__ == "__main__":
    main()
