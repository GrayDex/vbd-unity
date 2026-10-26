"""Tetrahedral grid fixed at one end, sagging under gravity: particle VBD with Stable Neo-Hookean and damping.

Run in the Newton env: python scene_hanging_grid.py
"""

import numpy as np
import warp as wp

import newton
from fixture import Fixture, color_groups, fixtures_dir

DIM = (6, 2, 2)
CELL = 0.1
DENSITY = 1.0e3
K_MU, K_LAMBDA, K_DAMP = 1.0e5, 1.0e5, 1.0e2
DT = 1.0 / 600.0
ITERATIONS = 10
STEPS = 60


def main():
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=DIM[0],
        dim_y=DIM[1],
        dim_z=DIM[2],
        cell_x=CELL,
        cell_y=CELL,
        cell_z=CELL,
        density=DENSITY,
        k_mu=K_MU,
        k_lambda=K_LAMBDA,
        k_damp=K_DAMP,
        fix_left=True,
    )
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model=model, iterations=ITERATIONS)
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    fx = Fixture("hanging_grid")
    for name, value in (("dt", DT), ("iterations", ITERATIONS), ("steps", STEPS), ("cell", CELL),
                        ("density", DENSITY), ("k_mu", K_MU), ("k_lambda", K_LAMBDA), ("k_damp", K_DAMP)):
        fx.param(name, value)
    fx.array("gravity", model.gravity.numpy()[0])
    fx.array("particle_q", model.particle_q.numpy())
    fx.array("particle_qd", model.particle_qd.numpy())
    fx.array("particle_mass", model.particle_mass.numpy())
    fx.array("particle_inv_mass", model.particle_inv_mass.numpy())
    fx.array("particle_flags", model.particle_flags.numpy())
    fx.array("tet_indices", model.tet_indices.numpy())
    fx.array("tet_poses", model.tet_poses.numpy())
    fx.array("tet_materials", model.tet_materials.numpy())
    colors, color_offsets = color_groups(model)
    fx.array("color_particles", colors)
    fx.array("color_offsets", color_offsets)
    adjacency = solver.particle_adjacency
    fx.array("v_adj_tets", adjacency.v_adj_tets.numpy())
    fx.array("v_adj_tets_offsets", adjacency.v_adj_tets_offsets.numpy())

    q = np.zeros((STEPS, model.particle_count, 3), np.float32)
    qd = np.zeros_like(q)
    for i in range(STEPS):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.particle_q.numpy()
        qd[i] = state_0.particle_qd.numpy()
    fx.array("steps_q", q)
    fx.array("steps_qd", qd)
    fx.save(fixtures_dir() / "hanging_grid.vbdf")
    print(f"hanging_grid: {model.particle_count} particles, {model.tet_count} tets, "
          f"{len(color_offsets) - 1} colors, tip z {q[-1, :, 2].min():.4f}")


if __name__ == "__main__":
    main()
