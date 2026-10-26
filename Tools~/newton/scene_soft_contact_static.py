"""Tetrahedral sheet dropped on world shapes (sphere, tilted capsule, rotated box, ground plane): particle-shape contacts.

Run in the Newton env: python scene_soft_contact_static.py
"""

import numpy as np
import warp as wp

import newton
from fixture import Fixture, color_groups, fixtures_dir

DIM = (10, 10, 2)
CELL = 0.05
DENSITY = 1.0e3
K_MU, K_LAMBDA, K_DAMP = 2.0e4, 2.0e4, 1.0e1
PARTICLE_RADIUS = 0.01
DT = 1.0 / 600.0
ITERATIONS = 10
STEPS = 240
MAX_RECORDS = 128


def main():
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(-0.25, -0.25, 0.25),
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
        particle_radius=PARTICLE_RADIUS,
    )
    cfg = newton.ModelBuilder.ShapeConfig(ke=5.0e3, kd=20.0, mu=0.6, margin=0.002)
    builder.add_ground_plane(cfg=cfg)
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(-0.12, -0.1, 0.12), wp.quat_identity()), radius=0.1, cfg=cfg)
    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, 0.0)), 1.2)
    builder.add_shape_capsule(-1, xform=wp.transform(wp.vec3(0.12, 0.05, 0.1), tilt), radius=0.04, half_height=0.12, cfg=cfg)
    turn = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 1.0, 0.5)), 0.7)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.18, 0.08), turn), hx=0.08, hy=0.05, hz=0.06, cfg=cfg)
    builder.color()
    model = builder.finalize()
    solver = newton.solvers.SolverVBD(model=model, iterations=ITERATIONS)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    fx = Fixture("soft_contact_static")
    for name, value in (("dt", DT), ("iterations", ITERATIONS), ("steps", STEPS),
                        ("soft_contact_ke", model.soft_contact_ke), ("soft_contact_kd", model.soft_contact_kd),
                        ("soft_contact_mu", model.soft_contact_mu), ("soft_contact_gap", pipeline.soft_contact_gap),
                        ("friction_epsilon", solver.friction_epsilon)):
        fx.param(name, value)
    fx.array("gravity", model.gravity.numpy()[0])
    for name in ("particle_q", "particle_qd", "particle_mass", "particle_flags", "particle_radius",
                 "tet_indices", "tet_poses", "tet_materials", "shape_body", "shape_transform", "shape_type",
                 "shape_scale", "shape_flags", "shape_margin", "shape_material_ke", "shape_material_kd",
                 "shape_material_mu"):
        fx.array(name, getattr(model, name).numpy())
    colors, color_offsets = color_groups(model)
    fx.array("color_particles", colors)
    fx.array("color_offsets", color_offsets)

    q = np.zeros((STEPS, model.particle_count, 3), np.float32)
    qd = np.zeros_like(q)
    counts = np.zeros(STEPS, np.int32)
    # Contact records found at the start of each step, padded with -1 particles.
    rec_particle = np.full((STEPS, MAX_RECORDS), -1, np.int32)
    rec_shape = np.full((STEPS, MAX_RECORDS), -1, np.int32)
    rec_body_pos = np.zeros((STEPS, MAX_RECORDS, 3), np.float32)
    rec_normal = np.zeros((STEPS, MAX_RECORDS, 3), np.float32)
    for i in range(STEPS):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        counts[i] = contacts.soft_contact_count.numpy()[0]
        c = min(counts[i], MAX_RECORDS)
        rec_particle[i, :c] = contacts.soft_contact_particle.numpy()[:c]
        rec_shape[i, :c] = contacts.soft_contact_shape.numpy()[:c]
        rec_body_pos[i, :c] = contacts.soft_contact_body_pos.numpy()[:c]
        rec_normal[i, :c] = contacts.soft_contact_normal.numpy()[:c]
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.particle_q.numpy()
        qd[i] = state_0.particle_qd.numpy()
    fx.array("steps_q", q)
    fx.array("steps_qd", qd)
    fx.array("steps_contact_count", counts)
    fx.array("steps_contact_particle", rec_particle)
    fx.array("steps_contact_shape", rec_shape)
    fx.array("steps_contact_body_pos", rec_body_pos)
    fx.array("steps_contact_normal", rec_normal)
    fx.save(fixtures_dir() / "soft_contact_static.vbdf")
    print(f"soft_contact_static: {model.particle_count} particles, {model.tet_count} tets, {model.shape_count} shapes, "
          f"contacts at last step {counts[-1]} (max {counts.max()}), lowest z {q[-1, :, 2].min():.4f}")


if __name__ == "__main__":
    main()
