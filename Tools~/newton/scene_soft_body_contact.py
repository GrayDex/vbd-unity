"""Bodies on a soft block: a free sphere and a free box fall onto a tetrahedral block with a pinned bottom,
a kinematic capsule presses into it and slides; contacts push particles and bodies both ways.

Run in the Newton env: python scene_soft_body_contact.py
"""

import math

import numpy as np
import warp as wp

import newton
from fixture import Fixture, color_groups, fixtures_dir
from rigid_scene import body_color_groups, box_inertia

# Grid cells along x, height, depth: the grid is turned so its y (fix_bottom pins y == 0) points up.
DIM = (8, 2, 4)
CELL = 0.05
DENSITY = 1.0e3
K_MU, K_LAMBDA, K_DAMP = 2.0e4, 2.0e4, 1.0e1
PARTICLE_RADIUS = 0.01
DT = 1.0 / 600.0
ITERATIONS = 10
STEPS = 300
CAPSULE = wp.vec3(0.13, 0.0, 0.15)
# The capsule lies along y: Newton's capsule axis is z.
CAPSULE_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)


def sphere_inertia(mass, r):
    i = 0.4 * mass * r * r
    return wp.mat33(i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i)


def capsule_pose(step):
    t = (step + 1) * DT
    press = math.sin(math.pi * min(t / 0.3, 1.0) * 0.5)
    p = CAPSULE + wp.vec3(0.02 * math.sin(6.0 * t), 0.0, -0.05 * press)
    return np.array([p[0], p[1], p[2], CAPSULE_ROT[0], CAPSULE_ROT[1], CAPSULE_ROT[2], CAPSULE_ROT[3]], np.float32)


def build(builder):
    builder.add_soft_grid(
        pos=wp.vec3(-0.2, 0.1, 0.0), rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=DIM[0], dim_y=DIM[1], dim_z=DIM[2], cell_x=CELL, cell_y=CELL, cell_z=CELL,
        density=DENSITY, k_mu=K_MU, k_lambda=K_LAMBDA, k_damp=K_DAMP,
        fix_bottom=True, particle_radius=PARTICLE_RADIUS,
    )
    # Shapes carry no mass: body masses and inertias are set explicitly. No shape-shape collisions: body-body
    # contacts are not ported.
    cfg = newton.ModelBuilder.ShapeConfig(ke=5.0e3, kd=20.0, mu=0.6, margin=0.002, density=0.0,
                                          has_shape_collision=False)

    sphere = builder.add_link(xform=wp.transform(wp.vec3(-0.12, 0.0, 0.2), wp.quat_identity()),
                              mass=0.5, inertia=sphere_inertia(0.5, 0.04))
    builder.add_shape_sphere(sphere, radius=0.04, cfg=cfg)
    turn = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.4)
    box = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.18), turn),
                           mass=0.3, inertia=box_inertia(0.3, 0.04, 0.03, 0.02))
    builder.add_shape_box(box, hx=0.04, hy=0.03, hz=0.02, cfg=cfg)
    capsule = builder.add_link(xform=wp.transform(CAPSULE, CAPSULE_ROT), mass=1.0,
                               inertia=box_inertia(1.0, 0.03, 0.03, 0.09), is_kinematic=True)
    builder.add_shape_capsule(capsule, radius=0.03, half_height=0.06, cfg=cfg)
    joints = [builder.add_joint_free(child=b) for b in (sphere, box, capsule)]
    for j in joints:
        builder.add_articulation([j])
    return capsule


def main():
    builder = newton.ModelBuilder()
    capsule = build(builder)
    builder.color()
    model = builder.finalize()
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = newton.solvers.SolverVBD(model=model, iterations=ITERATIONS)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    fx = Fixture("soft_body_contact")
    for name, value in (("dt", DT), ("iterations", ITERATIONS), ("steps", STEPS), ("kinematic_body", capsule),
                        ("soft_contact_ke", model.soft_contact_ke), ("soft_contact_kd", model.soft_contact_kd),
                        ("soft_contact_mu", model.soft_contact_mu), ("soft_contact_gap", pipeline.soft_contact_gap),
                        ("friction_epsilon", solver.friction_epsilon),
                        ("rigid_joint_linear_ke", solver.rigid_joint_linear_ke),
                        ("rigid_joint_angular_ke", solver.rigid_joint_angular_ke),
                        ("rigid_avbd_gamma", solver.rigid_avbd_gamma)):
        fx.param(name, value)
    fx.array("gravity", model.gravity.numpy()[0])
    for name in ("particle_q", "particle_qd", "particle_mass", "particle_flags", "particle_radius",
                 "tet_indices", "tet_poses", "tet_materials", "shape_body", "shape_transform", "shape_type",
                 "shape_scale", "shape_flags", "shape_margin", "shape_material_ke", "shape_material_kd",
                 "shape_material_mu", "body_q", "body_qd", "body_com", "body_mass", "body_inv_mass",
                 "body_inertia", "body_inv_inertia", "body_flags", "joint_type", "joint_enabled", "joint_parent",
                 "joint_child", "joint_X_p", "joint_X_c", "joint_axis", "joint_q_start", "joint_qd_start",
                 "joint_dof_dim", "joint_q", "joint_target_ke", "joint_target_kd", "joint_limit_lower",
                 "joint_limit_upper", "joint_limit_ke", "joint_limit_kd"):
        fx.array(name, getattr(model, name).numpy())
    fx.array("joint_target_q", control.joint_target_q.numpy())
    fx.array("joint_target_qd", control.joint_target_qd.numpy())
    colors, color_offsets = color_groups(model)
    fx.array("color_particles", colors)
    fx.array("color_offsets", color_offsets)
    body_colors, body_color_offsets = body_color_groups(model)
    fx.array("body_color_bodies", body_colors)
    fx.array("body_color_offsets", body_color_offsets)

    n, nb = model.particle_count, model.body_count
    q = np.zeros((STEPS, n, 3), np.float32)
    qd = np.zeros_like(q)
    bq = np.zeros((STEPS, nb, 7), np.float32)
    bqd = np.zeros((STEPS, nb, 6), np.float32)
    counts = np.zeros(STEPS, np.int32)
    rigid_counts = np.zeros(STEPS, np.int32)
    prescribed = np.zeros((STEPS, 7), np.float32)
    for i in range(STEPS):
        prescribed[i] = capsule_pose(i)
        body_q = state_0.body_q.numpy()
        body_q[capsule] = prescribed[i]
        state_0.body_q.assign(body_q)
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        counts[i] = contacts.soft_contact_count.numpy()[0]
        rigid_counts[i] = contacts.rigid_contact_count.numpy()[0]
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.particle_q.numpy()
        qd[i] = state_0.particle_qd.numpy()
        bq[i] = state_0.body_q.numpy()
        bqd[i] = state_0.body_qd.numpy()
    assert rigid_counts.max() == 0, f"rigid contacts {rigid_counts.max()}"
    fx.array("prescribed_q", prescribed)
    fx.array("steps_q", q)
    fx.array("steps_qd", qd)
    fx.array("steps_body_q", bq)
    fx.array("steps_body_qd", bqd)
    fx.array("steps_contact_count", counts)
    fx.save(fixtures_dir() / "soft_body_contact.vbdf")
    print(f"soft_body_contact: {n} particles, {model.tet_count} tets, {nb} bodies, contacts max {counts.max()}, "
          f"sphere z {bq[0, 0, 2]:.4f} -> {bq[-1, 0, 2]:.4f}, box z {bq[0, 1, 2]:.4f} -> {bq[-1, 1, 2]:.4f}, "
          f"particles moved up to {np.linalg.norm(q - model.particle_q.numpy()[None], axis=2).max():.4f}")


if __name__ == "__main__":
    main()
