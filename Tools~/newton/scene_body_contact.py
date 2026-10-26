"""Body-body contacts on the analytic shape pairs: bodies fall onto the ground and onto each other, a kinematic
box pushes a sphere. Spheres stack, a capsule lands on the ground and a crossing capsule on it, a sphere lands on a
box; Newton's CollisionPipeline + SolverVBD at their defaults. With BOXES=True (scene_body_stack.py) boxes stack
and a capsule lands on a box: the GJK/MPR pairs.

Run in the Newton env: python scene_body_contact.py
"""

import math

import numpy as np
import warp as wp

import newton
from fixture import Fixture, fixtures_dir
from rigid_scene import body_color_groups

DT = 1.0 / 600.0
ITERATIONS = 10
STEPS = 300
PUSH_SPEED = 0.5
ALONG_X = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
ALONG_Y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)


def tilt(q, axis, angle):
    return wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle) * q


def pusher_pose(start, step):
    p = start + wp.vec3(PUSH_SPEED * (step + 1) * DT, 0.0, 0.0)
    return np.array([p[0], p[1], p[2], 0.0, 0.0, 0.0, 1.0], np.float32)


def build(builder, boxes):
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=10.0, mu=0.5, density=500.0)
    builder.add_ground_plane(cfg=cfg)
    bodies = []

    def body(pos, rot=None, kinematic=False):
        b = builder.add_link(xform=wp.transform(pos, rot if rot is not None else wp.quat_identity()),
                             is_kinematic=kinematic)
        bodies.append(b)
        return b

    if not boxes:
        builder.add_shape_sphere(body(wp.vec3(0.0, 0.0, 0.3)), radius=0.1, cfg=cfg)
        builder.add_shape_sphere(body(wp.vec3(0.03, 0.0, 0.6)), radius=0.08, cfg=cfg)
        builder.add_shape_capsule(body(wp.vec3(1.0, 0.0, 0.2), tilt(ALONG_X, (0, 1, 0), 0.1)), radius=0.05,
                                  half_height=0.15, cfg=cfg)
        builder.add_shape_capsule(body(wp.vec3(1.02, 0.0, 0.45), ALONG_Y), radius=0.04, half_height=0.12, cfg=cfg)
        builder.add_shape_box(body(wp.vec3(2.0, 0.0, 0.25), tilt(wp.quat_identity(), (1, 1, 0), 0.15)),
                              hx=0.1, hy=0.08, hz=0.06, cfg=cfg)
        builder.add_shape_sphere(body(wp.vec3(2.02, 0.01, 0.55)), radius=0.06, cfg=cfg)
        pusher_start = wp.vec3(3.0, 0.0, 0.1)
        builder.add_shape_box(body(pusher_start, kinematic=True), hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_shape_sphere(body(wp.vec3(3.25, 0.01, 0.08)), radius=0.08, cfg=cfg)
    else:
        builder.add_shape_box(body(wp.vec3(0.0, 0.0, 0.07)), hx=0.1, hy=0.1, hz=0.06, cfg=cfg)
        builder.add_shape_box(body(wp.vec3(0.01, 0.0, 0.2), tilt(wp.quat_identity(), (0, 0, 1), 0.3)),
                              hx=0.08, hy=0.08, hz=0.06, cfg=cfg)
        builder.add_shape_box(body(wp.vec3(0.0, 0.01, 0.34), tilt(wp.quat_identity(), (1, 0, 0), 0.05)),
                              hx=0.06, hy=0.06, hz=0.05, cfg=cfg)
        builder.add_shape_box(body(wp.vec3(1.0, 0.0, 0.07)), hx=0.15, hy=0.1, hz=0.06, cfg=cfg)
        builder.add_shape_capsule(body(wp.vec3(1.0, 0.0, 0.23), tilt(ALONG_X, (0, 0, 1), 0.4)), radius=0.04,
                                  half_height=0.1, cfg=cfg)
        pusher_start = wp.vec3(2.0, 0.0, 0.1)
        builder.add_shape_box(body(pusher_start, kinematic=True), hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_shape_box(body(wp.vec3(2.25, 0.01, 0.07)), hx=0.06, hy=0.06, hz=0.06, cfg=cfg)
    for b in bodies:
        builder.add_articulation([builder.add_joint_free(child=b)])
    kinematic = len(bodies) - 2
    return kinematic, pusher_start


def run(name, boxes, device=None, save=True):
    with wp.ScopedDevice(device):
        return simulate(name, boxes, save)


def simulate(name, boxes, save):
    builder = newton.ModelBuilder()
    kinematic, pusher_start = build(builder, boxes)
    builder.color()
    model = builder.finalize()
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = newton.solvers.SolverVBD(model=model, iterations=ITERATIONS)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    fx = Fixture(name)
    for key, value in (("dt", DT), ("iterations", ITERATIONS), ("steps", STEPS), ("kinematic_body", kinematic),
                       ("friction_epsilon", solver.friction_epsilon),
                       ("rigid_joint_linear_ke", solver.rigid_joint_linear_ke),
                       ("rigid_joint_angular_ke", solver.rigid_joint_angular_ke),
                       ("rigid_avbd_gamma", solver.rigid_avbd_gamma),
                       ("rigid_contact_alpha", solver.rigid_contact_alpha)):
        fx.param(key, value)
    fx.array("gravity", model.gravity.numpy()[0])
    for key in ("shape_body", "shape_transform", "shape_type", "shape_scale", "shape_flags", "shape_margin",
                "shape_gap", "shape_collision_group", "shape_material_ke", "shape_material_kd", "shape_material_mu",
                "body_q", "body_qd", "body_com", "body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia",
                "body_flags", "joint_type", "joint_enabled", "joint_parent", "joint_child", "joint_X_p", "joint_X_c",
                "joint_axis", "joint_q_start", "joint_qd_start", "joint_dof_dim", "joint_q", "joint_target_ke",
                "joint_target_kd", "joint_limit_lower", "joint_limit_upper", "joint_limit_ke", "joint_limit_kd"):
        fx.array(key, getattr(model, key).numpy())
    filters = sorted(model.shape_collision_filter_pairs)
    fx.array("shape_collision_filter_pairs", np.array(filters, np.int32).reshape(-1, 2))
    fx.array("joint_target_q", control.joint_target_q.numpy())
    fx.array("joint_target_qd", control.joint_target_qd.numpy())
    colors, offsets = body_color_groups(model)
    fx.array("body_color_bodies", colors)
    fx.array("body_color_offsets", offsets)

    nb = model.body_count
    q = np.zeros((STEPS, nb, 7), np.float32)
    qd = np.zeros((STEPS, nb, 6), np.float32)
    counts = np.zeros(STEPS, np.int32)
    prescribed = np.zeros((STEPS, 7), np.float32)
    for i in range(STEPS):
        prescribed[i] = pusher_pose(pusher_start, i)
        body_q = state_0.body_q.numpy()
        body_q[kinematic] = prescribed[i]
        state_0.body_q.assign(body_q)
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        counts[i] = contacts.rigid_contact_count.numpy()[0]
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.body_q.numpy()
        qd[i] = state_0.body_qd.numpy()
    fx.array("prescribed_q", prescribed)
    fx.array("steps_body_q", q)
    fx.array("steps_body_qd", qd)
    fx.array("steps_contact_count", counts)
    if save:
        fx.save(fixtures_dir() / f"{name}.vbdf")
    print(f"{name}: {nb} bodies, contacts {counts.min()}..{counts.max()}, final z {np.round(q[-1, :, 2], 4)}, "
          f"pushed x {q[-1, -1, 0]:.4f}")
    return q, qd


if __name__ == "__main__":
    run("body_contact", boxes=False)
