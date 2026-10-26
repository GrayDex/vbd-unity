"""Shared runner for the rigid-body scenes: builds the model, runs Newton's SolverVBD, writes a fixture.

A scene module calls run(name, build, ...) where build(builder) adds bodies and joints.
Defaults follow Newton's SolverVBD: compliant ALM, rigid_joint_linear_ke = rigid_joint_angular_ke = 1e5.
"""

import numpy as np
import warp as wp

import newton
from fixture import Fixture, fixtures_dir

DT = 1.0 / 600.0
ITERATIONS = 10


def body_color_groups(model):
    """Newton's body color groups flattened: body ids by color, and color start offsets."""
    groups = [g.numpy() for g in model.body_color_groups]
    offsets = np.cumsum([0] + [len(g) for g in groups])
    return (np.concatenate(groups) if groups else np.zeros(0, np.int32)), offsets


def box_inertia(mass, hx, hy, hz):
    """Inertia of a solid box with half extents (hx, hy, hz) about its center."""
    return wp.mat33(
        mass / 3.0 * (hy * hy + hz * hz), 0.0, 0.0,
        0.0, mass / 3.0 * (hx * hx + hz * hz), 0.0,
        0.0, 0.0, mass / 3.0 * (hx * hx + hy * hy),
    )


def run(name, build, steps, dt=DT, iterations=ITERATIONS, prepare=None, before_step=None, device=None, save=True):
    """build(builder) -> None; prepare(model, control) edits targets once;
    before_step(step, state) -> optional (body, pose) the caller prescribes before each step."""
    builder = newton.ModelBuilder()
    build(builder)
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = newton.solvers.SolverVBD(model=model, iterations=iterations)
    control = model.control()
    if prepare is not None:
        prepare(model, control)
    state_0, state_1 = model.state(), model.state()

    fx = Fixture(name)
    for key, value in (("dt", dt), ("iterations", iterations), ("steps", steps),
                       ("rigid_joint_linear_ke", solver.rigid_joint_linear_ke),
                       ("rigid_joint_angular_ke", solver.rigid_joint_angular_ke),
                       ("rigid_joint_linear_kd", solver.rigid_joint_linear_kd),
                       ("rigid_joint_angular_kd", solver.rigid_joint_angular_kd),
                       ("rigid_avbd_gamma", solver.rigid_avbd_gamma),
                       ("rigid_joint_alpha", solver.rigid_joint_alpha)):
        fx.param(key, value)
    fx.array("gravity", model.gravity.numpy()[0])
    fx.array("body_q", model.body_q.numpy())
    fx.array("body_qd", model.body_qd.numpy())
    fx.array("body_com", model.body_com.numpy())
    fx.array("body_mass", model.body_mass.numpy())
    fx.array("body_inv_mass", model.body_inv_mass.numpy())
    fx.array("body_inertia", model.body_inertia.numpy())
    fx.array("body_inv_inertia", model.body_inv_inertia.numpy())
    fx.array("body_flags", model.body_flags.numpy())
    colors, color_offsets = body_color_groups(model)
    fx.array("body_color_bodies", colors)
    fx.array("body_color_offsets", color_offsets)
    for key in ("joint_type", "joint_enabled", "joint_parent", "joint_child", "joint_X_p", "joint_X_c",
                "joint_axis", "joint_q_start", "joint_qd_start", "joint_dof_dim", "joint_q",
                "joint_target_ke", "joint_target_kd", "joint_limit_lower", "joint_limit_upper",
                "joint_limit_ke", "joint_limit_kd"):
        fx.array(key, getattr(model, key).numpy())
    fx.array("joint_target_q", control.joint_target_q.numpy())
    fx.array("joint_target_qd", control.joint_target_qd.numpy())

    n = model.body_count
    q = np.zeros((steps, n, 7), np.float32)
    qd = np.zeros((steps, n, 6), np.float32)
    prescribed = []
    for i in range(steps):
        if before_step is not None:
            body, pose = before_step(i, state_0)
            body_q = state_0.body_q.numpy()
            body_q[body] = pose
            state_0.body_q.assign(body_q)
            prescribed.append(pose)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.body_q.numpy()
        qd[i] = state_0.body_qd.numpy()
    if prescribed:
        fx.array("prescribed_q", np.asarray(prescribed, np.float32))
    fx.array("steps_body_q", q)
    fx.array("steps_body_qd", qd)
    if save:
        fx.save(fixtures_dir() / f"{name}.vbdf")
    print(f"{name}: {n} bodies, {model.joint_count} joints, {len(color_offsets) - 1} colors, "
          f"last body p {q[-1, -1, :3]}")
    return q, qd
