"""External forces: State.body_f and Control.joint_f as Newton's SolverVBD feeds them into the forward step.

A free body pushed by body_f (force and torque) and by its FREE joint's joint_f at once; joint_f on a revolute hung
from the world, on a prismatic slider and the revolute below it (reaction on a dynamic parent), on a ball, on a D6
with one linear and two angular axes, on a D6 with three angular axes, and on a FREE joint whose parent is a body;
body_f also on a jointed body. Forces change every step and are written to the fixture with the run.
Also the shared runner of the scenes that build their own model (no eval_fk: the model's body_q is the start pose).
Run in the Newton env: python scene_external_forces.py [--noise] (--noise: Newton's CPU against its CUDA run, and the
CUDA run against itself with the forces scaled by 1 + 1e-6).
"""

import math
import sys

import numpy as np
import warp as wp

import newton
from fixture import Fixture, fixtures_dir
from noise_floor import rotation_deg
from rigid_scene import DT, ITERATIONS, body_color_groups, box_inertia

HZ = 0.15
JOINT_KEYS = ("joint_type", "joint_enabled", "joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_axis",
              "joint_q_start", "joint_qd_start", "joint_dof_dim", "joint_q", "joint_qd", "joint_f", "joint_target_ke",
              "joint_target_kd", "joint_limit_lower", "joint_limit_upper", "joint_limit_ke", "joint_limit_kd")
BODY_KEYS = ("body_q", "body_qd", "body_com", "body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia",
             "body_flags")


def model_arrays(fx, model, control):
    """Every model array the package's Model holds, plus the solver parameters and the body coloring."""
    fx.array("gravity", model.gravity.numpy()[0])
    for key in BODY_KEYS + JOINT_KEYS:
        fx.array(key, getattr(model, key).numpy())
    colors, offsets = body_color_groups(model)
    fx.array("body_color_bodies", colors)
    fx.array("body_color_offsets", offsets)
    fx.array("joint_target_q", control.joint_target_q.numpy())
    fx.array("joint_target_qd", control.joint_target_qd.numpy())


def simulate(name, build, steps, forces=None, device=None, save=True, dt=DT, iterations=ITERATIONS):
    """build(builder); forces(step, t) -> (body_f [n, 6] or None, joint_f [dofs] or None) applied before each step."""
    builder = newton.ModelBuilder()
    build(builder)
    builder.color()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverVBD(model=model, iterations=iterations)
    control = model.control()
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
    model_arrays(fx, model, control)

    n = model.body_count
    q = np.zeros((steps, n, 7), np.float32)
    body_f = np.zeros((steps, n, 6), np.float32)
    joint_f = np.zeros((steps, model.joint_dof_count), np.float32)
    for i in range(steps):
        state_0.clear_forces()
        if forces is not None:
            bf, jf = forces(i, (i + 1) * dt)
            if bf is not None:
                body_f[i] = bf
                state_0.body_f.assign(body_f[i])
            if jf is not None:
                joint_f[i] = jf
                control.joint_f.assign(joint_f[i])
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        q[i] = state_0.body_q.numpy()
    if forces is not None:
        fx.array("applied_body_f", body_f)
        fx.array("applied_joint_f", joint_f)
    fx.array("steps_body_q", q)
    if save:
        fx.save(fixtures_dir() / f"{name}.vbdf")
    print(f"{name}: {n} bodies, {model.joint_count} joints, {model.joint_dof_count} dofs, "
          f"{len(body_color_groups(model)[1]) - 1} colors, travel {np.abs(q[-1, :, :3] - q[0, :, :3]).max() * 1e3:.1f} mm")
    return q


def noise(name, build, steps, forces=None, eps=1e-6):
    """Worst deviation between Newton's CPU and CUDA runs of the scene, per step horizon and per body; then how far
    the CUDA run moves when every applied force is scaled by 1 + eps, the scene's sensitivity to input rounding."""
    def report(label, a, b):
        dp = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
        dr = rotation_deg(a[..., 3:], b[..., 3:])
        for horizon in (10, 30, 60, steps):
            print(f"{label} {name} first {horizon} steps: {dp[:horizon].max() * 1e3:.5f} mm, "
                  f"{dr[:horizon].max():.5f} deg; per body mm {np.round(dp[:horizon].max(axis=0) * 1e3, 5)}, "
                  f"deg {np.round(dr[:horizon].max(axis=0), 5)}")

    runs = [simulate(name, build, steps, forces, device=d, save=False) for d in ("cpu", "cuda:0")]
    report("NOISE", runs[0], runs[1])
    if forces is not None:
        def scaled(step, t):
            bf, jf = forces(step, t)
            return bf * np.float32(1 + eps), jf * np.float32(1 + eps)

        report(f"SENSITIVITY {eps:g}", runs[1], simulate(name, build, steps, scaled, device="cuda:0", save=False))


def link(builder, center, mass, rot=None, kinematic=False, half=(0.04, 0.05)):
    rot = wp.quat_identity() if rot is None else rot
    return builder.add_link(xform=wp.transform(wp.vec3(*center), rot), mass=mass,
                            inertia=box_inertia(mass, half[0], half[1], HZ), is_kinematic=kinematic)


def at(*p, rot=None):
    return wp.transform(wp.vec3(*p), wp.quat_identity() if rot is None else rot)


def build(builder):
    cfg = newton.ModelBuilder.JointDofConfig
    # Free body; its FREE joint is joint 0 (DOFs 0-5).
    builder.add_body(xform=at(0.0, 0.0, 1.0, rot=wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.4)),
                     com=wp.vec3(0.02, -0.01, 0.03), mass=1.5, inertia=box_inertia(1.5, 0.1, 0.15, 0.2))
    # Revolute pendulum from the world (joint 1, DOF 6).
    b = link(builder, (1.0, 0.0, 1.5 - HZ), 0.8)
    builder.add_articulation([builder.add_joint_revolute(-1, b, axis=wp.vec3(1.0, 0.0, 0.0),
                                                         parent_xform=at(1.0, 0.0, 1.5), child_xform=at(0.0, 0.0, HZ))])
    # Slider along a tilted axis with a revolute below it (joints 2, 3; DOFs 7, 8).
    slider = link(builder, (2.0, 0.0, 1.5), 1.0)
    arm = link(builder, (2.0, 0.0, 1.5 - 0.05 - HZ), 0.5)
    builder.add_articulation([
        builder.add_joint_prismatic(-1, slider, axis=wp.vec3(0.0, 1.0, 1.0), parent_xform=at(2.0, 0.0, 1.5)),
        builder.add_joint_revolute(slider, arm, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=at(0.0, 0.0, -0.05),
                                   child_xform=at(0.0, 0.0, HZ))])
    # A ball pendulum from the world (joint 4, DOFs 9-11). A D6 from the world (x slide; x, y swing, twist locked),
    # a three-axis angular D6 below it, and a FREE joint hanging a body off that one (joints 5-7; DOFs 12-14, 15-17,
    # 18-23). Each free twist belongs to one body: a twist shared by a chain integrates the solver's rounding.
    top = link(builder, (-1.0, 0.0, 2.0 - HZ), 0.7, half=(0.12, 0.12))
    mid = link(builder, (-1.5, 0.0, 2.0 - HZ), 0.6, half=(0.12, 0.12))
    low = link(builder, (-1.5, 0.0, 2.0 - 3 * HZ), 0.5, half=(0.12, 0.12))
    tail = link(builder, (-1.2, 0.1, 2.0 - 3 * HZ), 0.3, rot=wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.5))
    builder.add_articulation([builder.add_joint_ball(-1, top, parent_xform=at(-1.0, 0.0, 2.0), child_xform=at(0.0, 0.0, HZ))])
    builder.add_articulation([
        builder.add_joint_d6(-1, mid, linear_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.03, limit_upper=0.03)],
                             angular_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0)), cfg(axis=wp.vec3(0.0, 1.0, 0.0))],
                             parent_xform=at(-1.5, 0.0, 2.0), child_xform=at(0.0, 0.0, HZ)),
        builder.add_joint_d6(mid, low, angular_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0)), cfg(axis=wp.vec3(0.0, 1.0, 0.0)),
                                                     cfg(axis=wp.vec3(0.0, 0.0, 1.0))],
                             parent_xform=at(0.0, 0.0, -HZ), child_xform=at(0.0, 0.0, HZ)),
        builder.add_joint_free(tail, parent=low, parent_xform=at(0.1, 0.0, 0.0))])


def forces(step, t):
    s, c = math.sin(20.0 * t), math.cos(15.0 * t)
    body_f = np.zeros((8, 6), np.float32)
    body_f[0] = [3.0 * (1 + 0.5 * s), -2.0, 12.0, 0.4, -0.3 * c, 0.2]
    body_f[2] = [0.0, 1.5 * c, 0.0, 0.0, 0.0, 0.1 * s]
    joint_f = np.array([1.0, 0.5 * s, 2.0, 0.05, 0.1 * c, -0.05,
                        0.5 * c,
                        4.0 * s, 0.3 * c,
                        0.1, 0.0, -0.05 * s,
                        2.0 * c, 0.2 * s, -0.15,
                        0.1 * c, -0.1, 0.05 * s,
                        0.5, -0.3 * s, 1.0, 0.02, 0.0, -0.03 * c], np.float32)
    return body_f, joint_f


STEPS = 120

if __name__ == "__main__":
    if "--noise" in sys.argv:
        noise("external_forces", build, STEPS, forces)
    else:
        simulate("external_forces", build, STEPS, forces)
