"""Kinematic root moved by a prescribed pose every step (translation and yaw), a dynamic child hanging from it on a revolute."""

import math

import numpy as np
import warp as wp

from rigid_scene import DT, box_inertia, run

HZ = 0.2
ROOT = wp.vec3(0.0, 0.0, 1.5)


def build(builder):
    root = builder.add_link(xform=wp.transform(ROOT, wp.quat_identity()), mass=1.0,
                            inertia=box_inertia(1.0, 0.05, 0.05, 0.05), is_kinematic=True)
    child = builder.add_link(xform=wp.transform(ROOT + wp.vec3(0.0, 0.0, -0.05 - HZ), wp.quat_identity()),
                             mass=0.5, inertia=box_inertia(0.5, 0.03, 0.03, HZ))
    j0 = builder.add_joint_free(child=root)
    j1 = builder.add_joint_revolute(parent=root, child=child, axis=wp.vec3(1.0, 0.0, 0.0),
                                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
                                    child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity()))
    builder.add_articulation([j0, j1])


def before_step(step, state):
    t = (step + 1) * DT
    p = ROOT + wp.vec3(0.0, 0.3 * math.sin(8.0 * t), 0.1 * math.sin(5.0 * t))
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 2.0 * t)
    return 0, np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], np.float32)


if __name__ == "__main__":
    run("kinematic_child", build, steps=120, before_step=before_step)
