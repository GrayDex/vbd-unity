"""Free body with distinct principal inertias spinning near its intermediate axis (gyroscopic term), falling under gravity."""

import numpy as np
import warp as wp

from rigid_scene import run


def build(builder):
    rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.4)
    inertia = wp.mat33(0.03, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.07)
    builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), rot), com=wp.vec3(0.02, -0.01, 0.03),
                     mass=1.5, inertia=inertia)


def prepare(model, control):
    qd = model.body_qd.numpy()
    qd[0] = np.array([0.5, 0.0, 1.0, 0.2, 15.0, 0.3], np.float32)
    model.body_qd.assign(qd)


if __name__ == "__main__":
    run("free_spin", build, steps=120, prepare=prepare)
