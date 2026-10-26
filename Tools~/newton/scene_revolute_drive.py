"""Revolute joint with a PD position drive: starts at 0.3 rad (rest angle), driven to a 1.2 rad target against gravity."""

import numpy as np
import warp as wp

from rigid_scene import box_inertia, run

HZ = 0.25
TARGET = 1.2


def build(builder):
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0 - HZ), wp.quat_identity()),
                            mass=0.8, inertia=box_inertia(0.8, 0.04, 0.04, HZ))
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(0.0, 1.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                                       child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity()),
                                       target_ke=200.0, target_kd=2.0, limit_lower=-0.5, limit_upper=1.5)
    builder.add_articulation([joint])
    builder.joint_q[-1] = 0.3


def prepare(model, control):
    target = control.joint_target_q.numpy()
    target[:] = np.float32(TARGET)
    control.joint_target_q.assign(target)


if __name__ == "__main__":
    run("revolute_drive", build, steps=120, prepare=prepare)
