"""Kinematic palm moved every step with a finger-like chain: revolute (drive + limits), one-axis angular D6
(drive + limits), revolute with a velocity drive; plus a slider on a prismatic velocity drive with limits."""

import math

import numpy as np
import warp as wp

import newton
from rigid_scene import DT, box_inertia, run

PALM = wp.vec3(0.0, 0.0, 1.0)
HZ = 0.04


def build(builder):
    palm = builder.add_link(xform=wp.transform(PALM, wp.quat_identity()), mass=0.4,
                            inertia=box_inertia(0.4, 0.05, 0.05, 0.02), is_kinematic=True)
    joints = [builder.add_joint_free(child=palm)]
    cfg = newton.ModelBuilder.JointDofConfig
    parent, top = palm, PALM + wp.vec3(0.0, 0.0, -0.02)
    parent_offset = wp.vec3(0.0, 0.0, -0.02)
    for i in range(3):
        center = top + wp.vec3(0.0, 0.0, -HZ)
        body = builder.add_link(xform=wp.transform(center, wp.quat_identity()), mass=0.03,
                                inertia=box_inertia(0.03, 0.01, 0.01, HZ))
        px = wp.transform(parent_offset, wp.quat_identity())
        cx = wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity())
        if i == 0:
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=wp.vec3(1.0, 0.0, 0.0),
                                                     parent_xform=px, child_xform=cx, target_ke=2.0, target_kd=0.02,
                                                     limit_lower=-0.1, limit_upper=0.6, limit_ke=50.0, limit_kd=0.1))
        elif i == 1:
            joints.append(builder.add_joint_d6(parent=parent, child=body, parent_xform=px, child_xform=cx,
                                               angular_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), target_ke=1.0, target_kd=0.01,
                                                                 limit_lower=0.0, limit_upper=0.9, limit_ke=30.0, limit_kd=0.05)]))
        else:
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=wp.vec3(1.0, 0.0, 0.0),
                                                     parent_xform=px, child_xform=cx, target_kd=0.01))
        parent, top, parent_offset = body, center + wp.vec3(0.0, 0.0, -HZ), wp.vec3(0.0, 0.0, -HZ)
    slider = builder.add_link(xform=wp.transform(PALM + wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), mass=0.05,
                              inertia=box_inertia(0.05, 0.02, 0.02, 0.02))
    joints.append(builder.add_joint_prismatic(parent=palm, child=slider, axis=wp.vec3(0.0, 1.0, 0.0),
                                              parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                                              child_xform=wp.transform_identity(), target_kd=0.5,
                                              limit_lower=-0.02, limit_upper=0.03, limit_ke=200.0, limit_kd=1.0))
    builder.add_articulation(joints)
    builder.joint_q[builder.joint_q_start[1]] = 0.2


def prepare(model, control):
    target_q = control.joint_target_q.numpy()
    target_qd = control.joint_target_qd.numpy()
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    target_q[q_start[1]] = 0.5
    target_q[q_start[2]] = 1.2
    target_qd[qd_start[3]] = 2.0
    target_qd[qd_start[4]] = 0.4
    control.joint_target_q.assign(target_q)
    control.joint_target_qd.assign(target_qd)


def before_step(step, state):
    t = (step + 1) * DT
    p = PALM + wp.vec3(0.05 * math.sin(6.0 * t), 0.0, 0.03 * math.sin(9.0 * t))
    q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, 0.0, 1.0)), 1.5 * t)
    return 0, np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], np.float32)


if __name__ == "__main__":
    run("finger", build, steps=120, prepare=prepare, before_step=before_step)
