"""Revolute pendulum on a tilted axis (0, 1, 1): the axis has several nonzero components, so the joint projector
I - a a^T only removes the free rotation if the axis is exactly unit length."""

import warp as wp

import newton
from rigid_scene import box_inertia, run

HZ = 0.15


def build(builder):
    top = wp.vec3(0.0, 0.0, 1.5)
    body = builder.add_link(xform=wp.transform(top - wp.vec3(0.0, 0.0, HZ), wp.quat_identity()),
                            mass=0.4, inertia=box_inertia(0.4, 0.05, 0.05, HZ))
    joint = builder.add_joint_revolute(-1, body, axis=newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 1.0, 1.0)),
                                       parent_xform=wp.transform(top, wp.quat_identity()),
                                       child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity()))
    builder.add_articulation([joint])
    builder.joint_q[-1] = 0.7


if __name__ == "__main__":
    run("tilted_hinge", build, steps=120)
