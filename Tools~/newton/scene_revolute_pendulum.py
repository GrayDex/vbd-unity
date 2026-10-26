"""Single revolute pendulum: a rod hinged to the world about X, released from 0.8 rad (joint_q, so rest angle is nonzero)."""

import warp as wp

from rigid_scene import box_inertia, run

HZ = 0.4


def build(builder):
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 - HZ), wp.quat_identity()),
                            mass=1.0, inertia=box_inertia(1.0, 0.05, 0.05, HZ))
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(1.0, 0.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
                                       child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity()))
    builder.add_articulation([joint])
    builder.joint_q[-1] = 0.8


if __name__ == "__main__":
    run("revolute_pendulum", build, steps=120)
