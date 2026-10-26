"""Chain of 5 links on ball joints, the first hung from the world, released tilted so it swings in 3D."""

import warp as wp

from rigid_scene import box_inertia, run

LINKS = 5
HZ = 0.1


def build(builder):
    tilt = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.4, 0.0)), 1.0)
    anchor = wp.vec3(0.0, 0.0, 2.0)
    joints = []
    parent = -1
    for i in range(LINKS):
        center = anchor + wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, -HZ * (2 * i + 1)))
        body = builder.add_link(xform=wp.transform(center, tilt), mass=0.5 + 0.1 * i,
                                inertia=box_inertia(0.5 + 0.1 * i, 0.03, 0.03, HZ))
        parent_xform = (wp.transform(anchor, tilt) if parent < 0
                        else wp.transform(wp.vec3(0.0, 0.0, -HZ), wp.quat_identity()))
        joints.append(builder.add_joint_ball(parent=parent, child=body, parent_xform=parent_xform,
                                             child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity())))
        parent = body
    builder.add_articulation(joints)


if __name__ == "__main__":
    run("ball_chain", build, steps=120)
