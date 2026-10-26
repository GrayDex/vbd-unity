"""Welded pair: a link on a ball joint to the world and a second link welded to it off-axis, swinging as one body."""

import warp as wp

from rigid_scene import box_inertia, run


def build(builder):
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)
    anchor = wp.vec3(0.0, 0.0, 2.0)
    a = builder.add_link(xform=wp.transform(anchor + wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, -0.2)), tilt),
                         mass=1.0, inertia=box_inertia(1.0, 0.04, 0.04, 0.2))
    # B sits beside the lower end of A, rotated about A's axis.
    offset = wp.transform(wp.vec3(0.15, 0.0, -0.25), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5))
    b_pose = wp.transform_multiply(wp.transform(anchor + wp.quat_rotate(tilt, wp.vec3(0.0, 0.0, -0.2)), tilt), offset)
    b = builder.add_link(xform=b_pose, mass=0.6, inertia=box_inertia(0.6, 0.1, 0.03, 0.03))
    j0 = builder.add_joint_ball(parent=-1, child=a, parent_xform=wp.transform(anchor, wp.quat_identity()),
                                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_inverse(tilt)))
    j1 = builder.add_joint_fixed(parent=a, child=b, parent_xform=offset, child_xform=wp.transform_identity())
    builder.add_articulation([j0, j1])


if __name__ == "__main__":
    run("fixed_pair", build, steps=120)
