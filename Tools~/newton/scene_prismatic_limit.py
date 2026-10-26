"""Prismatic slider on an inclined axis, sliding down under gravity until it hits its lower limit."""

import warp as wp

from rigid_scene import box_inertia, run


def build(builder):
    incline = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5)
    anchor = wp.vec3(0.0, 0.0, 1.0)
    body = builder.add_link(xform=wp.transform(anchor, incline), mass=2.0, inertia=box_inertia(2.0, 0.1, 0.1, 0.05))
    joint = builder.add_joint_prismatic(parent=-1, child=body, axis=wp.vec3(0.0, 0.0, 1.0),
                                        parent_xform=wp.transform(anchor, incline),
                                        child_xform=wp.transform_identity(),
                                        limit_lower=-0.03, limit_upper=0.2, limit_ke=1.0e4, limit_kd=10.0)
    builder.add_articulation([joint])


if __name__ == "__main__":
    run("prismatic_limit", build, steps=120)
