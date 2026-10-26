"""D6 pendulum: one limited linear axis and two limited angular axes (twist about the third locked), released tilted."""

import warp as wp

import newton
from rigid_scene import box_inertia, run

HZ = 0.3


def build(builder):
    anchor = wp.vec3(0.0, 0.0, 2.0)
    body = builder.add_link(xform=wp.transform(anchor + wp.vec3(0.0, 0.0, -HZ), wp.quat_identity()),
                            mass=1.0, inertia=box_inertia(1.0, 0.05, 0.08, HZ))
    cfg = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        parent=-1, child=body,
        linear_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.02, limit_upper=0.02, limit_ke=1.0e4, limit_kd=5.0)],
        angular_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.15, limit_upper=0.15, limit_ke=1.0e3, limit_kd=2.0),
                      cfg(axis=wp.vec3(0.0, 1.0, 0.0), limit_lower=-0.1, limit_upper=0.1, limit_ke=1.0e3, limit_kd=2.0)],
        parent_xform=wp.transform(anchor, wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, HZ), wp.quat_identity()))
    builder.add_articulation([joint])
    # Start beyond both angular limits so they act from the first steps.
    builder.joint_q[-2] = 0.25
    builder.joint_q[-1] = -0.15


if __name__ == "__main__":
    run("d6_limits", build, steps=120)
