"""ModelBuilder joints: every joint type SolverVBD solves, built with newton.ModelBuilder, finalized, then stepped.

A chain from the world (revolute with drive and limits, prismatic with limits, fixed, a D6 with a limited
slide and two angular axes, ball, a FREE joint whose parent is a body); a body from add_body (FREE joint, articulation)
with a shape adding mass; a kinematic root on a FREE joint with a three-axis angular D6 (twist driven: no free twist
shared by two bodies) and a revolute below it (its axis given as a JointDofConfig, not normalized, target outside the
limits); joints to the world outside any articulation: a driven prismatic and a disabled ball.
The revolute's axis is not tilted: under a moving parent a tilted hinge axis with limits parts from Newton in the
port's joint solve (an open issue of the rigid part, not of the builder). Tests/Editor/JointBuilderTests.cs builds
the same scene.
Run in the Newton env: python scene_builder_joints.py [--noise]
"""

import sys

import warp as wp

import newton
from rigid_scene import box_inertia
from scene_external_forces import at, noise, simulate

HZ = 0.15


def rot(axis, angle):
    return wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle)


def build(b):
    cfg = newton.ModelBuilder.JointDofConfig

    def link(center, mass, q=None, kinematic=False, half=(0.05, 0.05, HZ)):
        return b.add_link(xform=at(*center, rot=q), mass=mass, inertia=box_inertia(mass, *half), is_kinematic=kinematic)

    # The ball comes last in the chain, so its free twist belongs to one body: a twist shared by several links
    # integrates the solver's rounding.
    b0 = link((0.0, 0.0, 2.0 - HZ), 1.0)
    b1 = link((0.0, 0.0, 2.0 - 3 * HZ), 0.8)
    b2 = link((0.0, 0.0, 2.0 - 5 * HZ), 0.6, q=rot((0, 0, 1), 0.4), half=(0.12, 0.12, HZ))
    b3 = link((0.0, 0.0, 2.0 - 7 * HZ), 0.5, half=(0.12, 0.12, HZ))
    b4 = link((0.0, 0.0, 2.0 - 9 * HZ), 0.4, half=(0.12, 0.12, HZ))
    b5 = link((0.3, 0.1, 2.0 - 9 * HZ), 0.3, q=rot((1, 1, 0), 0.5))
    b.add_articulation([
        b.add_joint_revolute(-1, b0, axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=at(0.0, 0.0, 2.0),
                             child_xform=at(0.0, 0.0, HZ), target_pos=0.3, target_ke=50.0, target_kd=2.0,
                             limit_lower=-1.0, limit_upper=1.0),
        b.add_joint_prismatic(b0, b1, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=at(0.0, 0.0, -HZ),
                              child_xform=at(0.0, 0.0, HZ), limit_lower=-0.05, limit_upper=0.05, limit_ke=2.0e4,
                              limit_kd=5.0),
        b.add_joint_fixed(b1, b2, parent_xform=at(0.0, 0.0, -HZ), child_xform=at(0.0, 0.0, HZ, rot=rot((0, 0, 1), -0.4))),
        b.add_joint_d6(b2, b3, linear_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.02, limit_upper=0.02)],
                       angular_axes=[cfg(axis=wp.vec3(0.0, 1.0, 0.0), target_pos=0.2, target_ke=20.0, target_kd=1.0),
                                     cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.5, limit_upper=0.5,
                                         limit_ke=1.0e3, limit_kd=2.0)],
                       parent_xform=at(0.0, 0.0, -HZ, rot=rot((0, 0, 1), -0.4)), child_xform=at(0.0, 0.0, HZ)),
        b.add_joint_ball(b3, b4, parent_xform=at(0.0, 0.0, -HZ), child_xform=at(0.0, 0.0, HZ)),
        b.add_joint_free(b5, parent=b4, parent_xform=at(0.1, 0.0, 0.0))])

    b6 = b.add_body(xform=at(1.0, 0.0, 1.0, rot=rot((0, 1, 0), 0.3)), com=wp.vec3(0.02, 0.0, 0.01), mass=2.0,
                    inertia=box_inertia(2.0, 0.1, 0.1, 0.1))
    b.add_shape_sphere(b6, xform=at(0.05, 0.0, 0.0), radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=300.0))

    b7 = link((-1.0, 0.0, 1.5), 1.0, kinematic=True, half=(0.05, 0.05, 0.05))
    b8 = link((-1.0, 0.0, 1.5 - 0.05 - HZ), 0.5)
    b9 = link((-1.0, 0.0, 1.5 - 0.05 - 3 * HZ), 0.4)
    b.add_articulation([
        b.add_joint_free(b7),
        b.add_joint_d6(b7, b8, angular_axes=[cfg(axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-0.6, limit_upper=0.6),
                                             cfg(axis=wp.vec3(0.0, 1.0, 0.0), target_pos=0.1, target_ke=30.0,
                                                 target_kd=1.0),
                                             cfg(axis=wp.vec3(0.0, 0.0, 1.0), target_ke=5.0, target_kd=0.5)],
                       parent_xform=at(0.0, 0.0, -0.05), child_xform=at(0.0, 0.0, HZ)),
        b.add_joint_revolute(b8, b9, axis=cfg(axis=wp.vec3(0.0, 2.0, 0.0), limit_lower=-1.0, limit_upper=0.8,
                                               target_pos=1.5, target_ke=10.0),
                             parent_xform=at(0.0, 0.0, -HZ), child_xform=at(0.0, 0.0, HZ))])

    b10 = link((0.0, 1.0, 1.0), 0.7)
    b.add_joint_prismatic(-1, b10, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=at(0.0, 1.0, 1.0), target_pos=0.1,
                          target_vel=0.2, target_ke=100.0, target_kd=5.0)
    b11 = link((0.0, -1.0, 1.0 - HZ), 0.6)
    b.add_joint_ball(-1, b11, parent_xform=at(0.0, -1.0, 1.0), child_xform=at(0.0, 0.0, HZ), enabled=False)


STEPS = 120

if __name__ == "__main__":
    if "--noise" in sys.argv:
        noise("builder_joints", build, STEPS)
    else:
        simulate("builder_joints", build, STEPS)
