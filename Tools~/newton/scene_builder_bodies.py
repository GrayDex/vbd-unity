"""ModelBuilder rigid bodies and shapes: densities, offset transforms, explicit mass properties, inertia correction.

Spec rows are written to the fixture, so the C# test builds the same scene from them.
Also the rigid body graph coloring of a joint graph (chain plus random links).
Run in the Newton env: python scene_builder_bodies.py
"""

import numpy as np
import warp as wp

import newton
from newton._src.sim.graph_coloring import ColoringAlgorithm, color_rigid_bodies
from fixture import Fixture, group_colors, fixtures_dir

PLANE, PLANE_EQ, GROUND = 1, -1, -2
SPHERE, CAPSULE, BOX = 3, 4, 7


def quat(axis, angle):
    q = wp.quat_from_axis_angle(wp.normalize(wp.vec3(*axis)), angle)
    return [q[0], q[1], q[2], q[3]]


ID = [0.0, 0.0, 0.0, 1.0]

# px py pz qx qy qz qw  comx comy comz  mass  Ixx Iyy Izz Ixy Ixz Iyz  lock
BODIES = [
    [0, 0, 0, *ID, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 3, *quat((0, 1, 0), 0.6), 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, *ID, 0.1, 0.0, 0.05, 2.0, 0.5, 0.6, 0.7, 0.01, -0.02, 0.03, 0],
    [0, 1, 0, *ID, 0, 0, 0, 1.0, 0.1, 0.1, 0.1, 0, 0, 0, 1],
    [2, 0, 0, *ID, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 5, *quat((1, 1, 0), 1.0), 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, *ID, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0],
    [4, 0, 0, *ID, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, *ID, 0, 0, 0, 3.0, 1.0, 1.0, 5.0, 0, 0, 0, 0],
]

# body type  px py pz qx qy qz qw  s0 s1 s2  density margin ke kd mu  solid particles visible gap
SHAPES = [
    [1, SPHERE, 0.2, 0, 0, *ID, 0.3, 0, 0, 500.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
    [1, CAPSULE, 0, 0.3, 0, *quat((1, 0, 0), 0.5), 0.1, 0.4, 0, 800.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
    [1, BOX, -0.1, 0.1, 0.4, *quat((1, 2, 3), 0.9), 0.2, 0.3, 0.1, 300.0, 0.01, 1.0e3, 50.0, 0.4, 1, 0, 1, 0.05],
    [2, BOX, 0.3, 0, 0, *ID, 0.25, 0.1, 0.15, 1000.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
    [3, SPHERE, 0, 0, 0, *ID, 0.4, 0, 0, 1000.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
    [5, SPHERE, 0, 0, 0.1, *ID, 0.5, 0, 0, 1000.0, 0.02, 2500.0, 100.0, 1.0, 0, 1, 1, -1],
    [5, BOX, 0, 0.2, 0, *quat((0, 0, 1), 0.3), 0.2, 0.2, 0.3, 700.0, 0.01, 2500.0, 100.0, 1.0, 0, 1, 1, -1],
    [6, CAPSULE, 0, 0, 0, *ID, 0.2, 0.3, 0, 0.0, 0.0, 2500.0, 100.0, 0.7, 1, 1, 1, -1],
    [6, BOX, 0.5, 0, 0, *ID, 0.1, 0.1, 0.1, 0.0, 0.0, 2500.0, 100.0, 0.2, 1, 0, 0, -1],
    [-1, PLANE, 0, 0, 0, *ID, 5.0, 4.0, 0, 0.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
    [-1, SPHERE, 0, 0, 1.0, *ID, 0.25, 0, 0, 1000.0, 0.0, 2500.0, 100.0, 0.6, 1, 1, 1, -1],
    [-1, PLANE_EQ, 0, 1.0, 0.5, *ID, -0.5, 10.0, 8.0, 0.0, 0.0, 800.0, 20.0, 0.3, 1, 1, 1, -1],
    [-1, GROUND, 0, 0, 0, *ID, 0.2, 0, 0, 0.0, 0.0, 2500.0, 100.0, 1.0, 1, 1, 1, -1],
]


def build():
    b = newton.ModelBuilder()
    for r in BODIES:
        inertia = wp.mat33(r[11], r[14], r[15], r[14], r[12], r[16], r[15], r[16], r[13])
        b.add_body(xform=wp.transform(wp.vec3(*r[0:3]), wp.quat(*r[3:7])), com=wp.vec3(*r[7:10]), mass=r[10],
                   inertia=inertia, lock_inertia=bool(r[17]))
    for r in SHAPES:
        body, kind = int(r[0]), int(r[1])
        cfg = newton.ModelBuilder.ShapeConfig(density=r[12], margin=r[13], ke=r[14], kd=r[15], mu=r[16],
                                              is_solid=bool(r[17]), has_particle_collision=bool(r[18]),
                                              is_visible=bool(r[19]), gap=None if r[20] < 0 else r[20])
        xf = wp.transform(wp.vec3(*r[2:5]), wp.quat(*r[5:9]))
        s = r[9:12]
        if kind == SPHERE:
            b.add_shape_sphere(body, xform=xf, radius=s[0], cfg=cfg)
        elif kind == CAPSULE:
            b.add_shape_capsule(body, xform=xf, radius=s[0], half_height=s[1], cfg=cfg)
        elif kind == BOX:
            b.add_shape_box(body, xform=xf, hx=s[0], hy=s[1], hz=s[2], cfg=cfg)
        elif kind == PLANE:
            b.add_shape_plane(xform=xf, width=s[0], length=s[1], body=body, cfg=cfg)
        elif kind == PLANE_EQ:
            b.add_shape_plane(plane=(r[2], r[3], r[4], s[0]), width=s[1], length=s[2], body=body, cfg=cfg)
        elif kind == GROUND:
            b.add_ground_plane(height=s[0], cfg=cfg)
    return b


def main():
    b = build()
    model = b.finalize()
    fx = Fixture("builder_bodies")
    fx.array("in_bodies", np.array(BODIES, np.float32))
    fx.array("in_shapes", np.array(SHAPES, np.float32))
    for name in ("body_q", "body_qd", "body_com", "body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia",
                 "shape_body", "shape_transform", "shape_type", "shape_scale", "shape_flags", "shape_margin",
                 "shape_gap", "shape_material_ke", "shape_material_kd", "shape_material_mu"):
        fx.array(name, getattr(model, name).numpy())

    rng = np.random.default_rng(3)
    n = 40
    parents = list(range(-1, n - 1)) + [int(x) for x in rng.integers(0, n, 30)]
    children = list(range(0, n)) + [int(x) for x in rng.integers(0, n, 30)]
    fx.array("joint_parent", np.array(parents, np.int32))
    fx.array("joint_child", np.array(children, np.int32))
    for name, algorithm, balance in (("mcs_bal", ColoringAlgorithm.MCS, True), ("mcs_raw", ColoringAlgorithm.MCS, False),
                                     ("greedy_bal", ColoringAlgorithm.GREEDY, True),
                                     ("greedy_raw", ColoringAlgorithm.GREEDY, False)):
        groups = color_rigid_bodies(n, parents, children, balance_colors=balance, algorithm=algorithm)
        fx.array(f"body_colors_{name}", group_colors(groups, n))
    fx.save(fixtures_dir() / "builder_bodies.vbdf")
    print(f"builder_bodies: {model.body_count} bodies, {model.shape_count} shapes")
    for name in ("body_mass", "body_inv_mass", "body_inertia"):
        print(name, getattr(model, name).numpy().reshape(model.body_count, -1)[:, :3])


if __name__ == "__main__":
    main()
