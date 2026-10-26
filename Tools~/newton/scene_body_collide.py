"""Shape-shape contacts of Newton's CollisionPipeline for every pair of the port's shape types, with filtering.

Bodies are posed by a script (no solver): each step moves and turns them a little, so pairs go from deep contact
through the gap to apart. Clusters along x, one shape pair each; the last ones must be filtered (same body, joint,
collision group, no shape collision, filter pair, static pairs). Spec rows go to the fixture, so the C# test can
also build the scene with its ModelBuilder and compare the filtering.
Run in the Newton env: python scene_body_collide.py
"""

import math

import numpy as np
import warp as wp

import newton
from fixture import Fixture, fixtures_dir

STEPS = 60
PLANE, SPHERE, CAPSULE, BOX = 1, 3, 4, 7
REVOLUTE, FIXED, FREE = 1, 3, 4  # newton.JointType values used here
ALONG_X = (0.0, 1.0, 0.0, 0.5 * math.pi)  # capsule axis z turned onto x
ALONG_Y = (1.0, 0.0, 0.0, 0.5 * math.pi)

# body rows: x y z, rotation axis + angle, motion amplitude (m), motion phase
BODIES = [
    (0.0, 0.0, 0.10, (1, 0, 0, 0.0), 0.04, 0.0),  # 0 sphere on the plane
    (1.0, 0.0, 0.06, ALONG_X, 0.04, 0.5),  # 1 capsule on the plane
    (2.0, 0.0, 0.07, (1, 1, 0, 0.2), 0.04, 1.0),  # 2 box on the plane
    (3.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 3 sphere | 4 sphere
    (3.0, 0.15, 0.5, (0, 0, 1, 0.0), 0.04, 1.5),
    (4.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 5 sphere | 6 capsule
    (4.1, 0.13, 0.5, ALONG_X, 0.04, 2.0),
    (5.0, 0.0, 0.5, ALONG_X, 0.0, 0.0),  # 7 capsule | 8 crossing capsule
    (5.05, 0.0, 0.59, ALONG_Y, 0.04, 2.5),
    (6.0, 0.0, 0.5, ALONG_X, 0.0, 0.0),  # 9 capsule | 10 parallel capsule
    (6.03, 0.0, 0.59, ALONG_X, 0.04, 3.0),
    (7.0, 0.0, 0.5, (0, 0, 1, 0.3), 0.0, 0.0),  # 11 box | 12 sphere
    (7.03, 0.02, 0.62, (0, 0, 1, 0.0), 0.04, 3.5),
    (8.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 13 box | 14 capsule
    (8.02, 0.0, 0.62, (0, 0, 1, 0.4), 0.04, 4.0),
    (9.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 15 box | 16 tilted box
    (9.03, 0.02, 0.6, (1, 2, 0, 0.15), 0.04, 4.5),
    (10.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 17 box | 18 box face on face
    (10.0, 0.01, 0.6, (0, 0, 1, 0.5), 0.03, 5.0),
    (11.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 19 two shapes, same body
    (12.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 20 | 21 revolute joint: filtered
    (12.0, 0.1, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),
    (13.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 22 | 23 groups 2 and 3: filtered
    (13.0, 0.1, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),
    (14.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 24 | 25 no shape collision
    (14.0, 0.1, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),
    (15.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 26 | 27 explicit filter pair
    (15.0, 0.1, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),
    (16.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),  # 28 fixed to the world box: filtered
    (17.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),  # 29 revolute to the world box: collides
    (18.0, 0.0, 0.5, (0, 0, 1, 0.0), 0.0, 0.0),  # 30 group -1 | 31 group 2: collide
    (18.0, 0.1, 0.5, (0, 0, 1, 0.0), 0.02, 0.0),
]

# shape rows: body, type, scale, margin, gap (-1: builder default), group, has_shape_collision
S = 0.08
SHAPES = [
    (-1, PLANE, (0.0, 0.0, 0.0), 0.0, -1, 1, 1),
    (0, SPHERE, (0.1, 0, 0), 0.0, 0.02, 1, 1),
    (1, CAPSULE, (0.05, 0.15, 0), 0.005, 0.02, 1, 1),
    (2, BOX, (0.1, 0.08, 0.06), 0.0, -1, 1, 1),
    (3, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (4, SPHERE, (S, 0, 0), 0.004, 0.02, 1, 1),
    (5, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (6, CAPSULE, (0.05, 0.12, 0), 0.0, 0.02, 1, 1),
    (7, CAPSULE, (0.05, 0.15, 0), 0.0, 0.02, 1, 1),
    (8, CAPSULE, (0.04, 0.12, 0), 0.0, 0.02, 1, 1),
    (9, CAPSULE, (0.05, 0.15, 0), 0.0, 0.02, 1, 1),
    (10, CAPSULE, (0.04, 0.1, 0), 0.003, 0.02, 1, 1),
    (11, BOX, (0.1, 0.1, 0.05), 0.0, 0.02, 1, 1),
    (12, SPHERE, (0.07, 0, 0), 0.0, 0.02, 1, 1),
    (13, BOX, (0.12, 0.1, 0.05), 0.0, 0.02, 1, 1),
    (14, CAPSULE, (0.05, 0.1, 0), 0.0, 0.02, 1, 1),
    (15, BOX, (0.1, 0.1, 0.05), 0.0, 0.02, 1, 1),
    (16, BOX, (0.06, 0.05, 0.05), 0.0, 0.02, 1, 1),
    (17, BOX, (0.15, 0.15, 0.05), 0.0, 0.02, 1, 1),
    (18, BOX, (0.08, 0.06, 0.05), 0.002, 0.02, 1, 1),
    (19, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (19, BOX, (S, S, S), 0.0, 0.02, 1, 1),
    (20, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (21, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (22, SPHERE, (S, 0, 0), 0.0, 0.02, 2, 1),
    (23, SPHERE, (S, 0, 0), 0.0, 0.02, 3, 1),
    (24, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (25, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 0),
    (26, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (27, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (-1, BOX, (0.1, 0.1, 0.1), 0.0, 0.02, 1, 1),  # world box under body 28
    (28, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (-1, BOX, (0.1, 0.1, 0.1), 0.0, 0.02, 1, 1),  # world box under body 29
    (29, SPHERE, (S, 0, 0), 0.0, 0.02, 1, 1),
    (-1, SPHERE, (0.2, 0, 0), 0.0, 0.02, 1, 1),  # two overlapping world shapes: never a pair
    (-1, SPHERE, (0.2, 0, 0), 0.0, 0.02, 1, 1),
    (30, SPHERE, (S, 0, 0), 0.0, 0.02, -1, 1),
    (31, SPHERE, (S, 0, 0), 0.0, 0.02, 2, 1),
]
WORLD_POSES = {30: (16.0, 0.0, 0.33), 32: (17.0, 0.0, 0.33), 34: (19.0, 0.0, 0.5), 35: (19.0, 0.1, 0.5)}
FILTER_PAIRS = [(28, 29)]
# joint rows: type, parent, child, articulation
JOINTS = [(FREE, -1, 20, 0), (REVOLUTE, 20, 21, 0), (FIXED, -1, 28, 1), (REVOLUTE, -1, 29, 2)]


def body_pose(row, step):
    x, y, z, (ax, ay, az, angle), amp, phase = row
    t = 2.0 * math.pi * step / STEPS
    base = wp.quat_from_axis_angle(wp.normalize(wp.vec3(float(ax), float(ay), float(az))), angle)
    wobble = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 2.0 * amp * math.sin(t + phase))
    q = wobble * base
    return [x + 0.3 * amp * math.cos(t + phase), y, z - amp * math.sin(t + phase), q[0], q[1], q[2], q[3]]


def build(builder):
    for row in BODIES:
        p = body_pose(row, 0)
        builder.add_link(xform=wp.transform(wp.vec3(*p[:3]), wp.quat(*p[3:])), mass=1.0)
    for i, (body, kind, scale, margin, gap, group, collide) in enumerate(SHAPES):
        cfg = newton.ModelBuilder.ShapeConfig(margin=margin, gap=None if gap < 0 else gap, collision_group=group,
                                              has_shape_collision=bool(collide), density=0.0)
        pose = wp.transform(wp.vec3(*WORLD_POSES.get(i, (0.0, 0.0, 0.0))), wp.quat_identity())
        if kind == PLANE:
            builder.add_ground_plane(cfg=cfg)
        elif kind == SPHERE:
            builder.add_shape_sphere(body, xform=pose, radius=scale[0], cfg=cfg)
        elif kind == CAPSULE:
            builder.add_shape_capsule(body, xform=pose, radius=scale[0], half_height=scale[1], cfg=cfg)
        else:
            builder.add_shape_box(body, xform=pose, hx=scale[0], hy=scale[1], hz=scale[2], cfg=cfg)
    for a, b in FILTER_PAIRS:
        builder.add_shape_collision_filter_pair(a, b)
    articulations = {}
    for kind, parent, child, articulation in JOINTS:
        if kind == REVOLUTE:
            j = builder.add_joint_revolute(parent, child, axis=wp.vec3(0.0, 0.0, 1.0))
        elif kind == FREE:
            j = builder.add_joint_free(child, parent=parent)
        else:
            j = builder.add_joint_fixed(parent, child)
        articulations.setdefault(articulation, []).append(j)
    for joints in articulations.values():
        builder.add_articulation(joints)


def main():
    builder = newton.ModelBuilder()
    build(builder)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state = model.state()

    fx = Fixture("body_collide")
    fx.param("steps", STEPS)
    fx.array("body_rows", np.array([[*r[:3], *r[3], r[4], r[5]] for r in BODIES], np.float32))
    fx.array("shape_rows", np.array([[r[0], r[1], *r[2], r[3], r[4], r[5], r[6], *WORLD_POSES.get(i, (0, 0, 0))]
                                     for i, r in enumerate(SHAPES)], np.float32))
    fx.array("filter_rows", np.array(FILTER_PAIRS, np.int32))
    fx.array("joint_rows", np.array(JOINTS, np.int32))
    for name in ("body_q", "shape_body", "shape_transform", "shape_type", "shape_scale", "shape_flags",
                 "shape_margin", "shape_gap", "shape_collision_group", "shape_contact_pairs"):
        fx.array(name, getattr(model, name).numpy())
    fx.array("shape_collision_filter_pairs", np.array(sorted(model.shape_collision_filter_pairs), np.int32))

    nb = model.body_count
    poses = np.zeros((STEPS, nb, 7), np.float32)
    records = []
    for i in range(STEPS):
        poses[i] = [body_pose(r, i) for r in BODIES]
        state.body_q.assign(poses[i])
        pipeline.collide(state, contacts)
        n = int(contacts.rigid_contact_count.numpy()[0])
        records.append([getattr(contacts, f"rigid_contact_{k}").numpy()[:n].reshape(n, -1)
                        for k in ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal",
                                  "margin0", "margin1")])
    width = max(len(r[0]) for r in records)
    counts = np.array([len(r[0]) for r in records], np.int32)
    table = np.full((STEPS, width, 19), np.nan, np.float32)
    for i, r in enumerate(records):
        table[i, :counts[i]] = np.concatenate(r, axis=1)
    fx.array("steps_body_q", poses)
    fx.array("steps_contact_count", counts)
    # Per contact: shape0, shape1, point0, point1, offset0, offset1, normal, margin0, margin1.
    fx.array("steps_contacts", table)
    fx.save(fixtures_dir() / "body_collide.vbdf")
    pairs = {tuple(int(v) for v in row[:2]) for r in table for row in r if not np.isnan(row[0])}
    print(f"body_collide: {model.shape_count} shapes, {len(model.shape_contact_pairs.numpy())} pairs, "
          f"contacts per step {counts.min()}..{counts.max()}, pairs in contact {sorted(pairs)}")


if __name__ == "__main__":
    main()
