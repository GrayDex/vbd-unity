"""ModelBuilder.add_soft_mesh from explicit vertices and tetrahedra with per-element materials, one inverted element.

The mesh is a jittered Kuhn grid with shuffled vertex and element order, so coloring tie-breaking is exercised.
Run in the Newton env: python scene_builder_mesh.py
"""

import numpy as np
import warp as wp

import newton
from fixture import Fixture, fixtures_dir
from scene_builder_grid import add_colorings

NX, NY, NZ = 5, 4, 3
SEED = 7


def kuhn_mesh(rng):
    grid = np.array([(x, y, z) for z in range(NZ + 1) for y in range(NY + 1) for x in range(NX + 1)], np.float64)
    index = lambda x, y, z: (NX + 1) * (NY + 1) * z + (NX + 1) * y + x
    verts = grid * np.array([0.07, 0.05, 0.06]) + rng.uniform(-0.01, 0.01, grid.shape)
    tets = []
    for z in range(NZ):
        for y in range(NY):
            for x in range(NX):
                c = [index(x + dx, y + dy, z + dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
                for a, b, d in ((1, 3, 7), (1, 5, 7), (2, 3, 7), (2, 6, 7), (4, 5, 7), (4, 6, 7)):
                    tets.append((c[0], c[a], c[b], c[d]))
    tets = np.array(tets)
    for t in range(len(tets)):
        p = verts[tets[t]]
        if np.linalg.det(np.stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]], axis=1)) < 0:
            tets[t, [1, 2]] = tets[t, [2, 1]]
    perm = rng.permutation(len(verts))
    verts_shuffled = np.empty_like(verts)
    verts_shuffled[perm] = verts
    tets = perm[tets]
    tets = tets[rng.permutation(len(tets))]
    tets[3, [1, 2]] = tets[3, [2, 1]]
    return verts_shuffled.astype(np.float32), tets.astype(np.int32)


def main():
    rng = np.random.default_rng(SEED)
    verts, tets = kuhn_mesh(rng)
    n_tets = len(tets)
    k_mu = rng.uniform(1.0e4, 9.0e4, n_tets).astype(np.float32)
    k_lambda = rng.uniform(1.0e4, 9.0e4, n_tets).astype(np.float32)
    k_lambda[5] = 0.0
    k_mu[5] = 0.0
    k_damp = rng.uniform(0.0, 50.0, n_tets).astype(np.float32)
    rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 0.3, -0.5)), -0.4)
    pos, scale, vel, density = (0.2, 0.3, 1.5), 1.25, (0.0, 0.5, 0.0), 650.0

    builder = newton.ModelBuilder()
    builder.add_soft_mesh(pos=wp.vec3(*pos), rot=rot, scale=scale, vel=wp.vec3(*vel), vertices=verts.tolist(),
                          indices=tets.flatten().tolist(), density=density, k_mu=k_mu, k_lambda=k_lambda,
                          k_damp=k_damp, particle_radius=0.015)
    fx = Fixture("builder_mesh")
    fx.array("in_vertices", verts)
    fx.array("in_indices", tets.flatten())
    fx.array("in_k_mu", k_mu)
    fx.array("in_k_lambda", k_lambda)
    fx.array("in_k_damp", k_damp)
    fx.array("in_pos", np.array(pos, np.float32))
    fx.array("in_rot", np.array(rot))
    fx.array("in_vel", np.array(vel, np.float32))
    fx.param("in_scale", scale)
    fx.param("in_density", density)
    add_colorings(fx, builder)
    builder.color()
    model = builder.finalize()
    for name in ("particle_q", "particle_qd", "particle_mass", "particle_inv_mass", "particle_radius",
                 "particle_flags", "tet_indices", "tet_poses", "tet_materials"):
        fx.array(name, getattr(model, name).numpy())
    fx.save(fixtures_dir() / "builder_mesh.vbdf")
    print(f"builder_mesh: {model.particle_count} particles, {n_tets} tets given, {model.tet_count} kept, "
          f"{len(builder.particle_color_groups)} colors")


if __name__ == "__main__":
    main()
