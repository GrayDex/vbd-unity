"""ModelBuilder.add_soft_grid with fixed sides and the graph colorings of a small and a large grid.

Run in the Newton env: python scene_builder_grid.py
"""

import numpy as np
import warp as wp

import newton
from newton._src.sim.graph_coloring import ColoringAlgorithm
from fixture import Fixture, group_colors, fixtures_dir

VARIANTS = (("mcs_bal", ColoringAlgorithm.MCS, True), ("mcs_raw", ColoringAlgorithm.MCS, False),
            ("greedy_bal", ColoringAlgorithm.GREEDY, True), ("greedy_raw", ColoringAlgorithm.GREEDY, False))


def add_colorings(fx, builder, prefix=""):
    for name, algorithm, balance in VARIANTS:
        builder.color(balance_colors=balance, coloring_algorithm=algorithm)
        fx.array(f"{prefix}colors_{name}", group_colors(builder.particle_color_groups, builder.particle_count))


def main():
    rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.0, 1.0, 1.0)), 0.7)
    builder = newton.ModelBuilder()
    builder.add_soft_grid(pos=wp.vec3(0.5, -1.0, 2.0), rot=rot, vel=wp.vec3(0.1, 0.0, -0.2),
                          dim_x=5, dim_y=3, dim_z=2, cell_x=0.1, cell_y=0.2, cell_z=0.15,
                          density=800.0, k_mu=7.0e4, k_lambda=2.0e4, k_damp=30.0,
                          fix_left=True, fix_bottom=True, particle_radius=0.02)
    fx = Fixture("builder_grid")
    fx.array("rot", np.array(rot))
    add_colorings(fx, builder)
    builder.color()
    model = builder.finalize()
    fx.array("gravity", model.gravity.numpy()[0])
    for name in ("particle_q", "particle_qd", "particle_mass", "particle_inv_mass", "particle_radius",
                 "particle_flags", "tet_indices", "tet_poses", "tet_materials"):
        fx.array(name, getattr(model, name).numpy())
    fx.param("soft_contact_ke", model.soft_contact_ke)
    fx.param("soft_contact_kd", model.soft_contact_kd)
    fx.param("soft_contact_mu", model.soft_contact_mu)
    fx.save(fixtures_dir() / "builder_grid.vbdf")
    print(f"builder_grid: {model.particle_count} particles, {model.tet_count} tets, "
          f"{len(builder.particle_color_groups)} colors")

    large = newton.ModelBuilder()
    large.add_soft_grid(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                        dim_x=30, dim_y=10, dim_z=10, cell_x=0.1, cell_y=0.1, cell_z=0.1,
                        density=1000.0, k_mu=1.0e5, k_lambda=1.0e5, k_damp=0.0)
    fx = Fixture("builder_large")
    add_colorings(fx, large)
    fx.save(fixtures_dir() / "builder_large.vbdf")
    print(f"builder_large: {large.particle_count} particles, {large.tet_count} tets")


if __name__ == "__main__":
    main()
