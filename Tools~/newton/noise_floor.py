"""Float32 noise floor of the rigid scenes: Newton's own CPU run against its CUDA run, worst deviation per scene.

The package's GPU port differs from Newton's CUDA run by rounding alone, like these two; the numbers show
how far that rounding carries each scene over its horizon. Run in the Newton env: python noise_floor.py
"""

import importlib

import numpy as np

from rigid_scene import run

SCENES = ["revolute_pendulum", "ball_chain", "fixed_pair", "prismatic_limit", "d6_limits", "revolute_drive",
          "kinematic_child", "free_spin", "finger"]


def rotation_deg(a, b):
    """Angle of the relative rotation between quaternion arrays (x, y, z, w), in degrees."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    w = np.abs(np.sum(a * b, axis=-1))
    v = np.linalg.norm(b[..., 3:4] * a[..., :3] - a[..., 3:4] * b[..., :3] - np.cross(b[..., :3], a[..., :3]), axis=-1)
    return np.degrees(2.0 * np.arctan2(v, w))


def main():
    for name in SCENES:
        module = importlib.import_module(f"scene_{name}")
        kwargs = {k: getattr(module, k) for k in ("prepare", "before_step") if hasattr(module, k)}
        runs = [run(name, module.build, steps=120, device=d, save=False, **kwargs)[0] for d in ("cpu", "cuda:0")]
        dp = np.linalg.norm(runs[0][..., :3] - runs[1][..., :3], axis=-1).max(axis=1)
        dr = rotation_deg(runs[0][..., 3:], runs[1][..., 3:]).max(axis=1)
        print(f"NOISE {name}: worst position {dp.max() * 1e3:.5f} mm at step {dp.argmax()}, first 10 steps "
              f"{dp[:10].max() * 1e3:.5f} mm, {dr[:10].max():.5f} deg, "
              f"worst rotation {dr.max():.5f} deg at step {dr.argmax()}")


if __name__ == "__main__":
    main()
