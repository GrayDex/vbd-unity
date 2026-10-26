"""Float32 noise floor of the body-body contact scenes: Newton's own CPU run against its CUDA run. Contacts switch
branches, so the numbers show how far rounding alone carries each scene over its horizon; the port's deviation from
Newton is judged against them.

Run in the Newton env: python noise_floor_body.py
"""

import numpy as np

import scene_body_contact as scene
from noise_floor import rotation_deg


def main():
    for name, boxes in (("body_contact", False), ("body_stack", True)):
        cpu = scene.run(name, boxes, device="cpu", save=False)[0]
        cuda = scene.run(name, boxes, device="cuda:0", save=False)[0]
        dp = np.linalg.norm(cpu[..., :3] - cuda[..., :3], axis=-1).max(axis=1)
        dr = rotation_deg(cpu[..., 3:], cuda[..., 3:]).max(axis=1)
        print(f"NOISE {name}: worst position {dp.max() * 1e3:.5f} mm at step {dp.argmax()}, first 10 steps "
              f"{dp[:10].max() * 1e3:.5f} mm, worst rotation {dr.max():.5f} deg at step {dr.argmax()}")


if __name__ == "__main__":
    main()
