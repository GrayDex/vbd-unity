"""Reference fixtures from Newton for the package tests.

File layout (little endian): b"VBDF", uint32 header length, UTF-8 JSON header padded to 4 bytes, raw array data.
Header: {"scene": str, "params": [{"name", "value"}], "arrays": [{"name", "dtype", "shape", "offset"}]};
dtype is "f4" or "i4", offset counts from the start of the data block. Read by Tests/Editor/Fixture.cs.
"""

import json
import struct
from pathlib import Path

import numpy as np


class Fixture:
    def __init__(self, scene: str):
        self.scene = scene
        self.params = []
        self.arrays = []

    def param(self, name: str, value: float):
        self.params.append({"name": name, "value": float(value)})

    def array(self, name: str, values):
        a = np.ascontiguousarray(values)
        if a.dtype.kind == "f":
            a = a.astype("<f4")
        elif a.dtype.kind in "iub":
            a = a.astype("<i4")
        else:
            raise TypeError(f"{name}: unsupported dtype {a.dtype}")
        self.arrays.append((name, a))

    def save(self, path: Path):
        entries, offset = [], 0
        for name, a in self.arrays:
            entries.append({"name": name, "dtype": a.dtype.str[1:], "shape": list(a.shape), "offset": offset})
            offset += a.nbytes
        header = json.dumps({"scene": self.scene, "params": self.params, "arrays": entries}).encode("utf-8")
        header += b" " * (-len(header) % 4)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"VBDF")
            f.write(struct.pack("<I", len(header)))
            f.write(header)
            for _, a in self.arrays:
                f.write(a.tobytes())


def color_groups(model):
    """Newton's particle color groups flattened: particle ids by color, and color start offsets."""
    groups = [g.numpy() for g in model.particle_color_groups]
    offsets = np.cumsum([0] + [len(g) for g in groups])
    return np.concatenate(groups), offsets


def fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "Tests" / "Fixtures~"


def group_colors(groups, count):
    """Per-node color from Newton's color groups (a list of index arrays)."""
    colors = np.full(count, -1, np.int32)
    for color, group in enumerate(groups):
        colors[np.asarray(group, dtype=np.int64)] = color
    return colors
