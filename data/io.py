"""DTU file readers."""

from __future__ import annotations

import re

import numpy as np


def read_pfm(filename: str) -> np.ndarray:
    """Read a PFM file and return a contiguous numpy array."""
    with open(filename, "rb") as file:
        header = file.readline().decode("utf-8").rstrip()
        if header == "PF":
            color = True
        elif header == "Pf":
            color = False
        else:
            raise ValueError("Not a PFM file.")

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(file.readline().decode("utf-8").rstrip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        return np.flipud(data).copy()


def read_camera_file(filename: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Read a DTU camera file.

    Returns:
        intrinsics, extrinsics, depth_min, depth_interval
    """
    with open(filename) as file:
        lines = [line.rstrip() for line in file.readlines()]

    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape((4, 4))
    intrinsics = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ").reshape((3, 3))
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_interval
