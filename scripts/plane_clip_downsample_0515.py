#!/usr/bin/env python
"""Clip a point cloud by a plane defined by a 4x4 transform matrix.

Keeps points on the +Z side of the plane (i.e. the side the plane's
Z-axis points toward).  Visualises the result with Open3D.
"""

import numpy as np
import open3d as o3d
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# --- plane transforms (4x4) ---
T_plane_1 = np.array([
    [-0.135466,  0.402313,  0.905424, -0.276897],
    [ 0.947717,  0.319113, -0.000000, -0.418919],
    [-0.288932,  0.858086, -0.424508,  1.411458],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
])

T_plane_2 = np.array([
    [ 0.087565, -0.897552,  0.432125,  0.239173],
    [ 0.995275,  0.097098, -0.000000, -0.444773],
    [-0.041959,  0.430083,  0.901814,  1.411695],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
])

T_plane_3 = np.array([
    [ 0.998534,  0.053937,  0.004515,  0.029245],
    [-0.006152,  0.030228,  0.999524,  0.293189],
    [ 0.053775, -0.998087,  0.030516,  1.103597],
    [ 0.000000,  0.000000,  0.000000,  1.000000],
])

# (plane_transform, keep_positive_side)
planes = [
    (T_plane_1, True),
    (T_plane_2, False),
    (T_plane_3, False),
]

# --- process all point clouds under train/ ---
MAX_POINTS = 100_000
ply_paths = sorted(DATA_DIR.glob("train/*/pcd/cloud.ply"))
print(f"Found {len(ply_paths)} point clouds to process")

for ply_path in ply_paths:
    print(f"\nProcessing: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    print(f"  Original points: {len(points)}")

    # half-space clip
    mask = np.ones(len(points), dtype=bool)
    for i, (T, keep_positive) in enumerate(planes):
        origin = T[:3, 3]
        normal = T[:3, 2]
        signed_dist = (points - origin) @ normal
        plane_mask = signed_dist >= 0 if keep_positive else signed_dist < 0
        mask &= plane_mask

    pcd_clipped = pcd.select_by_index(np.where(mask)[0])
    print(f"  After clipping: {mask.sum()} points remain")

    # downsample
    n_clipped = len(pcd_clipped.points)
    if n_clipped > MAX_POINTS:
        ratio = MAX_POINTS / n_clipped
        pcd_clipped = pcd_clipped.random_down_sample(ratio)
        print(f"  Downsampled to {len(pcd_clipped.points)} points")

    # overwrite original file
    o3d.io.write_point_cloud(str(ply_path), pcd_clipped)
    print(f"  Saved: {ply_path}")

print("\nDone.")
