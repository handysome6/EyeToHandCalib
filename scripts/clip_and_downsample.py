#!/usr/bin/env python
"""Clip point clouds by half-space planes and downsample.

Pipeline order: clip → downsample → transform (optional) → save.
Clipping planes are defined in camera frame. The optional T_cam2base
transform is applied after clipping and downsampling.

Usage:
    # Camera-frame clipping (no transform):
    python scripts/clip_and_downsample.py --data-dir data/train

    # Transform to base frame first, then clip:
    python scripts/clip_and_downsample.py --data-dir data/train \
        --calib data/0521calib_res/T_cam2base.json

    # Custom max points:
    python scripts/clip_and_downsample.py --data-dir data/train --max-points 50000

Plane matrices and keep-side flags are defined in PLANES below.
Edit them to match your workspace geometry.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Clipping planes — edit these to match your workspace.
# Each entry: (4x4 transform, keep_positive_z_side)
#   The plane passes through T[:3, 3] with normal T[:3, 2].
#   keep=True  → keep points where (p - origin) · normal >= 0
#   keep=False → keep points where (p - origin) · normal <  0
# ---------------------------------------------------------------------------
PLANES = [
    (np.array([
        [-0.135466,  0.402313,  0.905424, -0.276897],
        [ 0.947717,  0.319113, -0.000000, -0.418919],
        [-0.288932,  0.858086, -0.424508,  1.411458],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ]), True),
    (np.array([
        [ 0.087565, -0.897552,  0.432125,  0.239173],
        [ 0.995275,  0.097098, -0.000000, -0.444773],
        [-0.041959,  0.430083,  0.901814,  1.411695],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ]), False),
    (np.array([
        [ 0.998534,  0.053937,  0.004515,  0.029245],
        [-0.006152,  0.030228,  0.999524,  0.293189],
        [ 0.053775, -0.998087,  0.030516,  1.103597],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ]), False),
    (np.array([
        [ 0.342020,  0.000000, -0.939693,  0.320387],
        [ 0.000000,  1.000000,  0.000000,  0.000038],
        [ 0.939693,  0.000000,  0.342020,  0.835964],
        [ 0.000000,  0.000000,  0.000000,  1.000000],
    ]), True),
]


def load_T_cam2base(calib_path: Path) -> np.ndarray:
    with open(calib_path) as f:
        data = json.load(f)
    return np.array(data["T_cam2base"])


def transform_points_inplace(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> None:
    points = np.asarray(pcd.points)
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])
    pts_out = (T @ pts_h.T).T[:, :3]
    pcd.points = o3d.utility.Vector3dVector(pts_out)


def clip_by_planes(pcd: o3d.geometry.PointCloud, planes: list) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    mask = np.ones(len(points), dtype=bool)
    for T, keep_positive in planes:
        origin = T[:3, 3]
        normal = T[:3, 2]
        signed_dist = (points - origin) @ normal
        plane_mask = signed_dist >= 0 if keep_positive else signed_dist < 0
        mask &= plane_mask
    return pcd.select_by_index(np.where(mask)[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Clip and downsample point clouds")
    ap.add_argument("--data-dir", required=True, help="directory containing */pcd/cloud.ply")
    ap.add_argument("--calib", default=None,
                    help="path to T_cam2base.json calibration file; if omitted, no transform is applied")
    ap.add_argument("--voxel-size", type=float, default=0.0015, help="voxel size for downsampling in meters (default: 0.0015)")
    ap.add_argument("--glob", default="*/pcd/cloud.ply", help="glob pattern for PLY files (default: */pcd/cloud.ply)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    ply_paths = sorted(data_dir.glob(args.glob))
    if not ply_paths:
        print(f"No PLY files found matching {data_dir / args.glob}")
        return 1

    T_cam2base = None
    if args.calib:
        calib_path = Path(args.calib).expanduser().resolve()
        T_cam2base = load_T_cam2base(calib_path)
        print(f"Loaded T_cam2base from {calib_path}")
    else:
        print("No calibration file provided — clipping in camera frame")

    print(f"Found {len(ply_paths)} point clouds, voxel_size={args.voxel_size}\n")

    for ply_path in ply_paths:
        print(f"Processing: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        n_orig = len(pcd.points)
        print(f"  Original: {n_orig} points")

        # 1. Clip
        pcd = clip_by_planes(pcd, PLANES)
        n_clipped = len(pcd.points)
        print(f"  After clipping: {n_clipped} points")

        # 2. Voxel downsample
        pcd = pcd.voxel_down_sample(args.voxel_size)
        print(f"  After voxel downsample ({args.voxel_size}m): {len(pcd.points)} points")

        # 3. Transform (optional)
        if T_cam2base is not None:
            transform_points_inplace(pcd, T_cam2base)
            print("  Applied T_cam2base transform")

        # 4. Save
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"  Saved: {ply_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
