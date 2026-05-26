#!/usr/bin/env python
"""Convert a point cloud from camera coordinates to robot base coordinates.

Outputs both .ply and .pt (points + colors tensor pair).
"""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <calib_folder> <input.ply> [output_prefix]")
        print(f"\nAvailable calib folders in {DATA_DIR}:")
        for d in sorted(DATA_DIR.glob("*calib_res")):
            print(f"  {d.name}")
        sys.exit(1)

    calib_folder = sys.argv[1]
    input_path = Path(sys.argv[2])
    if len(sys.argv) >= 4:
        out_prefix = Path(sys.argv[3])
    else:
        out_prefix = input_path.with_stem(input_path.stem + "_base").with_suffix("")

    calib_file = DATA_DIR / calib_folder / "T_cam2base.json"
    print(f"Using calibration: {calib_file}")

    with open(calib_file) as f:
        data = json.load(f)
    T_cam2base = np.array(data["T_cam2base"])

    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"Loaded {len(points)} points from {input_path}")

    # Filter zero points
    valid = ~np.all(points == 0, axis=1)
    if not np.all(valid):
        print(f"Filtered {(~valid).sum()} zero points")
        points = points[valid]
        colors = colors[valid]

    # Transform: P_base = T_cam2base @ P_cam (homogeneous)
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])
    pts_base = (T_cam2base @ pts_h.T).T[:, :3]

    # Save .ply
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts_base)
    pcd_out.colors = o3d.utility.Vector3dVector(colors)
    out_ply = out_prefix.with_suffix(".ply")
    o3d.io.write_point_cloud(str(out_ply), pcd_out)
    print(f"Saved {out_ply}")

    # Save .pt pair
    out_points = out_prefix.with_name(out_prefix.name + "_points.pt")
    out_colors = out_prefix.with_name(out_prefix.name + "_colors.pt")
    torch.save(torch.from_numpy(pts_base), out_points)
    torch.save(torch.from_numpy(colors), out_colors)
    print(f"Saved {out_points} and {out_colors}")


if __name__ == "__main__":
    main()
