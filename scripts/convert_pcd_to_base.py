#!/usr/bin/env python
"""Convert a point cloud from camera coordinates to robot base coordinates."""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

CALIB_FILE = Path(__file__).resolve().parents[1] / "data" / "0521calibration" / "T_cam_base.json"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.ply> [output.ply]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_stem(input_path.stem + "_base")

    with open(CALIB_FILE) as f:
        data = json.load(f)
    # Per handeye.py convention: "T_cam_base" maps camera-frame → base-frame
    T_cam2base = np.array(data["T_cam_base"])

    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points from {input_path}")

    # Filter zero points
    valid = ~np.all(points == 0, axis=1)
    if not np.all(valid):
        print(f"Filtered {(~valid).sum()} zero points")
        pcd = pcd.select_by_index(np.where(valid)[0])
        points = np.asarray(pcd.points)

    # Transform: P_base = T_base_cam @ P_cam (homogeneous)
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])
    pts_base = (T_cam2base @ pts_h.T).T[:, :3]

    pcd.points = o3d.utility.Vector3dVector(pts_base)

    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"Saved transformed point cloud to {output_path}")


if __name__ == "__main__":
    main()
