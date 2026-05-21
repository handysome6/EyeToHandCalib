#!/usr/bin/env python
"""Restructure train data into the sample_data format.

Reads point clouds from data/train/{timestamp}/pcd/cloud.ply, transforms them
to robot base coordinates using T_cam2base, and saves in the demo_N/step_0
folder structure alongside converted target poses.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
CALIB_FILE = ROOT / "data" / "calibration" / "T_cam_base.json"
SAMPLE_DIR = ROOT / "data" / "sample_data"
OUTPUT_DIR = ROOT / "data" / "restructured"

POSE_MAP = [
    "17788391776099070",
    "17788393961033500",
    "17788395034040088",
    "17788396568104902",
    "17788398896214800",
    "17788400648020852",
    "17788401782317460",
    "17788403245530870",
    "17788404507856440",
    "17788405717836070",
]


def load_T_cam2base() -> np.ndarray:
    """Load the camera-frame -> base-frame transform (named T_cam_base in the JSON)."""
    with open(CALIB_FILE) as f:
        data = json.load(f)
    return np.array(data["T_cam_base"])


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])
    pts_base = (T @ pts_h.T).T[:, :3]
    return pts_base


def euler_to_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    r = Rotation.from_euler("XYZ", [rx_deg, ry_deg, rz_deg], degrees=True)
    return r.as_quat()  # [qx, qy, qz, qw]


def process_scene(scene_idx: int, T_cam2base: np.ndarray):
    timestamp = POSE_MAP[scene_idx]
    json_file = TRAIN_DIR / f"{scene_idx + 1:02d}.json"
    ply_file = TRAIN_DIR / timestamp / "pcd" / "cloud.ply"

    demo_dir = OUTPUT_DIR / "data" / f"demo_{scene_idx}"
    sample_demo = SAMPLE_DIR / "data" / f"demo_{scene_idx}"
    step0_dir = demo_dir / "step_0"
    step1_dir = demo_dir / "step_1"

    # Create directories
    for step_dir in [step0_dir, step1_dir]:
        for subdir in ["grasp_pcd", "scene_pcd", "target_poses"]:
            (step_dir / subdir).mkdir(parents=True, exist_ok=True)

    # --- demo metadata ---
    (demo_dir / "metadata.yaml").write_text("__type__: DemoSequence\nname: ''\n")

    # --- step metadata ---
    (step0_dir / "metadata.yaml").write_text("__type__: TargetPoseDemo\nname: pick\n")
    (step1_dir / "metadata.yaml").write_text("__type__: TargetPoseDemo\nname: place\n")

    # --- grasp_pcd: copy from sample_data for both steps ---
    for step_name in ["step_0", "step_1"]:
        src = sample_demo / step_name / "grasp_pcd"
        dst = demo_dir / step_name / "grasp_pcd"
        for fname in ["points.pt", "colors.pt"]:
            shutil.copy2(src / fname, dst / fname)
        (dst / "metadata.yaml").write_text(
            "__type__: PointCloud\nname: ''\nunit_length: 1 [m]\n"
        )

    # --- scene_pcd: read ply, filter zeros, transform to base (same for both steps) ---
    pcd = o3d.io.read_point_cloud(str(ply_file))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    valid = ~np.all(points == 0, axis=1)
    points = points[valid]
    colors = colors[valid]

    points_base = transform_points(points, T_cam2base)
    points_pt = torch.from_numpy(points_base)
    colors_pt = torch.from_numpy(colors)

    for step_dir in [step0_dir, step1_dir]:
        torch.save(points_pt, step_dir / "scene_pcd" / "points.pt")
        torch.save(colors_pt, step_dir / "scene_pcd" / "colors.pt")
        (step_dir / "scene_pcd" / "metadata.yaml").write_text(
            "__type__: PointCloud\nname: ''\nunit_length: 1 [m]\n"
        )

    # --- target_poses (step_0): convert mm+euler to m+quaternion ---
    with open(json_file) as f:
        pose_data = json.load(f)

    n_poses = pose_data["total_poses"]
    poses_tensor = torch.zeros(n_poses, 7, dtype=torch.float32)

    for p in pose_data["poses"]:
        idx = p["index"]
        vals = p["pose_mm_deg"]
        x_m, y_m, z_m = vals[0] / 1000.0, vals[1] / 1000.0, vals[2] / 1000.0
        quat = euler_to_quat(vals[3], vals[4], vals[5])
        poses_tensor[idx, :3] = torch.tensor([x_m, y_m, z_m], dtype=torch.float32)
        poses_tensor[idx, 3:] = torch.tensor(quat, dtype=torch.float32)

    torch.save(poses_tensor, step0_dir / "target_poses" / "poses.pt")
    (step0_dir / "target_poses" / "metadata.yaml").write_text(
        "__type__: SE3\nname: ''\nunit_length: 1 [m]\n"
    )

    # --- target_poses (step_1): copy from sample_data ---
    src_step1_poses = sample_demo / "step_1" / "target_poses"
    shutil.copy2(src_step1_poses / "poses.pt", step1_dir / "target_poses" / "poses.pt")
    (step1_dir / "target_poses" / "metadata.yaml").write_text(
        "__type__: SE3\nname: ''\nunit_length: 1 [m]\n"
    )

    n_grasp_pts = torch.load(
        sample_demo / "step_0" / "grasp_pcd" / "points.pt", weights_only=True
    ).shape[0]
    print(
        f"  demo_{scene_idx}: {points_base.shape[0]} scene points, "
        f"{n_grasp_pts} grasp points, "
        f"{n_poses} poses (shape {list(poses_tensor.shape)})"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write top-level data.yaml
    entries = [
        f'- path: "data/demo_{i}"\n  type: "DemoSequence"' for i in range(10)
    ]
    (OUTPUT_DIR / "data.yaml").write_text("\n".join(entries) + "\n")

    T_cam2base = load_T_cam2base()
    print(f"Loaded T_cam2base from {CALIB_FILE}")
    print(f"Output: {OUTPUT_DIR}\n")

    for i in range(10):
        process_scene(i, T_cam2base)

    print("\nDone.")


if __name__ == "__main__":
    main()
