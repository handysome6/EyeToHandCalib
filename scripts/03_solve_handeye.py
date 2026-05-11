#!/usr/bin/env python
"""Solve eye-to-hand AX=XB given a populated data/poses/<id>/ tree.

Each pose folder must contain:
    robot_pose.json   -- Fairino TCP pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    target_pose.json  -- written by 02_process_dataset.py (T_target_cam = target->cam)

Outputs to data/handeye/T_cam_base.json plus a per-pair residual report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config
from eye2hand.geometry import (
    fairino_pose_to_se3,
    se3_inv,
)
from eye2hand.handeye import save_solution, solve_all


def _load_pose_pair(pose_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    rp = pose_dir / "robot_pose.json"
    tp = pose_dir / "target_pose.json"
    if not (rp.exists() and tp.exists()):
        logger.warning("skipping {} (missing robot_pose.json or target_pose.json)", pose_dir.name)
        return None
    robot = json.loads(rp.read_text())
    target = json.loads(tp.read_text())
    pose_vec = np.array(robot["pose_mm_deg"], dtype=np.float64)
    base_T_gripper = fairino_pose_to_se3(pose_vec)
    T_target_cam = np.array(target["T_target_cam"], dtype=np.float64)
    # cv2.calibrateHandEye wants R/t_target2cam, i.e. transform that takes
    # target-frame points to cam frame. T_target_cam (named in target_pose.py
    # as "board frame -> camera frame") is exactly that, so we pass it through.
    cam_T_target = T_target_cam
    return base_T_gripper, cam_T_target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to calib.yaml (default: configs/calib.yaml)")
    ap.add_argument("--poses-dir", default=None, help="captured sample folder (default: config poses_dir)")
    ap.add_argument("--out", default=None, help="output dir (default: data/handeye)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    poses_dir = Path(args.poses_dir).expanduser().resolve() if args.poses_dir else cfg.poses_dir
    if not poses_dir.exists():
        logger.error("no captured poses at {}", poses_dir)
        return 1

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "handeye"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather pairs in deterministic order
    pose_dirs = sorted(p for p in poses_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    base_T_gripper: list[np.ndarray] = []
    cam_T_target: list[np.ndarray] = []
    used_ids: list[str] = []
    for pd in pose_dirs:
        pair = _load_pose_pair(pd)
        if pair is None:
            continue
        base_T_gripper.append(pair[0])
        cam_T_target.append(pair[1])
        used_ids.append(pd.name)

    if len(base_T_gripper) < cfg.capture.min_poses:
        logger.warning(
            "only {} valid pose pairs (< capture.min_poses={}); results may be unstable",
            len(base_T_gripper), cfg.capture.min_poses,
        )
    if len(base_T_gripper) < 3:
        logger.error("need at least 3 pose pairs to solve; have {}", len(base_T_gripper))
        return 1

    logger.info("solving with {} pose pairs", len(base_T_gripper))
    sols = solve_all(base_T_gripper, cam_T_target, methods=cfg.handeye.methods)
    if not sols:
        logger.error("all hand-eye methods failed")
        return 1

    # Pick the lowest-residual method as the canonical answer
    best = min(sols, key=lambda s: s.rmse_translation_m)
    logger.success("best method: {} (T_rmse={:.4f} m, R_rmse={:.4f} deg)",
                   best.method, best.rmse_translation_m, best.rmse_rotation_deg)

    save_solution(best, out_dir / cfg.handeye.output_filename)
    # also dump per-method comparison and per-pair residuals
    summary = {
        "n_pairs": int(len(base_T_gripper)),
        "pose_ids": used_ids,
        "methods": [
            {
                "method": s.method,
                "T_cam_base": s.T_cam_base.tolist(),
                "rmse_translation_m": float(s.rmse_translation_m),
                "rmse_rotation_deg": float(s.rmse_rotation_deg),
            }
            for s in sols
        ],
        "best_method": best.method,
    }
    (out_dir / "handeye_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote {} and handeye_summary.json", cfg.handeye.output_filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
