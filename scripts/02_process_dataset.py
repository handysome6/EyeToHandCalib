#!/usr/bin/env python
"""Run rectify + pcd + target-pose for every captured pose folder.

For each data/poses/<id>/ that has raw_left.jpg, raw_right.jpg,
camera_model.json:

    rect/                       <- broker rectification output
    pcd/                        <- broker pcd output (depth_meter.npy, K.txt, ...)
    target_pose.json            <- T_target_cam from depth-lifted ChArUco solve

Re-runs are idempotent: skip a pose if target_pose.json already exists, unless
--force is given.

Broker pcd server config: pass --pcd-config <yaml> or set env vars
PCD_PRIMARY_SERVER_URL / PCD_FALLBACK_SERVER_URL (see broker/pcd_generate/config.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config
from eye2hand.paths import prepare_paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to calib.yaml (default: configs/calib.yaml)")
    ap.add_argument("--pcd-config", default=None, help="path to a broker pcd-config yaml (primary/fallback server URLs)")
    ap.add_argument("--only", default=None, help="process only this pose id (folder name under data/poses)")
    ap.add_argument("--force", action="store_true", help="re-run pose even if target_pose.json already exists")
    ap.add_argument("--no-debug-vis", action="store_true", help="skip writing target_pose_vis.png debug overlay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prepare_paths(args.config)  # injects JetsonReborn_rebar onto sys.path

    # Imports that need the broker on sys.path go here:
    from eye2hand.pipeline import load_pcd_outputs, run_pcd, run_rectification  # noqa: E402
    from eye2hand.target_pose import compute_target_pose, save_target_pose  # noqa: E402

    poses_dir = cfg.poses_dir
    if not poses_dir.exists():
        logger.error("no captured poses at {}", poses_dir)
        return 1

    pose_dirs = sorted(p for p in poses_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if args.only:
        pose_dirs = [p for p in pose_dirs if p.name == args.only]
        if not pose_dirs:
            logger.error("no pose folder named {} under {}", args.only, poses_dir)
            return 1

    n_ok, n_fail, n_skip = 0, 0, 0
    for pd in pose_dirs:
        raw_left = pd / "raw_left.jpg"
        raw_right = pd / "raw_right.jpg"
        cam_model = pd / "camera_model.json"
        if not (raw_left.exists() and raw_right.exists() and cam_model.exists()):
            logger.warning("[{}] missing raw_left/raw_right/camera_model.json -- skipping", pd.name)
            continue

        target_path = pd / "target_pose.json"
        if target_path.exists() and not args.force:
            logger.info("[{}] target_pose.json exists, skipping (use --force to re-run)", pd.name)
            n_skip += 1
            continue

        rect_dir = pd / "rect"
        pcd_dir = pd / "pcd"

        try:
            run_rectification(raw_left, raw_right, cam_model, rect_dir)
            rect_left = rect_dir / "rect_left.jpg"
            rect_right = rect_dir / "rect_right.jpg"
            if not (rect_left.exists() and rect_right.exists()):
                # legacy A_*.jpg / D_*.jpg fallback
                a_cands = sorted(rect_dir.glob("A_*.*"))
                d_cands = sorted(rect_dir.glob("D_*.*"))
                if a_cands and d_cands:
                    rect_left = a_cands[0]
                    rect_right = d_cands[0]
            run_pcd(rect_left, rect_right, cam_model, pcd_dir, pcd_config=args.pcd_config)

            pcd_outputs = load_pcd_outputs(pcd_dir, rect_dir=rect_dir)
            debug_dir = None if args.no_debug_vis else pd
            result = compute_target_pose(pcd_outputs, cfg.board, cfg.target_pose, debug_dir=debug_dir)
            save_target_pose(result, target_path)
            logger.success(
                "[{}] {} | corners {}/{} | reproj_rmse={:.2f}px | rmse_3d={:.4f}m",
                pd.name, result.method, result.n_corners_used, result.n_corners_detected,
                result.reprojection_rmse_px, result.rmse_m,
            )
            n_ok += 1
        except Exception as e:
            logger.exception("[{}] failed: {}", pd.name, e)
            n_fail += 1

    logger.info("done: ok={}, fail={}, skipped={}", n_ok, n_fail, n_skip)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
