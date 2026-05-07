#!/usr/bin/env python
"""Interactive pose-capture loop.

Workflow:
    1. Operator jogs the robot to a fresh pose (manual teach pendant).
    2. ENTER in this terminal -> we read robot pose, trigger stereo, save the
       project folder under data/poses/<timestamp>/, and write robot_pose.json.
    3. The script refuses captures whose rotation is within
       capture.min_pose_diversity_deg of any already-stored pose, so the
       AX=XB problem stays well conditioned.
    4. Type 'q' + ENTER to quit.

The HIK save_frames function actually writes to AI_DIR (~/DCIM_AI) by upstream
convention; we then move/copy the project folder under our data/poses/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config
from eye2hand.geometry import fairino_pose_to_se3, relative_rotation_deg
from eye2hand.paths import prepare_paths


def _existing_poses(poses_dir: Path) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for pd in sorted(p for p in poses_dir.iterdir() if p.is_dir()):
        rp = pd / "robot_pose.json"
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text())
            pose = np.asarray(data["pose_mm_deg"], dtype=np.float64).reshape(6)
            out.append(fairino_pose_to_se3(pose))
        except Exception:  # noqa: BLE001 -- we just skip malformed
            continue
    return out


def _too_close(T_new: np.ndarray, existing: list[np.ndarray], min_deg: float) -> tuple[bool, float]:
    if not existing:
        return False, float("inf")
    deltas = [relative_rotation_deg(T_new, T) for T in existing]
    closest = float(min(deltas))
    return closest < min_deg, closest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--no-camera", action="store_true",
        help="skip stereo capture (just record robot poses; for dry-running the loop)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    prepare_paths(args.config)

    poses_dir = cfg.poses_dir
    poses_dir.mkdir(parents=True, exist_ok=True)
    existing = _existing_poses(poses_dir)
    logger.info("starting capture loop -- {} pose(s) already in {}", len(existing), poses_dir)
    logger.info("min rotation diversity: {:.1f} deg", cfg.capture.min_pose_diversity_deg)
    logger.info("target captures: {} (min_poses to solve: {})",
                cfg.capture.target_poses, cfg.capture.min_poses)

    # Robot
    from eye2hand.robot import RobotClient
    robot = RobotClient(cfg.robot.ip)

    # Camera (lazy, since it owns Qt + HIK SDK)
    cam = None
    if not args.no_camera:
        from eye2hand.camera import StereoCapture
        cam = StereoCapture()

    try:
        while True:
            n = len(existing)
            prompt = (
                f"\n[{n}/{cfg.capture.target_poses}] jog the arm to a fresh pose, "
                f"then ENTER to capture (or 'q' to quit): "
            )
            ans = input(prompt).strip().lower()
            if ans in {"q", "quit", "exit"}:
                break

            try:
                pose_raw = robot.read_pose_raw()
                T_new = fairino_pose_to_se3(pose_raw)
            except Exception as e:
                logger.error("could not read robot pose: {}", e)
                continue

            close, closest_deg = _too_close(T_new, existing, cfg.capture.min_pose_diversity_deg)
            if close:
                logger.warning(
                    "pose too close to an existing one ({:.1f} deg < {:.1f} deg minimum); skipping",
                    closest_deg, cfg.capture.min_pose_diversity_deg,
                )
                continue

            # Capture stereo first; only commit if it succeeded.
            if cam is not None:
                try:
                    project_folder, raw_left, raw_right = cam.capture_one(cfg.data_root)
                except Exception as e:
                    logger.exception("stereo capture failed: {}", e)
                    continue
                # Move the broker project folder under data/poses/<id>/ for consistency
                dest = poses_dir / project_folder.name
                if dest.exists():
                    logger.warning("destination {} already exists; appending '_dup'", dest)
                    dest = dest.with_name(dest.name + "_dup")
                shutil.move(str(project_folder), str(dest))
                pose_dir = dest
                logger.info("saved stereo pair under {}", pose_dir)
            else:
                # dry-run: just create a folder with the timestamp
                import time
                pose_dir = poses_dir / str(int(time.time() * 1e7))
                pose_dir.mkdir(parents=True, exist_ok=True)
                logger.info("[--no-camera] created empty pose folder {}", pose_dir)

            # Persist robot pose
            (pose_dir / "robot_pose.json").write_text(
                json.dumps({"pose_mm_deg": pose_raw.tolist()}, indent=2)
            )
            existing.append(T_new)
            logger.success("captured pose #{} under {}", len(existing), pose_dir.name)

            if len(existing) >= cfg.capture.target_poses:
                logger.success("reached target of {} captures", cfg.capture.target_poses)
                break
    finally:
        if cam is not None:
            cam.close()

    logger.info("done -- {} total pose(s) under {}", len(existing), poses_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
