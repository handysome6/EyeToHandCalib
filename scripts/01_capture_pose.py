#!/usr/bin/env python
"""Calibration data collection loop.

Workflow:
    1. Operator jogs the robot to a fresh pose (manual teach pendant).
    2. ENTER in this terminal -> we read robot pose, trigger stereo, save the
       project folder under the target folder, and write robot_pose.json.
    3. The script refuses captures whose rotation is within
       capture.min_pose_diversity_deg of any already-stored pose, so the
       AX=XB problem stays well conditioned.
    4. Type 'q' + ENTER to quit.

Live mode uses the Fairino SDK and HIK `HikSyncedCameras`. Hardware-free modes
(`--dry-run`, `--mock-robot`, `--mock-camera`, `--manual-robot`) exercise the
same folder/manifest logic without touching devices.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import DEFAULT_CONFIG_PATH, load_config
from eye2hand.geometry import fairino_pose_to_se3, relative_rotation_deg


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _existing_poses(poses_dir: Path) -> tuple[list[np.ndarray], set[str]]:
    out: list[np.ndarray] = []
    ids: set[str] = set()
    if not poses_dir.exists():
        return out, ids
    for pd in sorted(p for p in poses_dir.iterdir() if p.is_dir()):
        rp = pd / "robot_pose.json"
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text())
            pose = np.asarray(data["pose_mm_deg"], dtype=np.float64).reshape(6)
            out.append(fairino_pose_to_se3(pose))
            ids.add(pd.name)
        except Exception:  # noqa: BLE001 -- we just skip malformed
            continue
    return out, ids


def _too_close(T_new: np.ndarray, existing: list[np.ndarray], min_deg: float) -> tuple[bool, float]:
    if not existing:
        return False, float("inf")
    deltas = [relative_rotation_deg(T_new, T) for T in existing]
    closest = float(min(deltas))
    return closest < min_deg, closest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_pose_data(path: Path, index: int, sample_id: str, pose_raw: np.ndarray) -> None:
    pose = ", ".join(f"{v:.6f}" for v in pose_raw.tolist())
    line = f"pose {index} {sample_id} [{pose}]\n"
    with path.open("a") as f:
        f.write(line)


def _ensure_dataset_info(
    out_dir: Path,
    *,
    cfg,
    config_path: str | None,
    camera_mode: str,
    robot_mode: str,
    note: str | None,
) -> None:
    info_path = out_dir / "dataset_info.json"
    payload = {
        "schema_version": 1,
        "tool": "scripts/01_capture_pose.py",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "config_path": str(_resolve_path(config_path or DEFAULT_CONFIG_PATH)),
        "target_folder": str(out_dir),
        "jetson_reborn_path": str(cfg.jetson_reborn_path),
        "data_root": str(cfg.data_root),
        "robot": {
            "mode": robot_mode,
            "ip": cfg.robot.ip if robot_mode == "live" else None,
        },
        "camera": {
            "mode": camera_mode,
            "sdk": "JetsonReborn_rebar/hik.HikSyncedCameras" if camera_mode == "live" else None,
        },
        "sample_layout": {
            "sample_dir": "<target_folder>/<sample_id>/",
            "robot_pose": "robot_pose.json",
            "left_image": "raw_left.jpg",
            "right_image": "raw_right.jpg",
            "camera_model": "camera_model.json",
        },
        "note": note,
    }
    if info_path.exists():
        try:
            existing = json.loads(info_path.read_text())
            existing["updated_at"] = payload["updated_at"]
            existing["last_modes"] = {"camera": camera_mode, "robot": robot_mode}
            if note:
                existing["note"] = note
            _write_json(info_path, existing)
            return
        except Exception:  # noqa: BLE001 -- repair malformed metadata
            pass
    _write_json(info_path, payload)


def _sample_files(pose_dir: Path) -> dict[str, str | None]:
    names = {
        "raw_left": "raw_left.jpg",
        "raw_right": "raw_right.jpg",
        "camera_model": "camera_model.json",
    }
    return {
        key: str(pose_dir / name) if (pose_dir / name).exists() else None
        for key, name in names.items()
    }


def _manual_pose() -> np.ndarray:
    while True:
        raw = input("paste TCP pose [x_mm,y_mm,z_mm,rx_deg,ry_deg,rz_deg]: ").strip()
        vals = [v for v in raw.replace("[", "").replace("]", "").replace(",", " ").split() if v]
        try:
            pose = np.asarray([float(v) for v in vals], dtype=np.float64)
        except ValueError:
            logger.error("could not parse pose values")
            continue
        if pose.size != 6:
            logger.error("expected 6 values, got {}", pose.size)
            continue
        return pose


class ManualRobotClient:
    def read_pose_raw(self) -> np.ndarray:
        return _manual_pose()

    def close(self) -> None:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="target folder for captured samples (default: config poses_dir)")
    ap.add_argument("--count", type=int, default=None, help="number of new accepted samples to capture before exit")
    ap.add_argument("--auto", action="store_true", help="capture without waiting for ENTER; intended for dry-run/testing")
    ap.add_argument("--auto-delay-s", type=float, default=0.0, help="delay between --auto captures")
    ap.add_argument("--timeout-s", type=float, default=10.0, help="camera capture timeout in seconds")
    ap.add_argument("--min-rotation-deg", type=float, default=None, help="override config capture diversity threshold")
    ap.add_argument("--allow-low-diversity", action="store_true", help="warn but keep poses below the diversity threshold")
    ap.add_argument("--note", default=None, help="optional note stored in dataset_info.json")
    ap.add_argument(
        "--robot-mode",
        choices=["live", "mock", "manual"],
        default="live",
        help="pose source: live Fairino, deterministic mock, or manually pasted TCP vectors",
    )
    ap.add_argument(
        "--camera-mode",
        choices=["live", "mock", "none"],
        default="live",
        help="image source: live HIK stereo, deterministic mock images, or pose-only folders",
    )
    ap.add_argument(
        "--no-camera", action="store_true",
        help="alias for --camera-mode none",
    )
    ap.add_argument("--mock-robot", action="store_true", help="alias for --robot-mode mock")
    ap.add_argument("--manual-robot", action="store_true", help="alias for --robot-mode manual")
    ap.add_argument("--mock-camera", action="store_true", help="alias for --camera-mode mock")
    ap.add_argument("--dry-run", action="store_true", help="alias for --robot-mode mock --camera-mode mock")
    args = ap.parse_args()

    cfg = load_config(args.config)
    camera_mode = args.camera_mode
    robot_mode = args.robot_mode
    if args.no_camera:
        camera_mode = "none"
    if args.mock_camera:
        camera_mode = "mock"
    if args.mock_robot:
        robot_mode = "mock"
    if args.manual_robot:
        robot_mode = "manual"
    if args.dry_run:
        camera_mode = "mock"
        robot_mode = "mock"

    poses_dir = _resolve_path(args.out) if args.out else cfg.poses_dir
    poses_dir.mkdir(parents=True, exist_ok=True)
    existing, existing_ids = _existing_poses(poses_dir)
    min_rotation_deg = (
        float(args.min_rotation_deg)
        if args.min_rotation_deg is not None
        else float(cfg.capture.min_pose_diversity_deg)
    )
    target_new = args.count
    target_total = cfg.capture.target_poses if target_new is None else len(existing) + target_new

    _ensure_dataset_info(
        poses_dir,
        cfg=cfg,
        config_path=args.config,
        camera_mode=camera_mode,
        robot_mode=robot_mode,
        note=args.note,
    )

    logger.info("starting capture loop -- {} pose(s) already in {}", len(existing), poses_dir)
    logger.info("modes: robot={}, camera={}", robot_mode, camera_mode)
    logger.info("min rotation diversity: {:.1f} deg", min_rotation_deg)
    logger.info("target total: {} (min_poses to solve: {})", target_total, cfg.capture.min_poses)

    # Robot
    if robot_mode == "live":
        from eye2hand.robot import RobotClient
        robot = RobotClient(cfg.robot.ip)
    elif robot_mode == "mock":
        from eye2hand.robot import MockRobotClient
        robot = MockRobotClient(start_index=len(existing))
    else:
        robot = ManualRobotClient()

    # Camera (lazy, since it owns Qt + HIK SDK)
    cam = None
    if camera_mode == "live":
        from eye2hand.camera import StereoCapture
        cam = StereoCapture(cfg.jetson_reborn_path)
    elif camera_mode == "mock":
        from eye2hand.camera import MockStereoCapture
        cam = MockStereoCapture()

    n_new = 0

    try:
        while True:
            n = len(existing)
            if n >= target_total:
                logger.success("reached target of {} total captures", target_total)
                break
            if target_new is not None and n_new >= target_new:
                logger.success("captured requested {} new sample(s)", target_new)
                break

            if args.auto:
                if args.auto_delay_s > 0:
                    time.sleep(args.auto_delay_s)
                ans = ""
            else:
                prompt = (
                    f"\n[{n}/{target_total}] jog the arm to a fresh pose, "
                    f"then ENTER to capture (or 'q' to quit): "
                )
                ans = input(prompt).strip().lower()
                if ans in {"q", "quit", "exit"}:
                    break

            try:
                robot_read_start = time.time_ns()
                pose_raw = robot.read_pose_raw()
                robot_read_end = time.time_ns()
                T_new = fairino_pose_to_se3(pose_raw)
            except Exception as e:
                logger.error("could not read robot pose: {}", e)
                continue

            close, closest_deg = _too_close(T_new, existing, min_rotation_deg)
            if close and not args.allow_low_diversity:
                logger.warning(
                    "pose too close to an existing one ({:.1f} deg < {:.1f} deg minimum); skipping",
                    closest_deg, min_rotation_deg,
                )
                if args.auto:
                    logger.error("auto capture stopped after a low-diversity pose; move the robot or use --allow-low-diversity")
                    break
                continue
            if close:
                logger.warning(
                    "pose diversity is low ({:.1f} deg < {:.1f} deg), keeping because --allow-low-diversity is set",
                    closest_deg,
                    min_rotation_deg,
                )

            captured_at = _now_iso()
            capture_start = time.time_ns()
            if cam is not None:
                try:
                    pose_dir, raw_left, raw_right = cam.capture_one(poses_dir, timeout_s=args.timeout_s)
                except Exception as e:
                    logger.exception("stereo capture failed: {}", e)
                    continue
                logger.info("saved stereo pair under {}", pose_dir)
            else:
                # pose-only mode: create a folder with the same timestamp scheme.
                pose_dir = poses_dir / str(int(time.time() * 1e7))
                while pose_dir.name in existing_ids or pose_dir.exists():
                    time.sleep(0.001)
                    pose_dir = poses_dir / str(int(time.time() * 1e7))
                pose_dir.mkdir(parents=True, exist_ok=False)
                logger.info("[camera-mode=none] created pose-only folder {}", pose_dir)
            capture_end = time.time_ns()
            existing_ids.add(pose_dir.name)

            # Persist robot pose
            sample_index = len(existing) + 1
            robot_payload = {
                "schema_version": 1,
                "sample_id": pose_dir.name,
                "pose_index": sample_index,
                "captured_at": captured_at,
                "pose_mm_deg": pose_raw.tolist(),
                "T_base_tcp": T_new.tolist(),
                "units": {
                    "pose_translation": "mm",
                    "pose_rotation": "deg",
                    "T_base_tcp_translation": "m",
                },
                "robot": {
                    "mode": robot_mode,
                    "ip": cfg.robot.ip if robot_mode == "live" else None,
                    "read_start_time_ns": robot_read_start,
                    "read_end_time_ns": robot_read_end,
                },
            }
            _write_json(pose_dir / "robot_pose.json", robot_payload)

            manifest_payload = {
                "schema_version": 1,
                "sample_id": pose_dir.name,
                "sample_index": sample_index,
                "sample_dir": str(pose_dir),
                "captured_at": captured_at,
                "pose_mm_deg": pose_raw.tolist(),
                "T_base_tcp": T_new.tolist(),
                "closest_existing_rotation_deg": None if np.isinf(closest_deg) else closest_deg,
                "camera_mode": camera_mode,
                "robot_mode": robot_mode,
                "capture_start_time_ns": capture_start,
                "capture_end_time_ns": capture_end,
                "files": _sample_files(pose_dir),
            }
            _append_jsonl(poses_dir / "manifest.jsonl", manifest_payload)
            _append_pose_data(poses_dir / "pose_data.md", sample_index, pose_dir.name, pose_raw)

            existing.append(T_new)
            n_new += 1
            logger.success("captured pose #{} under {}", len(existing), pose_dir.name)
    finally:
        if cam is not None:
            cam.close()
        close_robot = getattr(robot, "close", None)
        if callable(close_robot):
            close_robot()

    logger.info("done -- {} total pose(s), {} new, under {}", len(existing), n_new, poses_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
