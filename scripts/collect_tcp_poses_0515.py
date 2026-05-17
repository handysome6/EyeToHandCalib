#!/usr/bin/env python
"""Collect multiple TCP poses from the robotic arm into a single JSON file.

Usage:
    python scripts/collect_tcp_poses.py
    python scripts/collect_tcp_poses.py --output poses.json
    python scripts/collect_tcp_poses.py --mock       # hardware-free test
    python scripts/collect_tcp_poses.py --manual     # paste poses manually
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _manual_pose() -> np.ndarray:
    while True:
        raw = input("  paste TCP pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]: ").strip()
        vals = [v for v in raw.replace("[", "").replace("]", "").replace(",", " ").split() if v]
        try:
            pose = np.asarray([float(v) for v in vals], dtype=np.float64)
        except ValueError:
            print("  error: could not parse values, try again")
            continue
        if pose.size != 6:
            print(f"  error: expected 6 values, got {pose.size}")
            continue
        return pose


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect TCP poses into a single JSON file")
    ap.add_argument("--config", default=None, help="path to calib.yaml")
    ap.add_argument("--output", "-o", default=None, help="output JSON path (default: data/calibration/tcp_poses.json)")
    ap.add_argument("--mock", action="store_true", help="use mock robot (no hardware)")
    ap.add_argument("--manual", action="store_true", help="paste poses manually instead of reading from robot")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = REPO_ROOT / "data" / "calibration" / "tcp_poses.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing poses if the file already exists
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        poses = existing.get("poses", [])
        print(f"loaded {len(poses)} existing pose(s) from {out_path}")
    else:
        poses = []

    # Connect to robot
    if args.manual:
        robot = None
        mode = "manual"
        print("manual mode: you will paste each pose")
    elif args.mock:
        from eye2hand.robot import MockRobotClient
        robot = MockRobotClient(start_index=len(poses))
        mode = "mock"
        print("mock mode: using simulated poses")
    else:
        from eye2hand.robot import RobotClient
        robot = RobotClient(cfg.robot.ip)
        mode = "live"
        print(f"connected to robot at {cfg.robot.ip}")

    print(f"output: {out_path}")
    print("press ENTER to capture a pose, 'q' to quit and save\n")

    try:
        while True:
            cmd = input(f"[{len(poses)} captured] ENTER=capture, q=quit: ").strip().lower()
            if cmd in {"q", "quit", "exit"}:
                break

            try:
                if args.manual:
                    pose_raw = _manual_pose()
                else:
                    pose_raw = robot.read_pose_raw()
            except Exception as e:
                print(f"  error reading pose: {e}")
                continue

            pose_list = pose_raw.tolist()
            entry = {
                "index": len(poses),
                "pose_mm_deg": pose_list,
                "captured_at": _now_iso(),
            }
            poses.append(entry)

            x, y, z, rx, ry, rz = pose_list
            print(f"  #{len(poses)}: [{x:.2f}, {y:.2f}, {z:.2f}, {rx:.2f}, {ry:.2f}, {rz:.2f}]")

    except (KeyboardInterrupt, EOFError):
        print()

    finally:
        if robot is not None:
            robot.close()

    if not poses:
        print("no poses collected, nothing saved")
        return 0

    payload = {
        "schema_version": 1,
        "robot_ip": cfg.robot.ip,
        "mode": mode,
        "total_poses": len(poses),
        "updated_at": _now_iso(),
        "units": {
            "translation": "mm",
            "rotation": "deg (extrinsic XYZ Euler)",
        },
        "poses": poses,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved {len(poses)} pose(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
