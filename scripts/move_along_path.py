#!/usr/bin/env python
"""Move the robot along a path with intermediate arc points between poses.

Reads a predictions JSON file and inserts an intermediate waypoint between
each adjacent pair of poses.  The waypoint sits at the midpoint of the two
poses, offset +200 mm along the Y axis, creating a rounded path.

Usage:
    python scripts/move_along_path.py --poses predictions.json --mock
    python scripts/move_along_path.py --poses predictions.json --no-confirm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config

def load_tcp_poses(path: Path) -> list[tuple[int, np.ndarray]]:
    """Load predictions JSON and return list of (node, tcp_array).

    TCP array: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
    """
    data = json.loads(path.read_text())
    results = []
    for pred in data["predictions"]:
        node = pred["node"]
        x_mm = pred["X_mm"]
        y_mm = pred["Y_mm"]
        z_mm = pred["Z_mm"]
        rx, ry, rz = pred["euler_deg"]["XYZ_intrinsic"]
        results.append((node, np.array([x_mm, y_mm, z_mm, rx, ry, rz])))
    return results


def build_path(poses: list[tuple[int, np.ndarray]], y_offset_mm: float) -> list[tuple[str, np.ndarray]]:
    """Build the full path with intermediate waypoints.

    Returns list of (label, tcp_array).  Between node A and node B, two
    intermediate points are inserted, both with Y offset.  The first keeps
    A's X, the second keeps B's X.  Z and orientation are averaged.
    """
    path: list[tuple[str, np.ndarray]] = []
    for i, (node, tcp) in enumerate(poses):
        path.append((f"node {node}", tcp))
        if i < len(poses) - 1:
            next_node, next_tcp = poses[i + 1]
            avg = (tcp + next_tcp) / 2.0

            wp1 = avg.copy()
            wp1[0] = tcp[0]
            wp1[1] = avg[1] + y_offset_mm
            path.append((f"arc {node}-{next_node}a", wp1))

            wp2 = avg.copy()
            wp2[0] = next_tcp[0]
            wp2[1] = avg[1] + y_offset_mm
            path.append((f"arc {node}-{next_node}b", wp2))
    return path


def print_path(path: list[tuple[str, np.ndarray]]) -> None:
    print(f"{'#':>3}  {'Label':<12} {'x_mm':>10} {'y_mm':>10} {'z_mm':>10}  {'rx':>10} {'ry':>10} {'rz':>10}")
    print("-" * 88)
    for i, (label, tcp) in enumerate(path, 1):
        print(f"{i:>3}  {label:<12} {tcp[0]:>10.2f} {tcp[1]:>10.2f} {tcp[2]:>10.2f}  {tcp[3]:>10.2f} {tcp[4]:>10.2f} {tcp[5]:>10.2f}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, type=Path, help="predictions JSON file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="dry run without hardware")
    ap.add_argument("--y-offset", type=float, default=200.0,
                    help="Y offset in mm for intermediate waypoints (default 200)")
    ap.add_argument("--no-confirm", action="store_true",
                    help="move to all points without asking for confirmation")
    ap.add_argument("--mode", choices=["MoveL", "MoveJ", "MoveCart"], default="MoveJ",
                    help="motion type: MoveL (linear), MoveJ (joint PTP), MoveCart (cartesian PTP)")
    ap.add_argument("--vel", type=float, default=10.0, help="move velocity (default 10)")
    ap.add_argument("--tool", type=int, default=0, help="tool coordinate number")
    ap.add_argument("--user", type=int, default=0, help="user coordinate number")
    args = ap.parse_args()

    cfg = load_config(args.config)
    poses = load_tcp_poses(args.poses)
    path = build_path(poses, args.y_offset)

    print(f"Loaded {len(poses)} poses → {len(path)} waypoints (with intermediates)")
    print_path(path)

    if args.mock:
        print("[mock mode] No robot connection.")
        for i, (label, tcp) in enumerate(path, 1):
            if not args.no_confirm:
                ans = input(f"Move to {label} ({i}/{len(path)})? (y/n/q): ").strip().lower()
                if ans == "q":
                    break
                if ans != "y":
                    print(f"  Skipped {label}")
                    continue
            print(f"  [mock] Would {args.mode} to {tcp.tolist()}")
        return 0

    robot_mod = __import__("fairino", fromlist=["Robot"]).Robot
    rpc = robot_mod.RPC(cfg.robot.ip)
    print(f"Connected to robot at {cfg.robot.ip}")

    err = rpc.Mode(0)
    if err != 0:
        print(f"Warning: Mode(0) returned {err}")
    err = rpc.RobotEnable(1)
    if err != 0:
        print(f"Warning: RobotEnable(1) returned {err}")
    print()

    try:
        for i, (label, tcp) in enumerate(path, 1):
            print(f"--- {label} ({i}/{len(path)}) ---")
            print(f"  Target: [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}, {tcp[3]:.2f}, {tcp[4]:.2f}, {tcp[5]:.2f}]")

            if not args.no_confirm:
                ans = input(f"  Move to {label}? (y/n/q): ").strip().lower()
                if ans == "q":
                    print("Aborted by user.")
                    break
                if ans != "y":
                    print(f"  Skipped {label}")
                    continue

            desc_pos = tcp.tolist()
            if args.mode == "MoveJ":
                err = rpc.MoveJ([0.0]*6, args.tool, args.user, desc_pos=desc_pos, vel=args.vel)
            elif args.mode == "MoveCart":
                err = rpc.MoveCart(desc_pos, args.tool, args.user, vel=args.vel)
            else:
                err = rpc.MoveL(desc_pos, args.tool, args.user, vel=args.vel)
            if err != 0:
                print(f"  {args.mode} error: {err}")
            else:
                print(f"  {args.mode} complete.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        for name in ("CloseRPC", "Logout", "Close", "close"):
            fn = getattr(rpc, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
