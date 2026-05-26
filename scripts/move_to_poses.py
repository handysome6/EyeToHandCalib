#!/usr/bin/env python
"""Load predicted poses from a JSON file and move the robot.

Reads a predictions JSON file (with X_mm, Y_mm, Z_mm and XYZ_intrinsic Euler
angles) and converts each entry to Fairino TCP format. Asks for confirmation
before each move.

Usage:
    python scripts/move_to_poses.py --poses predictions.json --mock
    python scripts/move_to_poses.py --poses predictions.json
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
    Euler angles are XYZ intrinsic (matching Fairino convention).
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, type=Path, help="predictions JSON file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="dry run without hardware")
    ap.add_argument("--mode", choices=["MoveL", "MoveJ", "MoveCart"], default="MoveJ",
                    help="motion type: MoveL (linear), MoveJ (joint PTP), MoveCart (cartesian PTP)")
    ap.add_argument("--vel", type=float, default=10.0, help="move velocity (default 10)")
    ap.add_argument("--tool", type=int, default=0, help="tool coordinate number")
    ap.add_argument("--user", type=int, default=0, help="user coordinate number")
    args = ap.parse_args()

    cfg = load_config(args.config)
    poses = load_tcp_poses(args.poses)

    print(f"Loaded {len(poses)} poses from {args.poses}")
    print(f"{'Node':>4}  {'x_mm':>10} {'y_mm':>10} {'z_mm':>10}  {'rx':>10} {'ry':>10} {'rz':>10}")
    print("-" * 74)
    for node, tcp in poses:
        print(f"{node:>4}  {tcp[0]:>10.2f} {tcp[1]:>10.2f} {tcp[2]:>10.2f}  {tcp[3]:>10.2f} {tcp[4]:>10.2f} {tcp[5]:>10.2f}")
    print()

    if args.mock:
        print("[mock mode] No robot connection.")
        for node, tcp in poses:
            ans = input(f"Move to node {node}? (y/n/q): ").strip().lower()
            if ans == "q":
                break
            if ans != "y":
                print(f"  Skipped node {node}")
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
        for i, (node, tcp) in enumerate(poses, 1):
            print(f"--- Node {node} ({i}/{len(poses)}) ---")
            print(f"  Target: [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}, {tcp[3]:.2f}, {tcp[4]:.2f}, {tcp[5]:.2f}]")
            ans = input(f"  Move to node {node}? (y/n/q): ").strip().lower()
            if ans == "q":
                print("Aborted by user.")
                break
            if ans != "y":
                print(f"  Skipped node {node}")
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
