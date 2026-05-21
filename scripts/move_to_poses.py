#!/usr/bin/env python
"""Convert 7-element poses (qw,qx,qy,qz,tx,ty,tz) to TCP and move the robot.

Asks for confirmation before each move.

Usage:
    python scripts/move_to_poses.py
    python scripts/move_to_poses.py --mock   # no hardware
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config

# fmt: off
# Each row: qw, qx, qy, qz, tx(m), ty(m), tz(m)
POSES_7 = np.array([
    [+0.2723, +0.6221, +0.7002, +0.2204,  -0.01899, -0.38644, +0.77681],
    [-0.0268, -0.2493, +0.9254, +0.2842,  -0.07665, -0.23776, +0.97920],
    [-0.0881, -0.3432, +0.6524, +0.6700,  -0.33331, -0.50539, +1.00056],
    [-0.3083, +0.0566, -0.7573, -0.5729,  -0.15901, -0.08815, +0.94834],
    [-0.3374, -0.2130, +0.9121, +0.0936,  +0.17795, -0.02192, +1.09323],
])
# fmt: on


def pose7_to_tcp(pose7: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz, tx, ty, tz] to [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
    qw, qx, qy, qz = pose7[0], pose7[1], pose7[2], pose7[3]
    tx, ty, tz = pose7[4], pose7[5], pose7[6]

    # scipy expects [qx, qy, qz, qw] (scalar-last)
    r = Rotation.from_quat([qx, qy, qz, qw])
    rx, ry, rz = r.as_euler("XYZ", degrees=True)

    x_mm, y_mm, z_mm = tx * 1000.0, ty * 1000.0, tz * 1000.0
    return np.array([x_mm, y_mm, z_mm, rx, ry, rz])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mock", action="store_true", help="dry run without hardware")
    ap.add_argument("--vel", type=float, default=10.0, help="move velocity (default 10)")
    ap.add_argument("--tool", type=int, default=0, help="tool coordinate number")
    ap.add_argument("--user", type=int, default=0, help="user coordinate number")
    args = ap.parse_args()

    cfg = load_config(args.config)

    tcp_poses = [pose7_to_tcp(p) for p in POSES_7]

    print("Converted TCP poses (mm, deg):")
    print(f"{'Node':>4}  {'x_mm':>10} {'y_mm':>10} {'z_mm':>10}  {'rx':>10} {'ry':>10} {'rz':>10}")
    print("-" * 74)
    for i, tcp in enumerate(tcp_poses, 1):
        print(f"{i:>4}  {tcp[0]:>10.2f} {tcp[1]:>10.2f} {tcp[2]:>10.2f}  {tcp[3]:>10.2f} {tcp[4]:>10.2f} {tcp[5]:>10.2f}")
    print()

    if args.mock:
        print("[mock mode] No robot connection.")
        for i, tcp in enumerate(tcp_poses, 1):
            ans = input(f"Move to pose {i}? (y/n/q): ").strip().lower()
            if ans == "q":
                break
            if ans != "y":
                print(f"  Skipped pose {i}")
                continue
            print(f"  [mock] Would MoveL to {tcp.tolist()}")
        return 0

    from eye2hand.robot import RobotClient
    robot_mod = __import__("fairino", fromlist=["Robot"]).Robot
    rpc = robot_mod.RPC(cfg.robot.ip)
    print(f"Connected to robot at {cfg.robot.ip}\n")

    try:
        for i, tcp in enumerate(tcp_poses, 1):
            print(f"--- Pose {i}/{len(tcp_poses)} ---")
            print(f"  Target: [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}, {tcp[3]:.2f}, {tcp[4]:.2f}, {tcp[5]:.2f}]")
            ans = input(f"  Move to pose {i}? (y/n/q): ").strip().lower()
            if ans == "q":
                print("Aborted by user.")
                break
            if ans != "y":
                print(f"  Skipped pose {i}")
                continue

            desc_pos = tcp.tolist()
            err = rpc.MoveL(desc_pos, args.tool, args.user, vel=args.vel)
            if err != 0:
                print(f"  MoveL error: {err}")
            else:
                print(f"  MoveL complete.")
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
