"""Fairino arm pose reader.

Wraps the FAIR-INNOVATION/fairino-python-sdk Robot.RPC client. The exact import
path depends on how the SDK is installed -- some builds expose `from fairino
import Robot`, others ship a top-level `Robot` module. We try both.

Public API:
    rc = RobotClient(ip)
    T_base_gripper = rc.read_pose_se3()       # 4x4 SE(3), meters
    pose_mm_deg    = rc.read_pose_raw()       # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from loguru import logger

from .geometry import fairino_pose_to_se3


def _import_robot_module() -> Any:
    """Import the Fairino Robot module; raise a friendly error otherwise."""
    try:
        from fairino import Robot  # type: ignore[import-not-found]
        return Robot
    except ImportError:
        pass
    try:
        import Robot  # type: ignore[import-not-found]
        return Robot
    except ImportError as e:
        raise ImportError(
            "fairino SDK not installed. Install from "
            "https://github.com/FAIR-INNOVATION/fairino-python-sdk"
        ) from e


class RobotClient:
    """Thin pose-reader wrapper around fairino Robot.RPC."""

    def __init__(self, ip: str):
        self._Robot = _import_robot_module()
        logger.info("connecting to fairino controller at {}", ip)
        self._rpc = self._Robot.RPC(ip)

    def read_pose_raw(self) -> np.ndarray:
        """Returns Fairino-native pose: [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
        # GetActualTCPPose(flag): flag=0 typically means blocking read
        result = self._rpc.GetActualTCPPose(0)
        # SDK returns (err_code, pose_list) on success, or just the list
        # depending on version; handle both.
        if (
            isinstance(result, (tuple, list))
            and len(result) == 2
            and isinstance(result[0], (int, np.integer))
        ):
            err, pose = result
            if err != 0:
                raise RuntimeError(f"GetActualTCPPose failed with error code {err}")
        elif isinstance(result, dict):
            pose = result.get("pose") or result.get("tcp_pose") or result.get("data")
            if pose is None:
                raise RuntimeError(f"unexpected pose dict from controller: {result}")
        else:
            pose = result
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise RuntimeError(f"unexpected pose shape from controller: {pose}")
        return pose

    def read_pose_se3(self) -> np.ndarray:
        """Returns 4x4 SE(3) base->gripper (meters)."""
        return fairino_pose_to_se3(self.read_pose_raw())

    def close(self) -> None:
        """Best-effort shutdown for SDK variants that expose a close method."""
        for name in ("CloseRPC", "Logout", "Close", "close"):
            fn = getattr(self._rpc, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:  # noqa: BLE001 -- shutdown must not mask capture results
                    logger.debug("{} during robot close raised: {}", name, e)
                return


class MockRobotClient:
    """Deterministic robot pose source for hardware-free collection tests."""

    _POSES = np.asarray(
        [
            [420.0, -150.0, 350.0, 0.0, 0.0, 0.0],
            [438.0, -128.0, 362.0, 18.0, -6.0, 12.0],
            [405.0, -116.0, 372.0, -16.0, 14.0, 28.0],
            [462.0, -142.0, 334.0, 25.0, 21.0, -18.0],
            [446.0, -172.0, 390.0, -30.0, -12.0, 36.0],
            [416.0, -196.0, 318.0, 12.0, 32.0, -34.0],
            [482.0, -104.0, 344.0, -22.0, 26.0, 52.0],
            [392.0, -162.0, 382.0, 35.0, -25.0, -46.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, start_index: int = 0, poses: Iterable[Iterable[float]] | None = None):
        self._poses = np.asarray(list(poses), dtype=np.float64) if poses is not None else self._POSES
        if self._poses.ndim != 2 or self._poses.shape[1] != 6:
            raise ValueError("mock poses must be an Nx6 array")
        self._i = int(start_index)

    def read_pose_raw(self) -> np.ndarray:
        base = self._poses[self._i % len(self._poses)].copy()
        cycle = self._i // len(self._poses)
        if cycle:
            base[:3] += np.array([12.0, -9.0, 7.0]) * cycle
            base[3:] += np.array([7.0, -5.0, 11.0]) * cycle
        self._i += 1
        return base

    def read_pose_se3(self) -> np.ndarray:
        return fairino_pose_to_se3(self.read_pose_raw())

    def close(self) -> None:
        return None
