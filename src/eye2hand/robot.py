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

from typing import Any

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
        if isinstance(result, tuple) and len(result) == 2:
            err, pose = result
            if err != 0:
                raise RuntimeError(f"GetActualTCPPose failed with error code {err}")
        else:
            pose = result
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise RuntimeError(f"unexpected pose shape from controller: {pose}")
        return pose

    def read_pose_se3(self) -> np.ndarray:
        """Returns 4x4 SE(3) base->gripper (meters)."""
        return fairino_pose_to_se3(self.read_pose_raw())
