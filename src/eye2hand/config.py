"""Calibration config loader.

Loads `configs/calib.yaml` (or a path passed via $EYE2HAND_CONFIG) into a
small dataclass so the rest of the code can use attribute access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "calib.yaml"


@dataclass
class BoardCfg:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float


@dataclass
class RobotCfg:
    ip: str


@dataclass
class CaptureCfg:
    min_pose_diversity_deg: float
    min_poses: int
    target_poses: int


@dataclass
class TargetPoseCfg:
    depth_patch_radius_px: int
    depth_min_m: float
    depth_max_m: float
    depth_neighbor_max_dev_m: float
    min_valid_corners: int


@dataclass
class HandEyeCfg:
    methods: list[str]
    output_filename: str


@dataclass
class CalibConfig:
    jetson_reborn_path: Path
    data_root: Path
    robot: RobotCfg
    board: BoardCfg
    capture: CaptureCfg
    target_pose: TargetPoseCfg
    handeye: HandEyeCfg

    @property
    def poses_dir(self) -> Path:
        return REPO_ROOT / "data" / "poses"


def _expand(p: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()


def load_config(path: Optional[str | Path] = None) -> CalibConfig:
    cfg_path = Path(path) if path is not None else Path(os.environ.get("EYE2HAND_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    return CalibConfig(
        jetson_reborn_path=_expand(raw["jetson_reborn_path"]),
        data_root=_expand(raw["data_root"]),
        robot=RobotCfg(**raw["robot"]),
        board=BoardCfg(**raw["board"]),
        capture=CaptureCfg(**raw["capture"]),
        target_pose=TargetPoseCfg(**raw["target_pose"]),
        handeye=HandEyeCfg(**raw["handeye"]),
    )
