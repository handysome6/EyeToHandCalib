"""Eye-to-hand AX=XB solver wrapping cv2.calibrateHandEye.

For an *eye-to-hand* configuration (camera fixed in world, target rigidly
attached to the gripper) the standard hand-eye formulation
    A_i * X = X * B_i
becomes, with cv2.calibrateHandEye's signature:

    R_gripper2base, t_gripper2base   <-- inverted robot poses
    R_target2cam,   t_target2cam     <-- as-is

The returned matrix is stored under the historical field name `T_cam_base`, but
it is the camera pose in the robot base frame: it maps camera-frame points into
base-frame coordinates (`^baseT_cam`). Its inverse, `T_base_cam`, maps base-frame
points into camera-frame coordinates (`^camT_base`).

We invert the gripper->base poses ourselves and pass the resulting list as
"gripper2base" to OpenCV: that flips the equation into the eye-to-hand form.
This is the convention the OpenCV samples document for eye-to-hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from loguru import logger

from .geometry import (
    relative_rotation_deg,
    residual_translation_m,
    se3,
    se3_inv,
    split_rt,
)


_METHOD_MAP = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeSolution:
    method: str
    T_cam_base: np.ndarray         # camera pose in base frame: camera -> base (4x4)
    T_base_cam: np.ndarray         # inverse: base -> camera
    rmse_translation_m: float      # mean |t_A X - X t_B| residual
    rmse_rotation_deg: float       # mean rotation residual on AX=XB closure
    n_pairs: int


def solve(
    base_T_gripper: Sequence[np.ndarray],
    cam_T_target: Sequence[np.ndarray],
    *,
    method: str = "DANIILIDIS",
) -> HandEyeSolution:
    """Solve eye-to-hand for one method.

    `base_T_gripper`: SE(3) base->gripper for each pose (read from robot).
    `cam_T_target`:   SE(3) target->camera for each pose (from depth-lifted solve).
    """
    if method not in _METHOD_MAP:
        raise ValueError(f"unknown method {method}; choose from {list(_METHOD_MAP)}")
    if len(base_T_gripper) != len(cam_T_target):
        raise ValueError("len(base_T_gripper) != len(cam_T_target)")
    if len(base_T_gripper) < 3:
        raise ValueError("hand-eye needs >=3 pose pairs (and ideally >=10)")

    # Eye-to-hand: invert base->gripper before passing it to OpenCV. With this
    # convention the returned X is the fixed camera pose in the robot base frame.
    R_g2b, t_g2b = [], []
    for T_bg in base_T_gripper:
        T_gb = se3_inv(T_bg)
        R, t = split_rt(T_gb)
        R_g2b.append(R)
        t_g2b.append(t)

    R_t2c, t_t2c = [], []
    for T_tc in cam_T_target:
        R, t = split_rt(T_tc)
        R_t2c.append(R)
        t_t2c.append(t)

    R_X, t_X = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=_METHOD_MAP[method],
    )
    T_cam_base = se3(np.asarray(R_X), np.asarray(t_X).reshape(3))
    T_base_cam = se3_inv(T_cam_base)

    # Residuals: for each pair, compute the AX vs XB closure error.
    # Form: A_i X = X B_i  with A_i = T_g2b_i_to_j, B_i = T_t2c_i_to_j.
    # Using consecutive-pair deltas:
    # Predict-vs-measured residual.
    #
    # Notation note: the variables are named after OpenCV's "AtoB" outputs, so
    # `T_cam_base` here is the matrix that takes a CAM-frame point to BASE
    # frame (cv2.calibrateHandEye's "cam2base" output), and `T_base_cam` is
    # its inverse. Likewise `cam_T_target[i]` takes a TARGET-frame point to
    # CAM frame (the target2cam input), and `base_T_gripper[i]` takes a
    # GRIPPER-frame point to BASE frame.
    #
    # Composition (left-to-right multiplication on column vectors):
    #   target -> gripper -> base -> cam:
    #   target2cam_i = base2cam @ gripper2base_i @ target2gripper
    # i.e.
    #   cam_T_target[i] = T_base_cam @ base_T_gripper[i] @ T_gripper_target
    #
    # Solve T_gripper_target ("target2gripper") from pose 0, then predict.
    n = len(base_T_gripper)
    T_gripper_target = (
        se3_inv(base_T_gripper[0]) @ T_cam_base @ cam_T_target[0]
    )
    t_errs, r_errs = [], []
    for i in range(1, n):
        pred = T_base_cam @ base_T_gripper[i] @ T_gripper_target
        t_errs.append(residual_translation_m(pred, cam_T_target[i]))
        r_errs.append(relative_rotation_deg(pred, cam_T_target[i]))

    rmse_t = float(np.sqrt(np.mean(np.array(t_errs) ** 2))) if t_errs else float("nan")
    rmse_r = float(np.sqrt(np.mean(np.array(r_errs) ** 2))) if r_errs else float("nan")

    return HandEyeSolution(
        method=method,
        T_cam_base=T_cam_base,
        T_base_cam=T_base_cam,
        rmse_translation_m=rmse_t,
        rmse_rotation_deg=rmse_r,
        n_pairs=n,
    )


def solve_all(
    base_T_gripper: Sequence[np.ndarray],
    cam_T_target: Sequence[np.ndarray],
    *,
    methods: Iterable[str] = ("TSAI", "PARK", "DANIILIDIS"),
) -> list[HandEyeSolution]:
    """Run several solvers; useful to cross-check stability."""
    out = []
    for m in methods:
        try:
            sol = solve(base_T_gripper, cam_T_target, method=m)
            logger.info(
                "{}: T_rmse={:.4f} m, R_rmse={:.4f} deg over {} pairs",
                m, sol.rmse_translation_m, sol.rmse_rotation_deg, sol.n_pairs,
            )
            out.append(sol)
        except Exception as e:
            logger.error("{} failed: {}", m, e)
    return out


def save_solution(sol: HandEyeSolution, out_path: Path | str) -> Path:
    import json
    out_path = Path(out_path)
    payload = {
        "method": sol.method,
        "T_cam_base": sol.T_cam_base.tolist(),
        "T_base_cam": sol.T_base_cam.tolist(),
        "rmse_translation_m": float(sol.rmse_translation_m),
        "rmse_rotation_deg": float(sol.rmse_rotation_deg),
        "n_pairs": int(sol.n_pairs),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
