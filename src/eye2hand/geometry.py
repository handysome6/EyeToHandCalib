"""Geometry helpers: SE(3) <-> Fairino RPY (mm/deg), inversion, Umeyama.

All internal SE(3) is 4x4 numpy with translation in **meters** and rotation as
a right-handed rotation matrix. Conversions to/from the robot's native
(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) live here so the rest of the code
deals only with metric SE(3).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------- SE(3) construction / inversion ----------

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose 4x4 SE(3) from 3x3 rotation and 3-vector translation (meters)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Rigid inverse of SE(3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ---------- Fairino-style pose conversion ----------
#
# Fairino reports TCP pose as [x, y, z, rx, ry, rz] with x/y/z in millimetres
# and rx/ry/rz Euler angles in degrees. The convention used by Fairino is
# fixed-axis XYZ Euler (a.k.a. ZYX intrinsic, RPY) — the same as cv2.Rodrigues
# of the equivalent rotation matrix is NOT this; we do explicit Euler.
#
# If your controller turns out to use a different ordering, change the order
# string in `_euler_to_R` / `_R_to_euler` here only.

_EULER_ORDER = "xyz"  # extrinsic (fixed-axis) XYZ; equivalent to intrinsic ZYX (RPY)


def _euler_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # extrinsic xyz: R = Rz @ Ry @ Rx (apply x first in world frame)
    return Rz @ Ry @ Rx


def _R_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of _euler_to_R, returns (rx_deg, ry_deg, rz_deg)."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)
    if abs(sy) < 1.0 - 1e-9:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # gimbal lock; roll absorbed into yaw
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return float(np.rad2deg(rx)), float(np.rad2deg(ry)), float(np.rad2deg(rz))


def fairino_pose_to_se3(pose_mm_deg: np.ndarray | list | tuple) -> np.ndarray:
    """Convert [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg] -> 4x4 SE(3) (meters)."""
    p = np.asarray(pose_mm_deg, dtype=np.float64).reshape(6)
    t_m = p[:3] * 1e-3
    R = _euler_to_R(p[3], p[4], p[5])
    return se3(R, t_m)


def se3_to_fairino_pose(T: np.ndarray) -> np.ndarray:
    """Inverse of `fairino_pose_to_se3`. Returns mm/deg vector of length 6."""
    rx, ry, rz = _R_to_euler(T[:3, :3])
    t_mm = T[:3, 3] * 1e3
    return np.array([t_mm[0], t_mm[1], t_mm[2], rx, ry, rz], dtype=np.float64)


# ---------- pose-diversity metric ----------

def rotation_angle_deg(R: np.ndarray) -> float:
    """Magnitude of rotation R, in degrees."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_theta)))


def relative_rotation_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Rotation angle between two SE(3) poses, in degrees."""
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    return rotation_angle_deg(R_rel)


# ---------- 3D-3D rigid registration (Umeyama / Kabsch) ----------

def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Best-fit rigid transform T such that `T @ src ≈ dst` for paired 3D points.

    src, dst: (N, 3). Returns 4x4 SE(3). Rotation only, no scale.
    Implements Kabsch with reflection guard (Umeyama 1991, with scale fixed to 1).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3 and src.shape[0] >= 3, \
        f"umeyama needs paired Nx3 arrays with N>=3, got {src.shape} and {dst.shape}"

    cs = src.mean(axis=0)
    cd = dst.mean(axis=0)
    s = src - cs
    d = dst - cd
    H = s.T @ d  # 3x3
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0
    R = (Vt.T @ D @ U.T)
    t = cd - R @ cs
    return se3(R, t)


def umeyama_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    inlier_thresh_m: float = 0.005,
    iters: int = 200,
    min_sample: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC wrapper around `umeyama`. Returns (T, inlier_mask)."""
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(src)
    if n < min_sample:
        raise ValueError(f"need at least {min_sample} correspondences, got {n}")

    best_inliers = np.zeros(n, dtype=bool)
    best_T = None
    src_h = src
    dst_h = dst

    for _ in range(iters):
        idx = rng.choice(n, size=min_sample, replace=False)
        try:
            T_try = umeyama(src_h[idx], dst_h[idx])
        except np.linalg.LinAlgError:
            continue
        pred = (T_try[:3, :3] @ src_h.T).T + T_try[:3, 3]
        err = np.linalg.norm(pred - dst_h, axis=1)
        inliers = err < inlier_thresh_m
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_T = T_try

    if best_T is None or best_inliers.sum() < min_sample:
        # fallback: plain Umeyama on all points
        return umeyama(src, dst), np.ones(n, dtype=bool)

    # final refit on inliers
    refined = umeyama(src_h[best_inliers], dst_h[best_inliers])
    return refined, best_inliers


# ---------- camera back-projection helper ----------

def backproject_pixel(u: float, v: float, z: float, K: np.ndarray) -> np.ndarray:
    """Lift a single (u, v) pixel + depth z (meters) to a camera-frame 3D point."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float64)


# ---------- residuals ----------

def residual_translation_m(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def residual_rotation_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return relative_rotation_deg(T_a, T_b)


# ---------- OpenCV interop helpers for hand-eye ----------

def split_rt(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split SE(3) into (R, t) numpy arrays for cv2.calibrateHandEye."""
    return T[:3, :3].copy(), T[:3, 3].reshape(3, 1).copy()


def rvec_tvec_to_se3(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV (rvec, tvec) -> 4x4 SE(3). tvec already in meters."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return se3(R, np.asarray(tvec, dtype=np.float64).reshape(3))
