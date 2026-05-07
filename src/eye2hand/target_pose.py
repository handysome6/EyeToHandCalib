"""Target-pose extraction from a rectified-left image + metric depth map.

Given a ChArUco board rigidly attached to the gripper, for each captured pose:

  1. Detect ChArUco corners at sub-pixel accuracy on `img0.jpg` (the rectified
     left image at the same resolution as `depth_meter.npy`).
  2. For each detected corner, sample `depth_meter` over a small patch (median,
     consistency-checked against neighbours), back-project to a 3D camera-frame
     point with the matching `K_scaled`.
  3. Solve `T_target_cam` by Umeyama 3D-3D rigid registration on the (board_3D,
     cam_3D) pairs, with RANSAC outlier rejection.
  4. If too few corners have valid depth (e.g., near board edges or in specular
     regions) fall back to `cv2.solvePnP` on the 2D corners.

The board model in board frame: z = 0 plane, origin at corner (0, 0), x along
columns (squares_x direction), y along rows. ChArUco interior corners have
known integer (col, row) indices that map to (col*square_len, row*square_len, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from .config import BoardCfg, TargetPoseCfg
from .geometry import backproject_pixel, rvec_tvec_to_se3, se3, umeyama_ransac
from .pipeline import PcdOutputs


@dataclass
class TargetPoseResult:
    T_target_cam: np.ndarray            # 4x4 SE(3): board frame -> camera frame
    method: str                          # "umeyama_depth" or "solvepnp"
    n_corners_detected: int
    n_corners_used: int
    rmse_m: float                        # RMSE of the 3D-3D residuals (m); -1 for PnP
    reprojection_rmse_px: float          # 2D reprojection RMSE on detected corners
    inlier_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


# ---------- ChArUco board construction ----------

def make_charuco(board: BoardCfg) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    """Build a cv2.aruco.CharucoBoard from our config dataclass."""
    dict_id = getattr(cv2.aruco, board.dictionary, None)
    if dict_id is None:
        raise ValueError(f"unknown aruco dictionary: {board.dictionary}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    cb = cv2.aruco.CharucoBoard(
        size=(board.squares_x, board.squares_y),
        squareLength=float(board.square_length_m),
        markerLength=float(board.marker_length_m),
        dictionary=aruco_dict,
    )
    return cb, aruco_dict


def _board_corner_3d(charuco_id: int, board_cfg: BoardCfg) -> np.ndarray:
    """Map a ChArUco interior corner id -> (x, y, 0) in board-frame meters.

    OpenCV indexes interior corners row-major across the board: there are
    (squares_x - 1) * (squares_y - 1) of them. Corner i is at
    (col = i % (sx-1) + 1, row = i // (sx-1) + 1) in *square* units.
    """
    sx = board_cfg.squares_x
    s = float(board_cfg.square_length_m)
    cols = sx - 1
    col = (charuco_id % cols) + 1
    row = (charuco_id // cols) + 1
    return np.array([col * s, row * s, 0.0], dtype=np.float64)


# ---------- depth sampling ----------

def _sample_depth(
    depth: np.ndarray,
    u: float,
    v: float,
    radius: int,
    z_min: float,
    z_max: float,
    max_dev: float,
) -> Optional[float]:
    """Median depth in a small patch around (u,v); rejects if patch is too inconsistent."""
    H, W = depth.shape[:2]
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < W and 0 <= iv < H):
        return None
    u0 = max(0, iu - radius)
    u1 = min(W, iu + radius + 1)
    v0 = max(0, iv - radius)
    v1 = min(H, iv + radius + 1)
    patch = depth[v0:v1, u0:u1]
    valid = patch[(patch > z_min) & (patch < z_max) & np.isfinite(patch)]
    if valid.size < 3:
        return None
    z = float(np.median(valid))
    if not (z_min < z < z_max):
        return None
    if float(np.max(valid) - np.min(valid)) > max_dev:
        return None
    return z


# ---------- main entrypoint ----------

def compute_target_pose(
    pcd: PcdOutputs,
    board_cfg: BoardCfg,
    tp_cfg: TargetPoseCfg,
    *,
    debug_dir: Optional[Path] = None,
) -> TargetPoseResult:
    """Run depth-lifted ChArUco -> SE(3) target pose extraction."""
    cb, aruco_dict = make_charuco(board_cfg)
    detector = cv2.aruco.CharucoDetector(cb)

    img = cv2.imread(str(pcd.img0_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read {pcd.img0_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        raise RuntimeError(f"ChArUco detection failed on {pcd.img0_path} (got {0 if charuco_ids is None else len(charuco_ids)} corners)")

    # sub-pixel refine (CharucoDetector already does it, but cheap insurance)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    corners_2d = charuco_corners.reshape(-1, 1, 2).astype(np.float32)
    corners_2d = cv2.cornerSubPix(gray, corners_2d, (5, 5), (-1, -1), crit)
    corners_2d = corners_2d.reshape(-1, 2)

    ids = charuco_ids.flatten().astype(int)
    n_detected = len(ids)
    logger.info("detected {} ChArUco corners on {}", n_detected, pcd.img0_path.name)

    # build 3D board points and 3D camera points for matched corners
    board_pts: list[np.ndarray] = []
    cam_pts: list[np.ndarray] = []
    valid_idx: list[int] = []  # index into corners_2d that has valid depth
    for i, cid in enumerate(ids):
        u, v = float(corners_2d[i, 0]), float(corners_2d[i, 1])
        z = _sample_depth(
            pcd.depth_meter, u, v,
            radius=tp_cfg.depth_patch_radius_px,
            z_min=tp_cfg.depth_min_m,
            z_max=tp_cfg.depth_max_m,
            max_dev=tp_cfg.depth_neighbor_max_dev_m,
        )
        if z is None:
            continue
        cam_pts.append(backproject_pixel(u, v, z, pcd.K_scaled))
        board_pts.append(_board_corner_3d(int(cid), board_cfg))
        valid_idx.append(i)

    n_used = len(board_pts)
    logger.info("{} / {} corners had valid depth", n_used, n_detected)

    # all corners' 3D in board frame for PnP fallback / reprojection
    board_pts_all = np.stack([_board_corner_3d(int(c), board_cfg) for c in ids], axis=0)

    if n_used >= tp_cfg.min_valid_corners:
        src = np.stack(board_pts, axis=0)
        dst = np.stack(cam_pts, axis=0)
        T, inliers = umeyama_ransac(src, dst, inlier_thresh_m=0.005, iters=200)
        # 3D-3D residual on inliers
        pred = (T[:3, :3] @ src.T).T + T[:3, 3]
        err3d = np.linalg.norm(pred[inliers] - dst[inliers], axis=1)
        rmse_3d = float(np.sqrt(np.mean(err3d ** 2))) if err3d.size else float("nan")
        method = "umeyama_depth"
    else:
        logger.warning("falling back to solvePnP (only {} valid corners < {})", n_used, tp_cfg.min_valid_corners)
        # PnP on the 2D corners with the *scaled* intrinsics, dist=zeros (rectified)
        ok, rvec, tvec = cv2.solvePnP(
            board_pts_all.astype(np.float32),
            corners_2d.astype(np.float32),
            pcd.K_scaled.astype(np.float32),
            np.zeros(5, dtype=np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed on fallback path")
        T = rvec_tvec_to_se3(rvec, tvec)
        rmse_3d = -1.0
        inliers = np.zeros(n_used, dtype=bool)
        method = "solvepnp"

    # 2D reprojection RMSE on all detected corners (sanity metric)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    proj, _ = cv2.projectPoints(
        board_pts_all.astype(np.float32),
        rvec, T[:3, 3].astype(np.float32),
        pcd.K_scaled.astype(np.float32),
        np.zeros(5, dtype=np.float32),
    )
    proj = proj.reshape(-1, 2)
    err_px = np.linalg.norm(proj - corners_2d, axis=1)
    rmse_px = float(np.sqrt(np.mean(err_px ** 2)))

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        vis = img.copy()
        for (u, v), e in zip(corners_2d, err_px):
            color = (0, 255, 0) if e < 1.5 else (0, 165, 255) if e < 3.0 else (0, 0, 255)
            cv2.circle(vis, (int(round(u)), int(round(v))), 4, color, 1)
        # axes
        axis_len = float(board_cfg.square_length_m) * 3.0
        axis_pts = np.array([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len]], dtype=np.float32)
        ap, _ = cv2.projectPoints(axis_pts, rvec, T[:3, 3].astype(np.float32),
                                  pcd.K_scaled.astype(np.float32), np.zeros(5, dtype=np.float32))
        ap = ap.reshape(-1, 2).astype(int)
        cv2.line(vis, tuple(ap[0]), tuple(ap[1]), (0, 0, 255), 2)
        cv2.line(vis, tuple(ap[0]), tuple(ap[2]), (0, 255, 0), 2)
        cv2.line(vis, tuple(ap[0]), tuple(ap[3]), (255, 0, 0), 2)
        cv2.imwrite(str(debug_dir / "target_pose_vis.png"), vis)

    return TargetPoseResult(
        T_target_cam=T,
        method=method,
        n_corners_detected=n_detected,
        n_corners_used=n_used,
        rmse_m=rmse_3d,
        reprojection_rmse_px=rmse_px,
        inlier_mask=inliers,
    )


def save_target_pose(result: TargetPoseResult, out_path: Path | str) -> Path:
    import json
    out_path = Path(out_path)
    payload = {
        "T_target_cam": result.T_target_cam.tolist(),
        "method": result.method,
        "n_corners_detected": int(result.n_corners_detected),
        "n_corners_used": int(result.n_corners_used),
        "rmse_m": float(result.rmse_m),
        "reprojection_rmse_px": float(result.reprojection_rmse_px),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path
