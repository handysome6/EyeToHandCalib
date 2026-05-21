#!/usr/bin/env python
"""Calibrate from the flat `/Users/andyliu/DCIM_AI` collected-data layout.

The original pipeline scripts expect pose folders under `data/poses/<id>/` with
broker outputs nested below `pcd/`. The May 2026 collection already has one
timestamp folder per sample directly under `DCIM_AI`, and each timestamp folder
already contains:

    img0.jpg, img1.jpg, depth_meter.npy, K.txt, rect_left.jpg, rect_right.jpg

This entrypoint reads robot TCP poses from `pose_data.md`, reuses those existing
depth artifacts, writes `robot_pose.json` and `target_pose.json` beside each
sample, then solves the fixed eye-to-hand transform.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import load_config  # noqa: E402
from eye2hand.geometry import (  # noqa: E402
    fairino_pose_to_se3,
    relative_rotation_deg,
    residual_rotation_deg,
    residual_translation_m,
    se3_inv,
)
from eye2hand.handeye import HandEyeSolution, save_solution, solve_all  # noqa: E402
from eye2hand.pipeline import load_pcd_outputs  # noqa: E402
from eye2hand.target_pose import compute_target_pose, save_target_pose  # noqa: E402


POSE_RE = re.compile(
    r"pose\s+(?P<index>\d+)\s+(?P<sample_id>\d+)\s+\[(?P<pose>[^\]]+)\]",
    re.IGNORECASE,
)


@dataclass
class PoseRecord:
    sample_id: str
    pose_index: int
    sample_dir: Path
    pose_mm_deg: np.ndarray
    base_T_gripper: np.ndarray
    cam_T_target: np.ndarray
    target_metrics: dict


def _parse_pose_data(path: Path) -> dict[str, tuple[int, np.ndarray]]:
    out: dict[str, tuple[int, np.ndarray]] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        m = POSE_RE.search(line)
        if not m:
            continue
        sample_id = m.group("sample_id")
        vals = [float(x.strip()) for x in m.group("pose").split(",")]
        if len(vals) != 6:
            raise ValueError(f"{path}:{line_no}: expected 6 pose values, got {len(vals)}")
        if sample_id in out:
            raise ValueError(f"{path}:{line_no}: duplicate sample id {sample_id}")
        out[sample_id] = (int(m.group("index")), np.asarray(vals, dtype=np.float64))
    if not out:
        raise ValueError(f"no pose lines found in {path}")
    return out


def _sample_dirs(data_root: Path) -> list[Path]:
    return sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and re.fullmatch(r"\d+", p.name)
    )


def _write_robot_pose(path: Path, record_id: str, pose_index: int, pose: np.ndarray, source: Path) -> None:
    payload = {
        "sample_id": record_id,
        "pose_index": int(pose_index),
        "pose_mm_deg": pose.tolist(),
        "source": str(source),
    }
    path.write_text(json.dumps(payload, indent=2))


def _target_metrics_payload(result) -> dict:
    return {
        "method": result.method,
        "n_corners_detected": int(result.n_corners_detected),
        "n_corners_used": int(result.n_corners_used),
        "rmse_m": float(result.rmse_m),
        "reprojection_rmse_px": float(result.reprojection_rmse_px),
    }


def _load_or_compute_record(
    sample_dir: Path,
    pose_index: int,
    pose_vec: np.ndarray,
    pose_data_path: Path,
    cfg,
    *,
    force_target: bool,
    debug_vis: bool,
) -> PoseRecord:
    for required in ("img0.jpg", "depth_meter.npy", "K.txt"):
        if not (sample_dir / required).exists():
            raise FileNotFoundError(f"{sample_dir.name}: missing {required}")

    _write_robot_pose(
        sample_dir / "robot_pose.json",
        sample_dir.name,
        pose_index,
        pose_vec,
        pose_data_path,
    )

    target_path = sample_dir / "target_pose.json"
    if target_path.exists() and not force_target:
        target = json.loads(target_path.read_text())
        cam_T_target = np.asarray(target["T_target_cam"], dtype=np.float64)
        metrics = {
            "method": target.get("method", "unknown"),
            "n_corners_detected": int(target.get("n_corners_detected", -1)),
            "n_corners_used": int(target.get("n_corners_used", -1)),
            "rmse_m": float(target.get("rmse_m", float("nan"))),
            "reprojection_rmse_px": float(target.get("reprojection_rmse_px", float("nan"))),
        }
        logger.info("[{}] using existing target_pose.json", sample_dir.name)
    else:
        pcd = load_pcd_outputs(sample_dir, rect_dir=sample_dir)
        result = compute_target_pose(
            pcd,
            cfg.board,
            cfg.target_pose,
            debug_dir=sample_dir if debug_vis else None,
        )
        save_target_pose(result, target_path)
        cam_T_target = result.T_target_cam
        metrics = _target_metrics_payload(result)
        logger.success(
            "[{}] {} | corners {}/{} | reproj_rmse={:.2f}px | rmse_3d={:.4f}m",
            sample_dir.name,
            result.method,
            result.n_corners_used,
            result.n_corners_detected,
            result.reprojection_rmse_px,
            result.rmse_m,
        )

    return PoseRecord(
        sample_id=sample_dir.name,
        pose_index=pose_index,
        sample_dir=sample_dir,
        pose_mm_deg=pose_vec,
        base_T_gripper=fairino_pose_to_se3(pose_vec),
        cam_T_target=cam_T_target,
        target_metrics=metrics,
    )


def _drop_duplicate_robot_poses(records: list[PoseRecord]) -> tuple[list[PoseRecord], list[dict]]:
    kept: list[PoseRecord] = []
    first_by_pose: dict[tuple[float, ...], PoseRecord] = {}
    dropped: list[dict] = []
    for rec in records:
        key = tuple(np.round(rec.pose_mm_deg, decimals=6))
        first = first_by_pose.get(key)
        if first is None:
            first_by_pose[key] = rec
            kept.append(rec)
            continue
        dropped.append({
            "sample_id": rec.sample_id,
            "reason": f"duplicate robot TCP pose already used by {first.sample_id}",
            "duplicate_of": first.sample_id,
        })
    return kept, dropped


def _pose_diversity(records: list[PoseRecord]) -> dict:
    if len(records) < 2:
        return {"min_rotation_deg": None, "median_rotation_deg": None}
    deltas = [
        relative_rotation_deg(records[i].base_T_gripper, records[j].base_T_gripper)
        for i in range(len(records))
        for j in range(i + 1, len(records))
    ]
    return {
        "min_rotation_deg": float(np.min(deltas)),
        "median_rotation_deg": float(np.median(deltas)),
    }


def _per_pose_residuals(records: list[PoseRecord], sol: HandEyeSolution) -> list[dict]:
    # Estimate the fixed target mount once, then predict target->camera for every pose.
    gripper_T_target = se3_inv(records[0].base_T_gripper) @ sol.T_cam2base @ records[0].cam_T_target
    out = []
    for rec in records:
        pred = sol.T_base2cam @ rec.base_T_gripper @ gripper_T_target
        out.append({
            "sample_id": rec.sample_id,
            "translation_error_m": residual_translation_m(pred, rec.cam_T_target),
            "rotation_error_deg": residual_rotation_deg(pred, rec.cam_T_target),
        })
    return out


def _solution_payload(sol: HandEyeSolution) -> dict:
    return {
        "method": sol.method,
        "T_cam2base": sol.T_cam2base.tolist(),
        "T_base2cam": sol.T_base2cam.tolist(),
        "rmse_translation_m": float(sol.rmse_translation_m),
        "rmse_rotation_deg": float(sol.rmse_rotation_deg),
        "n_pairs": int(sol.n_pairs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="path to calib.yaml (default: configs/calib.yaml)")
    ap.add_argument("--data-root", default=None, help="flat collected dataset root (default: config data_root)")
    ap.add_argument("--pose-data", default=None, help="pose_data.md path (default: <data-root>/pose_data.md)")
    ap.add_argument("--out", default=None, help="output dir (default: <data-root>/handeye)")
    ap.add_argument("--exclude-id", action="append", default=[], help="sample id to exclude; repeatable")
    ap.add_argument("--force-target", action="store_true", help="recompute target_pose.json even if it exists")
    ap.add_argument("--no-debug-vis", action="store_true", help="skip target_pose_vis.png overlays")
    ap.add_argument(
        "--drop-duplicate-robot-poses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop later samples with identical TCP pose vectors (default: true)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else cfg.data_root
    pose_data_path = Path(args.pose_data).expanduser().resolve() if args.pose_data else data_root / "pose_data.md"
    out_dir = Path(args.out).expanduser().resolve() if args.out else data_root / "handeye"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        logger.error("data root does not exist: {}", data_root)
        return 1
    if not pose_data_path.exists():
        logger.error("pose data file does not exist: {}", pose_data_path)
        return 1

    pose_map = _parse_pose_data(pose_data_path)
    exclude_ids = set(args.exclude_id)

    records: list[PoseRecord] = []
    failures: list[dict] = []
    for sample_dir in _sample_dirs(data_root):
        if sample_dir.name in exclude_ids:
            logger.warning("[{}] excluded by --exclude-id", sample_dir.name)
            continue
        if sample_dir.name not in pose_map:
            logger.warning("[{}] no TCP pose in {}; skipping", sample_dir.name, pose_data_path)
            continue
        pose_index, pose_vec = pose_map[sample_dir.name]
        try:
            records.append(_load_or_compute_record(
                sample_dir,
                pose_index,
                pose_vec,
                pose_data_path,
                cfg,
                force_target=args.force_target,
                debug_vis=not args.no_debug_vis,
            ))
        except Exception as e:  # noqa: BLE001 -- keep processing the rest of the dataset
            logger.exception("[{}] failed: {}", sample_dir.name, e)
            failures.append({"sample_id": sample_dir.name, "error": repr(e)})

    dropped: list[dict] = []
    solve_records = records
    if args.drop_duplicate_robot_poses:
        solve_records, dropped = _drop_duplicate_robot_poses(records)
        for item in dropped:
            logger.warning("[{}] dropped: {}", item["sample_id"], item["reason"])

    if len(solve_records) < 3:
        logger.error("need at least 3 valid pose pairs; have {}", len(solve_records))
        return 1

    logger.info(
        "solving with {} pose pairs ({} loaded, {} dropped, {} failed)",
        len(solve_records),
        len(records),
        len(dropped),
        len(failures),
    )
    sols = solve_all(
        [r.base_T_gripper for r in solve_records],
        [r.cam_T_target for r in solve_records],
        methods=cfg.handeye.methods,
    )
    if not sols:
        logger.error("all hand-eye methods failed")
        return 1

    best = min(sols, key=lambda s: s.rmse_translation_m)
    save_solution(best, out_dir / cfg.handeye.output_filename)

    summary = {
        "data_root": str(data_root),
        "pose_data": str(pose_data_path),
        "board": asdict(cfg.board),
        "target_pose": asdict(cfg.target_pose),
        "used_pose_ids": [r.sample_id for r in solve_records],
        "dropped_pose_ids": dropped,
        "failures": failures,
        "pose_diversity": _pose_diversity(solve_records),
        "target_pose_metrics": [
            {"sample_id": r.sample_id, **r.target_metrics}
            for r in records
        ],
        "methods": [_solution_payload(s) for s in sols],
        "best_method": best.method,
        "per_pose_residuals": _per_pose_residuals(solve_records, best),
    }
    (out_dir / "handeye_summary.json").write_text(json.dumps(summary, indent=2))

    logger.success(
        "best method: {} | T_rmse={:.4f} m | R_rmse={:.3f} deg | output={}",
        best.method,
        best.rmse_translation_m,
        best.rmse_rotation_deg,
        out_dir / cfg.handeye.output_filename,
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
