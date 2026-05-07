"""Thin wrapper around the upstream broker pipelines.

Sequence per pose:
    1. raw L/R + camera_model.json  --rectification_pipeline-->  rect_left.jpg, rect_right.jpg
    2. rect L/R + camera_model.json --pcd_generation_pipeline--> img0.jpg (scaled),
                                                                  depth_meter.npy,
                                                                  K.txt (full-res),
                                                                  cloud.ply, vis.png

The pcd server may downscale (scale<1) before computing depth, so depth_meter and
img0 live at a possibly-smaller resolution than rect_left, while the K.txt carries
the FULL-resolution intrinsics. `load_pcd_outputs` returns a `PcdOutputs` whose
`K_scaled` field is K rescaled to the depth-map resolution, ready to back-project
corners detected on `img0_path`.

`prepare_paths()` MUST be called before importing this module's pipeline functions
in an entrypoint, since they re-export broker bits at call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from loguru import logger


@dataclass
class PcdOutputs:
    """Loaded artifacts ready for downstream geometry."""
    img0_path: Path           # scaled, rectified left image (matches depth_meter)
    img1_path: Path           # scaled, rectified right image
    depth_meter: np.ndarray   # (H', W') float, metric depth on img0
    K_full: np.ndarray        # 3x3 full-resolution intrinsics from K.txt
    K_scaled: np.ndarray      # 3x3 intrinsics scaled to img0 resolution
    baseline_m: float         # stereo baseline (meters)
    scale: float              # img0_width / rect_left_width  (== K_scaled[0,0]/K_full[0,0])

    @property
    def H(self) -> int:
        return self.depth_meter.shape[0]

    @property
    def W(self) -> int:
        return self.depth_meter.shape[1]


def run_rectification(
    raw_left: Path | str,
    raw_right: Path | str,
    camera_model_path: Path | str,
    out_dir: Path | str,
):
    """Call broker's rectification_pipeline. Returns broker's RectificationResult."""
    from broker.imgproc import rectification_pipeline  # type: ignore[import-not-found]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("rectifying -> {}", out_dir)
    return rectification_pipeline(
        raw_left_path=Path(raw_left),
        raw_right_path=Path(raw_right),
        camera_model_path=Path(camera_model_path),
        output_dir=out_dir,
    )


def run_pcd(
    rect_left: Path | str,
    rect_right: Path | str,
    camera_model_path: Path | str,
    out_dir: Path | str,
    pcd_config: Optional[Union[str, Path, dict]] = None,
):
    """Call broker's pcd_generation_pipeline. Returns broker's PCDGenerationResult.

    `pcd_config` is either a path to a yaml understood by broker.pcd_generate.config
    (with primary_server/fallback_server.server_url) or None to fall back to the
    PCD_PRIMARY_SERVER_URL env vars.
    """
    from broker.pcd_generate import pcd_generation_pipeline  # type: ignore[import-not-found]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("generating pcd -> {}", out_dir)
    return pcd_generation_pipeline(
        rectified_left_path=Path(rect_left),
        rectified_right_path=Path(rect_right),
        camera_model_path=Path(camera_model_path),
        output_dir=out_dir,
        pcd_config=pcd_config,
    )


def parse_K_txt(k_txt_path: Path | str) -> tuple[np.ndarray, float]:
    """Parse broker K.txt: line 1 is 9 floats (row-major K), line 2 is baseline (m)."""
    with open(k_txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"unexpected K.txt format: {k_txt_path}")
    vals = [float(x) for x in lines[0].split()]
    if len(vals) != 9:
        raise ValueError(f"K.txt first line must have 9 floats, got {len(vals)}")
    K = np.array(vals, dtype=np.float64).reshape(3, 3)
    baseline_m = float(lines[1])
    return K, baseline_m


def _scale_K(K: np.ndarray, scale: float) -> np.ndarray:
    Ks = K.copy()
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


def load_pcd_outputs(pcd_dir: Path | str, rect_dir: Optional[Path | str] = None) -> PcdOutputs:
    """Load broker outputs from a pose's pcd folder into geometry-ready arrays.

    `rect_dir` is used to recover the unscaled width for the intrinsic scale factor;
    if not provided we fall back to inferring scale from the K principal point
    (which is roughly image_center for a rectified pair).
    """
    pcd_dir = Path(pcd_dir)
    img0 = pcd_dir / "img0.jpg"
    if not img0.exists():
        # legacy png fallback (mirrors broker.PCDGenerationResult.from_result_folder)
        img0 = pcd_dir / "img0.png"
    img1 = pcd_dir / "img1.jpg"
    if not img1.exists():
        img1 = pcd_dir / "img1.png"
    depth = np.load(pcd_dir / "depth_meter.npy")
    K, baseline = parse_K_txt(pcd_dir / "K.txt")

    img0_im = cv2.imread(str(img0), cv2.IMREAD_UNCHANGED)
    if img0_im is None:
        raise FileNotFoundError(f"could not read {img0}")
    H_d, W_d = depth.shape[:2]
    if img0_im.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"img0 {img0_im.shape[:2]} != depth {depth.shape[:2]}; broker output mismatch"
        )

    if rect_dir is not None:
        rect_dir = Path(rect_dir)
        rect_left = rect_dir / "rect_left.jpg"
        if not rect_left.exists():
            # legacy A_*.jpg
            cands = sorted(rect_dir.glob("A_*.*"))
            rect_left = cands[0] if cands else None
        if rect_left is not None and rect_left.exists():
            rect_im = cv2.imread(str(rect_left), cv2.IMREAD_UNCHANGED)
            if rect_im is not None:
                scale = W_d / float(rect_im.shape[1])
            else:
                scale = (2.0 * K[0, 2]) / float(W_d)
                scale = 1.0 / scale
        else:
            scale = float(W_d) / (2.0 * K[0, 2])
    else:
        # principal-point heuristic: cx ≈ W/2 at full res, so scale ≈ W_depth / (2 * cx_full)
        scale = float(W_d) / (2.0 * K[0, 2])

    K_scaled = _scale_K(K, scale)

    return PcdOutputs(
        img0_path=img0,
        img1_path=img1,
        depth_meter=depth.astype(np.float64, copy=False),
        K_full=K,
        K_scaled=K_scaled,
        baseline_m=baseline,
        scale=scale,
    )
