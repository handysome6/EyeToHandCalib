#!/usr/bin/env python
"""Qt GUI for calibration data collection with live camera preview.

Replaces the terminal-based 01_capture_pose.py with a visual workflow:
  - Live camera feed with ChArUco corner overlay
  - Manual capture via button (or 'S' key)
  - Pose diversity check with visual feedback
  - Status panel showing progress

Requires an x86_64 Python venv (Rosetta) so the HIK MVS SDK loads natively.

    uv venv --python cpython-3.11-macos-x86_64-none && uv sync
    uv run python scripts/01_capture_pose_gui.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eye2hand.config import DEFAULT_CONFIG_PATH, load_config
from eye2hand.geometry import fairino_pose_to_se3, relative_rotation_deg
from eye2hand.target_pose import make_charuco


# ── helpers (reused from 01_capture_pose.py) ──────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _existing_poses(poses_dir: Path) -> tuple[list[np.ndarray], set[str]]:
    out: list[np.ndarray] = []
    ids: set[str] = set()
    if not poses_dir.exists():
        return out, ids
    for pd in sorted(p for p in poses_dir.iterdir() if p.is_dir()):
        rp = pd / "robot_pose.json"
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text())
            pose = np.asarray(data["pose_mm_deg"], dtype=np.float64).reshape(6)
            out.append(fairino_pose_to_se3(pose))
            ids.add(pd.name)
        except Exception:
            continue
    return out, ids


def _too_close(T_new: np.ndarray, existing: list[np.ndarray], min_deg: float) -> tuple[bool, float]:
    if not existing:
        return False, float("inf")
    deltas = [relative_rotation_deg(T_new, T) for T in existing]
    closest = float(min(deltas))
    return closest < min_deg, closest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_pose_data(path: Path, index: int, sample_id: str, pose_raw: np.ndarray) -> None:
    pose = ", ".join(f"{v:.6f}" for v in pose_raw.tolist())
    line = f"pose {index} {sample_id} [{pose}]\n"
    with path.open("a") as f:
        f.write(line)


def _sample_files(pose_dir: Path) -> dict[str, str | None]:
    names = {
        "raw_left": "raw_left.jpg",
        "raw_right": "raw_right.jpg",
        "camera_model": "camera_model.json",
    }
    return {
        key: str(pose_dir / name) if (pose_dir / name).exists() else None
        for key, name in names.items()
    }


# ── Qt GUI ────────────────────────────────────────────────────────────

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _np_to_qpixmap(img_rgb: np.ndarray) -> QPixmap:
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class CaptureWindow(QMainWindow):
    capture_requested = Signal()

    def __init__(
        self,
        cfg,
        cam,
        robot,
        *,
        poses_dir: Path,
        min_rotation_deg: float,
        target_total: int,
        robot_mode: str,
        camera_mode: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.cam = cam
        self.robot = robot
        self.poses_dir = poses_dir
        self.min_rotation_deg = min_rotation_deg
        self.target_total = target_total
        self.robot_mode = robot_mode
        self.camera_mode = camera_mode

        self.existing, self.existing_ids = _existing_poses(poses_dir)
        self.n_new = 0

        self._charuco_board, self._aruco_dict = make_charuco(cfg.board)
        self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)

        self._last_left_rgb: np.ndarray | None = None
        self._last_n_corners = 0
        self._capturing = False

        self._init_ui()

        if hasattr(cam, "frame_signal"):
            cam.frame_signal.connect(self._on_frame)
            cam.start_streaming()

        if hasattr(cam, "_mock_timer"):
            cam._frame_callback = self._on_mock_frame

        self._update_status()

    def _init_ui(self):
        self.setWindowTitle("Eye-to-Hand Calibration Capture")
        self.setMinimumSize(960, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # preview
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        self._preview.setMinimumHeight(480)
        layout.addWidget(self._preview, stretch=1)

        # status
        status_row = QHBoxLayout()
        self._lbl_progress = QLabel()
        self._lbl_charuco = QLabel()
        self._lbl_diversity = QLabel()
        for lbl in (self._lbl_progress, self._lbl_charuco, self._lbl_diversity):
            lbl.setStyleSheet("font-size: 13px; padding: 4px 8px;")
            status_row.addWidget(lbl)
        layout.addLayout(status_row)

        # message
        self._lbl_message = QLabel()
        self._lbl_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_message.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(self._lbl_message)

        # buttons
        btn_row = QHBoxLayout()
        self._btn_capture = QPushButton("Capture  [S]")
        self._btn_capture.setStyleSheet("font-size: 15px; padding: 10px 32px;")
        self._btn_capture.clicked.connect(self._on_capture_click)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_capture)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        QShortcut(QKeySequence("S"), self, self._on_capture_click)
        QShortcut(QKeySequence("Q"), self, self.close)

    def _update_status(self):
        n = len(self.existing)
        self._lbl_progress.setText(f"Poses: {n} / {self.target_total}")
        if n >= self.target_total:
            self._lbl_progress.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #4caf50;")

    def _detect_charuco(self, img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._charuco_detector.detectBoard(gray)
        vis = img_rgb.copy()

        if marker_corners is not None and len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids, cornerColor=(0, 255, 0))
            self._last_n_corners = len(charuco_ids)
        else:
            self._last_n_corners = 0

        total_corners = (self.cfg.board.squares_x - 1) * (self.cfg.board.squares_y - 1)
        if self._last_n_corners > 0:
            self._lbl_charuco.setText(f"ChArUco: {self._last_n_corners}/{total_corners} corners")
            self._lbl_charuco.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #4caf50;")
        else:
            self._lbl_charuco.setText("ChArUco: not detected")
            self._lbl_charuco.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #f44336;")

        return vis, charuco_corners, charuco_ids

    @Slot(object, object)
    def _on_frame(self, frame_type, frame: np.ndarray):
        from hik.hik_sync_cam import FrameType
        if frame_type != FrameType.LEFT:
            return
        self._last_left_rgb = frame.copy()
        vis, _, _ = self._detect_charuco(self._last_left_rgb)
        self._show_preview(vis)

    def _on_mock_frame(self, frame_rgb: np.ndarray):
        self._last_left_rgb = frame_rgb.copy()
        vis, _, _ = self._detect_charuco(self._last_left_rgb)
        self._show_preview(vis)

    def _show_preview(self, img_rgb: np.ndarray):
        preview_w = self._preview.width()
        preview_h = self._preview.height()
        if preview_w < 10 or preview_h < 10:
            return
        h, w = img_rgb.shape[:2]
        scale = min(preview_w / w, preview_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._preview.setPixmap(_np_to_qpixmap(resized))

    @Slot()
    def _on_capture_click(self):
        if self._capturing:
            return
        self._capturing = True
        self._btn_capture.setEnabled(False)
        self._lbl_message.setText("Capturing...")
        self._lbl_message.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px; color: #2196f3;")

        QTimer.singleShot(50, self._do_capture)

    def _do_capture(self):
        try:
            self._capture_impl()
        except Exception as e:
            logger.exception("capture failed: {}", e)
            self._show_message(f"Capture failed: {e}", error=True)
        finally:
            self._capturing = False
            self._btn_capture.setEnabled(True)

    def _capture_impl(self):
        # read robot pose
        try:
            robot_read_start = time.time_ns()
            pose_raw = self.robot.read_pose_raw()
            robot_read_end = time.time_ns()
            T_new = fairino_pose_to_se3(pose_raw)
        except Exception as e:
            self._show_message(f"Robot pose read failed: {e}", error=True)
            return

        # diversity check
        close, closest_deg = _too_close(T_new, self.existing, self.min_rotation_deg)
        if close:
            self._show_message(
                f"Too close to existing pose ({closest_deg:.1f}° < {self.min_rotation_deg:.1f}° min). Move the arm more.",
                error=True,
            )
            self._lbl_diversity.setText(f"Diversity: {closest_deg:.1f}°  ✗")
            self._lbl_diversity.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #f44336;")
            return

        self._lbl_diversity.setText(f"Diversity: {closest_deg:.1f}°")
        self._lbl_diversity.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #4caf50;")

        # capture stereo pair
        captured_at = _now_iso()
        capture_start = time.time_ns()
        if self.cam is not None and hasattr(self.cam, "capture_one"):
            try:
                pose_dir, _, _ = self.cam.capture_one(self.poses_dir, timeout_s=10.0)
            except Exception as e:
                self._show_message(f"Stereo capture failed: {e}", error=True)
                return
        else:
            pose_dir = self.poses_dir / str(int(time.time() * 1e7))
            while pose_dir.name in self.existing_ids or pose_dir.exists():
                time.sleep(0.001)
                pose_dir = self.poses_dir / str(int(time.time() * 1e7))
            pose_dir.mkdir(parents=True, exist_ok=False)
        capture_end = time.time_ns()
        self.existing_ids.add(pose_dir.name)

        # save robot pose
        sample_index = len(self.existing) + 1
        robot_payload = {
            "schema_version": 1,
            "sample_id": pose_dir.name,
            "pose_index": sample_index,
            "captured_at": captured_at,
            "pose_mm_deg": pose_raw.tolist(),
            "T_base_tcp": T_new.tolist(),
            "units": {
                "pose_translation": "mm",
                "pose_rotation": "deg",
                "T_base_tcp_translation": "m",
            },
            "robot": {
                "mode": self.robot_mode,
                "ip": self.cfg.robot.ip if self.robot_mode == "live" else None,
                "read_start_time_ns": robot_read_start,
                "read_end_time_ns": robot_read_end,
            },
        }
        _write_json(pose_dir / "robot_pose.json", robot_payload)

        manifest_payload = {
            "schema_version": 1,
            "sample_id": pose_dir.name,
            "sample_index": sample_index,
            "sample_dir": str(pose_dir),
            "captured_at": captured_at,
            "pose_mm_deg": pose_raw.tolist(),
            "T_base_tcp": T_new.tolist(),
            "closest_existing_rotation_deg": None if np.isinf(closest_deg) else closest_deg,
            "camera_mode": self.camera_mode,
            "robot_mode": self.robot_mode,
            "capture_start_time_ns": capture_start,
            "capture_end_time_ns": capture_end,
            "files": _sample_files(pose_dir),
        }
        _append_jsonl(self.poses_dir / "manifest.jsonl", manifest_payload)
        _append_pose_data(self.poses_dir / "pose_data.md", sample_index, pose_dir.name, pose_raw)

        self.existing.append(T_new)
        self.n_new += 1
        self._update_status()
        self._show_message(f"Captured pose #{len(self.existing)} -> {pose_dir.name}")
        logger.success("captured pose #{} under {}", len(self.existing), pose_dir.name)

    def _show_message(self, text: str, *, error: bool = False):
        color = "#f44336" if error else "#4caf50"
        self._lbl_message.setText(text)
        self._lbl_message.setStyleSheet(f"font-size: 14px; font-weight: bold; padding: 4px; color: {color};")

    def closeEvent(self, event):
        if self.cam is not None:
            self.cam.close()
        close_fn = getattr(self.robot, "close", None)
        if callable(close_fn):
            close_fn()
        logger.info("done -- {} total pose(s), {} new", len(self.existing), self.n_new)
        event.accept()


# ── Mock camera for GUI testing without hardware ──────────────────────


class MockStreamingCapture:
    """Generates synthetic frames with a charuco-like pattern for GUI testing."""

    def __init__(self, board_cfg, width: int = 1280, height: int = 720):
        self._w = width
        self._h = height
        self._i = 0
        self._frame_callback = None
        self._timer = QTimer()
        self._timer.setInterval(66)  # ~15 fps
        self._timer.timeout.connect(self._tick)
        self._board_cfg = board_cfg

        self._charuco_board, _ = make_charuco(board_cfg)
        board_img = self._charuco_board.generateImage((400, 300))
        self._board_pattern = cv2.cvtColor(board_img, cv2.COLOR_GRAY2RGB)
        self._mock_timer = True

    def start_streaming(self) -> None:
        self._timer.start()

    def stop_streaming(self) -> None:
        self._timer.stop()

    def _tick(self):
        frame = np.full((self._h, self._w, 3), 200, dtype=np.uint8)
        bh, bw = self._board_pattern.shape[:2]
        ox = (self._w - bw) // 2 + int(40 * np.sin(self._i * 0.05))
        oy = (self._h - bh) // 2 + int(30 * np.cos(self._i * 0.07))
        ox = max(0, min(ox, self._w - bw))
        oy = max(0, min(oy, self._h - bh))
        frame[oy:oy + bh, ox:ox + bw] = self._board_pattern
        self._i += 1

        if self._frame_callback:
            self._frame_callback(frame)

    def capture_one(self, out_root: Path | str, timeout_s: float = 10.0):
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        project_folder = out_root / str(int(time.time() * 1e7))
        project_folder.mkdir(parents=True, exist_ok=True)

        frame = np.full((self._h, self._w, 3), 200, dtype=np.uint8)
        bh, bw = self._board_pattern.shape[:2]
        ox = (self._w - bw) // 2
        oy = (self._h - bh) // 2
        frame[oy:oy + bh, ox:ox + bw] = self._board_pattern

        left_path = project_folder / "raw_left.jpg"
        right_path = project_folder / "raw_right.jpg"
        cv2.imwrite(str(left_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(right_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return project_folder, left_path, right_path

    def close(self) -> None:
        self._timer.stop()


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Qt GUI calibration capture")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="target folder for captured samples")
    ap.add_argument("--min-rotation-deg", type=float, default=None)
    ap.add_argument(
        "--robot-mode",
        choices=["live", "mock", "manual"],
        default="live",
    )
    ap.add_argument(
        "--camera-mode",
        choices=["live", "mock"],
        default="live",
    )
    ap.add_argument("--dry-run", action="store_true", help="alias for --robot-mode mock --camera-mode mock")
    ap.add_argument("--mock-robot", action="store_true")
    ap.add_argument("--mock-camera", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    camera_mode = args.camera_mode
    robot_mode = args.robot_mode
    if args.mock_camera:
        camera_mode = "mock"
    if args.mock_robot:
        robot_mode = "mock"
    if args.dry_run:
        camera_mode = "mock"
        robot_mode = "mock"

    poses_dir = _resolve_path(args.out) if args.out else cfg.poses_dir
    poses_dir.mkdir(parents=True, exist_ok=True)

    min_rotation_deg = (
        float(args.min_rotation_deg)
        if args.min_rotation_deg is not None
        else float(cfg.capture.min_pose_diversity_deg)
    )
    existing, _ = _existing_poses(poses_dir)
    target_total = cfg.capture.target_poses

    app = QApplication(sys.argv)

    # robot
    if robot_mode == "live":
        from eye2hand.robot import RobotClient
        robot = RobotClient(cfg.robot.ip)
    elif robot_mode == "mock":
        from eye2hand.robot import MockRobotClient
        robot = MockRobotClient(start_index=len(existing))
    else:
        from scripts import _manual_robot_not_supported
        raise SystemExit("manual robot mode not supported in GUI; use terminal script instead")

    # camera
    if camera_mode == "live":
        from eye2hand.camera import DirectStereoCapture
        cam = DirectStereoCapture(
            cfg.jetson_reborn_path,
            exposure_us=cfg.camera.exposure_us,
            gain_db=cfg.camera.gain_db,
        )
    else:
        cam = MockStreamingCapture(cfg.board)

    logger.info("starting GUI capture -- {} existing pose(s) in {}", len(existing), poses_dir)
    logger.info("modes: robot={}, camera={}", robot_mode, camera_mode)

    win = CaptureWindow(
        cfg,
        cam,
        robot,
        poses_dir=poses_dir,
        min_rotation_deg=min_rotation_deg,
        target_total=target_total,
        robot_mode=robot_mode,
        camera_mode=camera_mode,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
