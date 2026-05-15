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
        *,
        poses_dir: Path,
        min_rotation_deg: float,
        target_total: int,
        robot_mode: str,
        camera_mode: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.cam = None
        self.robot = None
        self.poses_dir = poses_dir
        self.min_rotation_deg = min_rotation_deg
        self.target_total = target_total
        self.robot_mode = robot_mode
        self.camera_mode = camera_mode

        self.existing, self.existing_ids = _existing_poses(poses_dir)
        self.n_new = 0

        self._last_left_rgb: np.ndarray | None = None
        self._last_right_rgb: np.ndarray | None = None
        self._capturing = False

        self._init_ui()
        self._update_connection_ui()

    def _init_ui(self):
        self.setWindowTitle("Eye-to-Hand Calibration Capture")
        self.setMinimumSize(1280, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # stereo preview (left + right side by side)
        preview_row = QHBoxLayout()
        preview_row.setSpacing(6)

        self._preview_left = QLabel()
        self._preview_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_left.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        self._preview_left.setMinimumHeight(400)

        self._preview_right = QLabel()
        self._preview_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_right.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        self._preview_right.setMinimumHeight(400)

        lbl_left = QLabel("Left")
        lbl_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_left.setStyleSheet("font-size: 11px; color: #888;")
        lbl_right = QLabel("Right")
        lbl_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_right.setStyleSheet("font-size: 11px; color: #888;")

        left_col = QVBoxLayout()
        left_col.addWidget(self._preview_left, stretch=1)
        left_col.addWidget(lbl_left)

        right_col = QVBoxLayout()
        right_col.addWidget(self._preview_right, stretch=1)
        right_col.addWidget(lbl_right)

        preview_row.addLayout(left_col, stretch=1)
        preview_row.addLayout(right_col, stretch=1)
        layout.addLayout(preview_row, stretch=1)

        # connection row
        conn_row = QHBoxLayout()
        self._btn_connect_robot = QPushButton("Connect Robot")
        self._btn_connect_robot.setStyleSheet("font-size: 13px; padding: 6px 16px;")
        self._btn_connect_robot.clicked.connect(self._on_connect_robot)
        self._lbl_robot_status = QLabel()
        self._lbl_robot_status.setStyleSheet("font-size: 13px; padding: 4px 8px;")

        self._btn_connect_camera = QPushButton("Connect Camera")
        self._btn_connect_camera.setStyleSheet("font-size: 13px; padding: 6px 16px;")
        self._btn_connect_camera.clicked.connect(self._on_connect_camera)
        self._lbl_camera_status = QLabel()
        self._lbl_camera_status.setStyleSheet("font-size: 13px; padding: 4px 8px;")

        conn_row.addWidget(self._btn_connect_robot)
        conn_row.addWidget(self._lbl_robot_status)
        conn_row.addStretch()
        conn_row.addWidget(self._btn_connect_camera)
        conn_row.addWidget(self._lbl_camera_status)
        layout.addLayout(conn_row)

        # status
        status_row = QHBoxLayout()
        self._lbl_progress = QLabel()
        self._lbl_diversity = QLabel()
        for lbl in (self._lbl_progress, self._lbl_diversity):
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

    def _update_connection_ui(self):
        if self.robot is not None:
            self._lbl_robot_status.setText("Connected")
            self._lbl_robot_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #4caf50;")
            self._btn_connect_robot.setText("Reconnect Robot")
        else:
            self._lbl_robot_status.setText("Disconnected")
            self._lbl_robot_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #f44336;")
            self._btn_connect_robot.setText("Connect Robot")

        if self.cam is not None:
            self._lbl_camera_status.setText("Connected")
            self._lbl_camera_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #4caf50;")
            self._btn_connect_camera.setText("Reconnect Camera")
        else:
            self._lbl_camera_status.setText("Disconnected")
            self._lbl_camera_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #f44336;")
            self._btn_connect_camera.setText("Connect Camera")

        can_capture = self.robot is not None and self.cam is not None
        self._btn_capture.setEnabled(can_capture and not self._capturing)

    def _start_camera_streaming(self):
        if self.cam is None:
            return
        if hasattr(self.cam, "frame_signal"):
            self.cam.frame_signal.connect(self._on_frame)
            self.cam.start_streaming()
        if hasattr(self.cam, "_mock_timer"):
            self.cam._frame_callback = self._on_mock_frame
            self.cam.start_streaming()

    @Slot()
    def _on_connect_robot(self):
        self._btn_connect_robot.setEnabled(False)
        self._lbl_robot_status.setText("Connecting...")
        self._lbl_robot_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #ff9800;")
        QTimer.singleShot(50, self._do_connect_robot)

    def _do_connect_robot(self):
        # close existing
        if self.robot is not None:
            close_fn = getattr(self.robot, "close", None)
            if callable(close_fn):
                close_fn()
            self.robot = None

        try:
            if self.robot_mode == "live":
                from eye2hand.robot import RobotClient
                client = RobotClient(self.cfg.robot.ip)
                client.read_pose_raw()
                self.robot = client
            elif self.robot_mode == "mock":
                from eye2hand.robot import MockRobotClient
                self.robot = MockRobotClient(start_index=len(self.existing))
            self._show_message("Robot connected")
            logger.info("robot connected (mode={})", self.robot_mode)
        except Exception as e:
            self.robot = None
            self._show_message(f"Robot connection failed: {e}", error=True)
            logger.error("robot connection failed: {}", e)
        finally:
            self._btn_connect_robot.setEnabled(True)
            self._update_connection_ui()

    @Slot()
    def _on_connect_camera(self):
        self._btn_connect_camera.setEnabled(False)
        self._lbl_camera_status.setText("Connecting...")
        self._lbl_camera_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #ff9800;")
        QTimer.singleShot(50, self._do_connect_camera)

    def _do_connect_camera(self):
        # close existing
        if self.cam is not None:
            self.cam.close()
            self.cam = None

        try:
            if self.camera_mode == "live":
                from eye2hand.camera import DirectStereoCapture
                self.cam = DirectStereoCapture(
                    self.cfg.jetson_reborn_path,
                    exposure_us=self.cfg.camera.exposure_us,
                    gain_db=self.cfg.camera.gain_db,
                )
            else:
                self.cam = MockStreamingCapture(self.cfg.board)
            self._start_camera_streaming()
            self._show_message("Camera connected")
            logger.info("camera connected (mode={})", self.camera_mode)
        except Exception as e:
            self.cam = None
            self._show_message(f"Camera connection failed: {e}", error=True)
            logger.error("camera connection failed: {}", e)
        finally:
            self._btn_connect_camera.setEnabled(True)
            self._update_connection_ui()

    @Slot(object, object)
    def _on_frame(self, frame_type, frame: np.ndarray):
        from hik.hik_sync_cam import FrameType
        if frame_type == FrameType.LEFT:
            self._last_left_rgb = frame.copy()
            self._show_on_label(self._preview_left, self._last_left_rgb)
        elif frame_type == FrameType.RIGHT:
            self._last_right_rgb = frame.copy()
            self._show_on_label(self._preview_right, self._last_right_rgb)

    def _on_mock_frame(self, frame_rgb: np.ndarray):
        self._last_left_rgb = frame_rgb.copy()
        self._show_on_label(self._preview_left, frame_rgb)
        self._show_on_label(self._preview_right, frame_rgb)

    def _show_on_label(self, label: QLabel, img_rgb: np.ndarray):
        lw = label.width()
        lh = label.height()
        if lw < 10 or lh < 10:
            return
        h, w = img_rgb.shape[:2]
        scale = min(lw / w, lh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label.setPixmap(_np_to_qpixmap(resized))

    @Slot()
    def _on_capture_click(self):
        if self._capturing:
            return
        if self.robot is None or self.cam is None:
            self._show_message("Connect robot and camera first", error=True)
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
            self._update_connection_ui()

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

    if robot_mode == "manual":
        raise SystemExit("manual robot mode not supported in GUI; use terminal script instead")

    app = QApplication(sys.argv)

    logger.info("starting GUI capture -- {} existing pose(s) in {}", len(existing), poses_dir)
    logger.info("modes: robot={}, camera={}", robot_mode, camera_mode)

    win = CaptureWindow(
        cfg,
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
