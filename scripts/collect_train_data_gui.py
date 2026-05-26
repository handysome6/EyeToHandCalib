#!/usr/bin/env python
"""Qt GUI for training data collection with decoupled image capture and pose recording.

Workflow per sample:
  1. Click "Capture Image" to save one stereo pair into a new timestamped folder.
  2. Click "Record Pose" up to 5 times — each records the robot arm's current TCP
     pose and appends it to tcp_poses.json inside the same folder.
  3. Click "Capture Image" again to start the next sample.

Device reconnection and live stereo preview are carried over from 01_capture_pose_gui.py.

    uv venv --python cpython-3.11-macos-x86_64-none && uv sync
    uv run python scripts/collect_train_data_gui.py
    uv run python scripts/collect_train_data_gui.py --dry-run   # no hardware
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
from eye2hand.target_pose import make_charuco

MAX_POSES_PER_CAPTURE = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


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


class TrainDataWindow(QMainWindow):
    capture_requested = Signal()

    def __init__(
        self,
        cfg,
        *,
        out_dir: Path,
        robot_mode: str,
        camera_mode: str,
    ):
        super().__init__()
        self.cfg = cfg
        self.cam = None
        self.robot = None
        self.out_dir = out_dir
        self.robot_mode = robot_mode
        self.camera_mode = camera_mode

        self._last_left_rgb: np.ndarray | None = None
        self._last_right_rgb: np.ndarray | None = None
        self._capturing = False

        self._current_sample_dir: Path | None = None
        self._current_poses: list[dict] = []
        self._total_samples = 0

        self._init_ui()
        self._update_connection_ui()
        self._update_button_states()

    def _init_ui(self):
        self.setWindowTitle("Training Data Collection")
        self.setMinimumSize(1280, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # stereo preview
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

        # status row
        status_row = QHBoxLayout()
        self._lbl_samples = QLabel()
        self._lbl_poses = QLabel()
        for lbl in (self._lbl_samples, self._lbl_poses):
            lbl.setStyleSheet("font-size: 13px; padding: 4px 8px;")
            status_row.addWidget(lbl)
        status_row.addStretch()
        layout.addLayout(status_row)

        # message
        self._lbl_message = QLabel()
        self._lbl_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_message.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(self._lbl_message)

        # buttons
        btn_row = QHBoxLayout()

        self._btn_capture = QPushButton("Capture Image  [S]")
        self._btn_capture.setStyleSheet("font-size: 15px; padding: 10px 32px;")
        self._btn_capture.clicked.connect(self._on_capture_click)

        self._btn_record = QPushButton("Record Pose  [R]")
        self._btn_record.setStyleSheet("font-size: 15px; padding: 10px 32px;")
        self._btn_record.clicked.connect(self._on_record_click)

        btn_row.addStretch()
        btn_row.addWidget(self._btn_capture)
        btn_row.addWidget(self._btn_record)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        QShortcut(QKeySequence("S"), self, self._on_capture_click)
        QShortcut(QKeySequence("R"), self, self._on_record_click)
        QShortcut(QKeySequence("Q"), self, self.close)

    def _update_status(self):
        self._lbl_samples.setText(f"Samples: {self._total_samples}")
        n = len(self._current_poses)
        if self._current_sample_dir is not None:
            self._lbl_poses.setText(f"Poses: {n} / {MAX_POSES_PER_CAPTURE}")
            color = "#4caf50" if n == MAX_POSES_PER_CAPTURE else "#2196f3"
            self._lbl_poses.setStyleSheet(f"font-size: 13px; padding: 4px 8px; color: {color};")
        else:
            self._lbl_poses.setText("")

    def _update_button_states(self):
        devices_ready = self.robot is not None and self.cam is not None
        has_sample = self._current_sample_dir is not None
        poses_full = len(self._current_poses) >= MAX_POSES_PER_CAPTURE

        self._btn_capture.setEnabled(
            devices_ready and not self._capturing and (not has_sample or poses_full)
        )
        self._btn_record.setEnabled(
            devices_ready and not self._capturing and has_sample and not poses_full
        )
        self._update_status()

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

    def _start_camera_streaming(self):
        if self.cam is None:
            return
        if hasattr(self.cam, "frame_signal"):
            self.cam.frame_signal.connect(self._on_frame)
            self.cam.start_streaming()
        if hasattr(self.cam, "_mock_timer"):
            self.cam._frame_callback = self._on_mock_frame
            self.cam.start_streaming()

    # ── device connection ──

    @Slot()
    def _on_connect_robot(self):
        self._btn_connect_robot.setEnabled(False)
        self._lbl_robot_status.setText("Connecting...")
        self._lbl_robot_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #ff9800;")
        QTimer.singleShot(50, self._do_connect_robot)

    def _do_connect_robot(self):
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
                self.robot = MockRobotClient()
            self._show_message("Robot connected")
            logger.info("robot connected (mode={})", self.robot_mode)
        except Exception as e:
            self.robot = None
            self._show_message(f"Robot connection failed: {e}", error=True)
            logger.error("robot connection failed: {}", e)
        finally:
            self._btn_connect_robot.setEnabled(True)
            self._update_connection_ui()
            self._update_button_states()

    @Slot()
    def _on_connect_camera(self):
        self._btn_connect_camera.setEnabled(False)
        self._lbl_camera_status.setText("Connecting...")
        self._lbl_camera_status.setStyleSheet("font-size: 13px; padding: 4px 8px; color: #ff9800;")
        QTimer.singleShot(50, self._do_connect_camera)

    def _do_connect_camera(self):
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
            self._update_button_states()

    # ── frame display ──

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

    # ── capture image ──

    @Slot()
    def _on_capture_click(self):
        if self._capturing:
            return
        if self.robot is None or self.cam is None:
            self._show_message("Connect robot and camera first", error=True)
            return
        has_sample = self._current_sample_dir is not None
        poses_full = len(self._current_poses) >= MAX_POSES_PER_CAPTURE
        if has_sample and not poses_full:
            return
        self._capturing = True
        self._update_button_states()
        self._lbl_message.setText("Capturing image...")
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
            self._update_button_states()

    def _capture_impl(self):
        if self.cam is not None and hasattr(self.cam, "capture_one"):
            try:
                sample_dir, _, _ = self.cam.capture_one(self.out_dir, timeout_s=10.0)
            except Exception as e:
                self._show_message(f"Stereo capture failed: {e}", error=True)
                return
        else:
            sample_dir = self.out_dir / str(int(time.time() * 1e7))
            while sample_dir.exists():
                time.sleep(0.001)
                sample_dir = self.out_dir / str(int(time.time() * 1e7))
            sample_dir.mkdir(parents=True, exist_ok=False)

        self._current_sample_dir = sample_dir
        self._current_poses = []
        self._total_samples += 1

        self._show_message(
            f"Sample #{self._total_samples} captured -> {sample_dir.name}. Now record poses."
        )
        logger.success("captured stereo pair #{} under {}", self._total_samples, sample_dir.name)

    # ── record pose ──

    @Slot()
    def _on_record_click(self):
        if self._capturing:
            return
        if self.robot is None:
            self._show_message("Connect robot first", error=True)
            return
        if self._current_sample_dir is None:
            self._show_message("Capture an image first", error=True)
            return
        if len(self._current_poses) >= MAX_POSES_PER_CAPTURE:
            return
        self._capturing = True
        self._update_button_states()
        QTimer.singleShot(50, self._do_record)

    def _do_record(self):
        try:
            self._record_impl()
        except Exception as e:
            logger.exception("pose record failed: {}", e)
            self._show_message(f"Pose record failed: {e}", error=True)
        finally:
            self._capturing = False
            self._update_button_states()

    def _record_impl(self):
        try:
            pose_raw = self.robot.read_pose_raw()
        except Exception as e:
            self._show_message(f"Robot pose read failed: {e}", error=True)
            return

        entry = {
            "index": len(self._current_poses),
            "pose_mm_deg": pose_raw.tolist(),
            "captured_at": _now_iso(),
        }
        self._current_poses.append(entry)
        self._save_poses()

        n = len(self._current_poses)
        x, y, z, rx, ry, rz = pose_raw.tolist()
        self._show_message(
            f"Pose {n}/{MAX_POSES_PER_CAPTURE} recorded: "
            f"[{x:.1f}, {y:.1f}, {z:.1f}, {rx:.1f}, {ry:.1f}, {rz:.1f}]"
        )
        logger.info(
            "recorded pose {}/{} in {}",
            n, MAX_POSES_PER_CAPTURE, self._current_sample_dir.name,
        )

    def _save_poses(self):
        payload = {
            "schema_version": 1,
            "sample_id": self._current_sample_dir.name,
            "robot_ip": self.cfg.robot.ip if self.robot_mode == "live" else None,
            "mode": self.robot_mode,
            "total_poses": len(self._current_poses),
            "updated_at": _now_iso(),
            "units": {
                "translation": "mm",
                "rotation": "deg (extrinsic XYZ Euler)",
            },
            "poses": self._current_poses,
        }
        _write_json(self._current_sample_dir / "tcp_poses.json", payload)

    # ── helpers ──

    def _show_message(self, text: str, *, error: bool = False):
        color = "#f44336" if error else "#4caf50"
        self._lbl_message.setText(text)
        self._lbl_message.setStyleSheet(
            f"font-size: 14px; font-weight: bold; padding: 4px; color: {color};"
        )

    def closeEvent(self, event):
        if self.cam is not None:
            self.cam.close()
        close_fn = getattr(self.robot, "close", None)
        if callable(close_fn):
            close_fn()
        logger.info("done -- {} sample(s) collected", self._total_samples)
        event.accept()


# ── Mock camera (copied from 01_capture_pose_gui.py) ────────────────


class MockStreamingCapture:
    def __init__(self, board_cfg, width: int = 1280, height: int = 720):
        self._w = width
        self._h = height
        self._i = 0
        self._frame_callback = None
        self._timer = QTimer()
        self._timer.setInterval(66)
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
    ap = argparse.ArgumentParser(description="Training data collection GUI")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="output folder (default: data/train)")
    ap.add_argument(
        "--robot-mode",
        choices=["live", "mock"],
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

    out_dir = _resolve_path(args.out) if args.out else (REPO_ROOT / "data" / "train")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = QApplication(sys.argv)

    logger.info("starting training data collection GUI -- output: {}", out_dir)
    logger.info("modes: robot={}, camera={}", robot_mode, camera_mode)

    win = TrainDataWindow(
        cfg,
        out_dir=out_dir,
        robot_mode=robot_mode,
        camera_mode=camera_mode,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
