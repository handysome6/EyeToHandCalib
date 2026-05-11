"""Headless wrapper around the existing HIK stereo camera stack.

Spins up a QCoreApplication-backed `HikSyncedCameras`, sends one software
trigger per requested capture, and writes the resulting raw_left.jpg /
raw_right.jpg / camera_model.json into a fresh project folder under
data_root (HIK's `save_frames` already does the file/copy work for us).

Two patterns are supported:
    with StereoCapture() as cam:
        project_folder, raw_left, raw_right = cam.capture_one(out_root)
        ...

This needs the JetsonReborn_rebar `hik` module on sys.path -- call
`prepare_paths()` first.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger


class StereoCapture:
    """Owns a HikSyncedCameras instance and a QCoreApplication for event pumping."""

    def __init__(self):
        # PySide6 imports deferred so that `import eye2hand.camera` doesn't crash
        # in test environments without Qt.
        from PySide6.QtCore import QCoreApplication
        import sys

        # HikSyncedCameras emits cross-thread signals, so we need a Qt event
        # loop. A QCoreApplication is enough (no GUI required).
        self._app = QCoreApplication.instance() or QCoreApplication(sys.argv)

        try:
            from hik.hik_sync_cam import HikSyncedCameras  # type: ignore[import-not-found]
        except SystemExit as e:
            raise RuntimeError(
                "HIK SDK import aborted. On macOS, verify MVCAM_SDK_PATH "
                "points to the old MVS SDK root (default: /Library/MVS_SDK) "
                "and run an x86_64 Python if the SDK dylibs are x86_64-only."
            ) from e

        self._cams = HikSyncedCameras()
        logger.info("initializing HIK camera group...")
        try:
            self._cams.initialize_camera_group()
        except SystemExit as e:
            raise RuntimeError("HIK camera initialization aborted") from e
        logger.info("HIK camera group ready")

    def __enter__(self) -> "StereoCapture":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _pump(self, ms: int = 5) -> None:
        """Pump pending Qt events for `ms` ms."""
        from PySide6.QtCore import QCoreApplication, QEventLoop

        self._app.processEvents(QEventLoop.AllEvents, ms)

    def capture_one(self, out_root: Path | str, timeout_s: float = 10.0):
        """Trigger one synced stereo pair and save it.

        Returns: (project_folder: Path, raw_left: Path, raw_right: Path)
        """
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        # Reset frame buffers, send software trigger, wait for both frames.
        self._cams.left_frame = None
        self._cams.right_frame = None
        self._cams.capture_dual_camera()

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._pump(10)
            if self._cams.left_frame is not None and self._cams.right_frame is not None:
                break
        else:
            raise TimeoutError(f"stereo capture timed out after {timeout_s}s")

        # save_frames creates a timestamped subfolder under out_root and copies
        # camera_model.json from GLOBAL_CAM_PATH if it exists -- exactly the
        # broker rectify input layout we want.
        project_folder, left_path, right_path = self._cams.save_frames(out_root)
        return Path(project_folder), Path(left_path), Path(right_path)

    def close(self) -> None:
        try:
            self._cams._deinit_cameras()
        except Exception as e:
            logger.warning("camera deinit raised: {}", e)


class MockStereoCapture:
    """Writes deterministic placeholder stereo files for dry-run testing."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = int(width)
        self.height = int(height)
        self._i = 0

    def __enter__(self) -> "MockStereoCapture":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def capture_one(self, out_root: Path | str, timeout_s: float = 10.0):
        del timeout_s
        import cv2
        import numpy as np

        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        project_folder = self._new_project_folder(out_root)

        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        left = np.dstack([
            np.tile(x, (self.height, 1)),
            np.tile(y, (1, self.width)),
            np.full((self.height, self.width), 80 + (self._i * 13) % 120, dtype=np.uint8),
        ])
        right = np.roll(left, shift=8, axis=1)
        cv2.putText(left, f"mock left {self._i}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(right, f"mock right {self._i}", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        left_path = project_folder / "raw_left.jpg"
        right_path = project_folder / "raw_right.jpg"
        cv2.imwrite(str(left_path), left)
        cv2.imwrite(str(right_path), right)

        (project_folder / "camera_model.json").write_text(json.dumps({
            "mock": True,
            "width": self.width,
            "height": self.height,
            "note": "Placeholder only; not valid for calibration.",
        }, indent=2))
        self._i += 1
        return project_folder, left_path, right_path

    @staticmethod
    def _new_project_folder(out_root: Path) -> Path:
        while True:
            project_folder = out_root / str(int(time.time() * 1e7))
            try:
                project_folder.mkdir(parents=True, exist_ok=False)
                return project_folder
            except FileExistsError:
                time.sleep(0.001)

    def close(self) -> None:
        return None
