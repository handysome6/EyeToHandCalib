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

import time
from pathlib import Path
from typing import Optional

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

        from hik.hik_sync_cam import HikSyncedCameras  # type: ignore[import-not-found]

        self._cams = HikSyncedCameras()
        logger.info("initializing HIK camera group...")
        self._cams.initialize_camera_group()
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
