"""Stereo capture adapters for HIK cameras.

StereoCapture calls `hik/hik_capture_cli.py` inside the JetsonReborn_rebar
checkout as a subprocess.  This decouples the x86_64 HIK MVS SDK from the
host process — the caller can run on native ARM64 Python while the capture
CLI runs under Rosetta with an x86_64 venv.

    cam = StereoCapture(jetson_reborn_path)
    project_folder, raw_left, raw_right = cam.capture_one(out_root)
    cam.close()
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from loguru import logger


class StereoCapture:
    """Subprocess-based stereo capture using JetsonReborn's hik_capture_cli."""

    def __init__(self, jetson_reborn_path: Path | str):
        self._jr = Path(jetson_reborn_path).expanduser().resolve()
        self._python = self._jr / ".venv" / "bin" / "python"
        if not self._python.exists():
            raise FileNotFoundError(
                f"JetsonReborn x86_64 venv not found at {self._python}. "
                f"Create it with: cd {self._jr} && uv venv --python cpython-3.11-macos-x86_64-none && uv sync"
            )
        cli = self._jr / "hik" / "hik_capture_cli.py"
        if not cli.exists():
            raise FileNotFoundError(f"hik_capture_cli.py not found at {cli}")

    def __enter__(self) -> "StereoCapture":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def capture_one(self, out_root: Path | str, timeout_s: float = 10.0):
        """Trigger one synced stereo capture via subprocess.

        Returns: (project_folder: Path, raw_left: Path, raw_right: Path)
        """
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self._python), "-m", "hik.hik_capture_cli",
            "--out", str(out_root),
            "--timeout", str(timeout_s),
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(self._jr),
            capture_output=True,
            text=True,
            timeout=timeout_s + 30,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(f"hik_capture_cli failed (rc={proc.returncode}): {stderr}")

        # The HIK SDK prints debug text to stdout before the JSON result.
        # Extract the last JSON object from the output.
        stdout = proc.stdout.strip()
        json_start = stdout.rfind("{")
        if json_start == -1:
            raise RuntimeError(f"no JSON in hik_capture_cli output: {stdout[:200]}")
        result = json.loads(stdout[json_start:])
        return (
            Path(result["project_folder"]),
            Path(result["raw_left"]),
            Path(result["raw_right"]),
        )

    def close(self) -> None:
        pass


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
