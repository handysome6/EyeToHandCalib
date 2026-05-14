# EyeToHandCalib

Eye-to-hand calibration for a HIK stereo rig + Fairino arm, leveraging the existing
`broker` rectify + point-cloud pipeline (`JetsonReborn_rebar/broker/`) for depth and
the existing `hik` capture stack (`JetsonReborn_rebar/hik/`) for synced stereo grabs.

## Approach

A ChArUco board is rigidly mounted on the end-effector. For each pose:

1. Read `T_base_tcp` from the Fairino controller.
2. Trigger the synced stereo and save raw L/R images.
3. Run broker rectification + pcd generation → metric depth on rectified left.
4. Detect ChArUco corners on rectified left, lift each corner to a 3D point in the
   camera frame using the metric depth map, and solve `T_target_cam` by rigid
   3D-3D registration (Umeyama). Falls back to `solvePnP` if too few corners
   have valid depth.

Once enough poses are captured we feed `(T_base_tcp, T_target_cam)` pairs to
`cv2.calibrateHandEye` (eye-to-hand convention) to solve the fixed camera pose
in the robot base frame. The output JSON keeps both directions:
`T_cam_base` maps camera-frame points into the robot base frame, and
`T_base_cam` is its inverse.

## Layout

```
configs/calib.yaml              # robot IP, board, paths, thresholds
src/eye2hand/                   # library code
scripts/01_capture_pose.py      # target-folder collection loop (live/mock/manual)
scripts/02_process_dataset.py   # rectify + pcd + target-pose for every captured folder
scripts/03_solve_handeye.py     # AX=XB, write T_cam_base.json
scripts/04_calibrate_collected_dataset.py
                                # one-shot solve for flat /Users/andyliu/DCIM_AI data
data/poses/<id>/                # per-pose data (gitignored)
```

## Setup

### 1. This repository

```sh
uv sync
```

Edit `configs/calib.yaml` — at minimum set `jetson_reborn_path`, `robot.ip`, and
the `board.*` dimensions to match your printed ChArUco target.

### 2. JetsonReborn_rebar (required for live capture and processing)

Clone the `JetsonReborn_rebar` repo and point `jetson_reborn_path` in
`configs/calib.yaml` to it. This repo uses JetsonReborn in two ways:

- **Scripts 02/04 (processing):** import `broker` modules (rectification, PCD
  generation) in-process via a `sys.path` injection. The broker dependencies
  (`open3d`, `requests`, `imageio`, `omegaconf`, `tqdm`, etc.) are included in
  this repo's `pyproject.toml`, so `uv sync` covers them.
- **Script 01 (capture):** runs `hik/hik_capture_cli.py` as a **subprocess**
  using JetsonReborn's own venv. This decouples the x86_64 HIK MVS SDK from the
  host process — this repo runs native ARM64 Python while the capture subprocess
  runs under Rosetta.

### 3. macOS camera setup (required for `--camera-mode live`)

The HIK MVS SDK ships x86_64-only dylibs. On Apple Silicon you need an x86_64
Python venv inside JetsonReborn:

```sh
# Install the HIK MVS SDK (default: /Library/MVS_SDK).
# Set MVCAM_SDK_PATH if installed elsewhere.

cd /path/to/JetsonReborn_rebar
uv venv --python cpython-3.11-macos-x86_64-none
uv sync
```

Verify the capture CLI works standalone:

```sh
cd /path/to/JetsonReborn_rebar
.venv/bin/python -m hik.hik_capture_cli --out /tmp/hik_test --timeout 10
```

### 4. Fairino SDK (required for `--robot-mode live`)

```sh
pip install path/to/fairino-python-sdk
```

## Collect A New Dataset

Live collection writes directly into the target folder. Each accepted sample is
stored as `<target>/<timestamp>/` with `raw_left.jpg`, `raw_right.jpg`,
`camera_model.json`, and `robot_pose.json`; the target root also gets
`dataset_info.json`, `manifest.jsonl`, and `pose_data.md`.

```sh
uv run python scripts/01_capture_pose.py \
  --out /Users/andyliu/DCIM_AI/eye2hand_run_001 \
  --count 20
```

For a hardware-free logic check:

```sh
uv run python scripts/01_capture_pose.py \
  --dry-run --auto --count 3 \
  --out /tmp/eye2hand_collect_dryrun
```

Useful collection modes:

- `--robot-mode live` reads Fairino `GetActualTCPPose(0)` from `robot.ip`.
- `--robot-mode manual` lets you paste `[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]`.
- `--camera-mode live` calls `hik/hik_capture_cli.py` in JetsonReborn's x86_64
  venv as a subprocess.
- `--camera-mode none` records TCP-only pose folders.
- `--dry-run` is `--robot-mode mock --camera-mode mock`.

Process and solve an arbitrary collection folder with:

```sh
uv run python scripts/02_process_dataset.py \
  --poses-dir /Users/andyliu/DCIM_AI/eye2hand_run_001

uv run python scripts/03_solve_handeye.py \
  --poses-dir /Users/andyliu/DCIM_AI/eye2hand_run_001 \
  --out /Users/andyliu/DCIM_AI/eye2hand_run_001/handeye
```

## Output Structure

### Phase 1 — Capture (`01_capture_pose.py`)

```
<target_folder>/
├── dataset_info.json               # collection metadata (tool, modes, config, timestamps)
├── manifest.jsonl                  # one JSON line per accepted sample
├── pose_data.md                    # human-readable pose log
└── <timestamp>/                    # one per accepted capture
    ├── robot_pose.json             # pose_mm_deg, T_base_tcp, timing, robot mode
    ├── raw_left.jpg                # camera-mode live/mock only
    ├── raw_right.jpg               # camera-mode live/mock only
    └── camera_model.json           # camera-mode live/mock only
```

### Phase 2 — Process (`02_process_dataset.py`)

Adds to each `<timestamp>/` folder:

```
<timestamp>/
├── rect/                           # broker rectification output
│   ├── rect_left.jpg
│   └── rect_right.jpg
├── pcd/                            # broker PCD output
│   ├── img0.jpg                    # scaled rectified left (matches depth resolution)
│   ├── img1.jpg                    # scaled rectified right
│   ├── depth_meter.npy             # (H', W') float metric depth
│   ├── K.txt                       # line 1: 9 floats (row-major K, full-res); line 2: baseline (m)
│   ├── cloud.ply                   # point cloud
│   └── vis.png                     # depth visualisation
├── target_pose.json                # T_target_cam, method, corner counts, RMSE metrics
└── target_pose_vis.png             # debug overlay (corners + coordinate axes on img0)
```

### Phase 3 — Solve (`03_solve_handeye.py`)

```
<output_dir>/                       # e.g. <target_folder>/handeye
├── T_cam_base.json                 # best solution: T_cam_base, T_base_cam, RMSE, method, n_pairs
└── handeye_summary.json            # all methods compared, pose IDs used, per-method metrics
```

## Existing DCIM_AI Dataset

For the collected timestamp folders under `/Users/andyliu/DCIM_AI`, run:

```sh
python scripts/04_calibrate_collected_dataset.py --force-target
```

The script reads `/Users/andyliu/DCIM_AI/pose_data.md`, reuses the existing
`img0.jpg`, `depth_meter.npy`, and `K.txt` files in each timestamp folder,
writes per-sample `robot_pose.json` / `target_pose.json`, drops later samples
with an identical TCP pose vector, and writes the calibration under
`/Users/andyliu/DCIM_AI/handeye/`.
