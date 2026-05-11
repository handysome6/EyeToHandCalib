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

```sh
uv sync
# Optional, for live capture only:
#   pip install path/to/fairino-python-sdk
#   ensure HIK MVS SDK is installed where hik.utils.load_hik_sdk() expects it
```

Edit `configs/calib.yaml` — at minimum set `jetson_reborn_path`, `robot.ip`, and
the `board.*` dimensions to match your printed ChArUco target.

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
- `--camera-mode live` uses `JetsonReborn_rebar/hik.HikSyncedCameras`.
- `--camera-mode none` records TCP-only pose folders.
- `--dry-run` is `--robot-mode mock --camera-mode mock`.

On macOS, the updated HIK wrapper follows `hik.utils.load_hik_sdk()` in
`JetsonReborn_rebar`. Set `MVCAM_SDK_PATH` only if the old MVS SDK is not at
`/Library/MVS_SDK`; on Apple Silicon, use an x86_64 Python interpreter if the
installed MVS dylibs are x86_64-only.

Process and solve an arbitrary collection folder with:

```sh
uv run python scripts/02_process_dataset.py \
  --poses-dir /Users/andyliu/DCIM_AI/eye2hand_run_001

uv run python scripts/03_solve_handeye.py \
  --poses-dir /Users/andyliu/DCIM_AI/eye2hand_run_001 \
  --out /Users/andyliu/DCIM_AI/eye2hand_run_001/handeye
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
