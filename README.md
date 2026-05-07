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

Once ~20 poses are captured we feed `(T_base_tcp, T_target_cam)` pairs to
`cv2.calibrateHandEye` (eye-to-hand convention) to solve for `T_cam_base`.

## Layout

```
configs/calib.yaml              # robot IP, board, paths, thresholds
src/eye2hand/                   # library code
scripts/01_capture_pose.py      # interactive jog+ENTER capture loop
scripts/02_process_dataset.py   # rectify + pcd + target-pose for every captured folder
scripts/03_solve_handeye.py     # AX=XB, write T_cam_base.json
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
