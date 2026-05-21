# CLAUDE.md

## Transform naming convention

The calibration JSON files (`T_cam_base.json`) use a **source-first** naming convention:

- `T_cam_base` : camera-frame points -> base-frame (use this to transform point clouds from camera to robot base)
- `T_base_cam` : base-frame points -> camera-frame (inverse of above)

The name reads as `T_{from}_{to}`. This is defined in `src/eye2hand/handeye.py`.

## Robot TCP pose format

Fairino arm TCP poses are `[x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]`.
Euler convention: **XYZ intrinsic** (uppercase `"XYZ"` in scipy), equivalent to ZYX extrinsic.

## Environment

- Python 3.11 x86_64 (Rosetta) via `uv venv`
- Run scripts with `.venv/bin/python`
- Fairino SDK vendored in `vendor/fairino-sdk/`
