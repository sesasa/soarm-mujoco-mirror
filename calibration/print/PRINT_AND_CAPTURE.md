Print both pages of camera_calibration_A4.pdf on A4 landscape paper, at
100% / Actual size. Disable Fit, Shrink, and borderless enlargement. Verify
both 100 mm scale bars with a physical ruler before using the sheets.

Page 1: 25 mm squares, overhead camera. Page 2: 10 mm squares, wrist camera.
Each pattern has 10 × 7 squares and 9 × 6 internal corners. Fix the sheets
flat to stiff card; keep a white border around the pattern. Avoid glossy
lamination. Do not resize the small board to fill the page.

Hold the robot stationary (Space in the control viewer holds its current
pose). Leave the camera mounts and lenses fixed. Move the printed board in
front of each camera; the capture tool never opens the robot bus.

Start with the overhead camera:

    .venv/bin/python camera_calibration.py capture --camera overhead --scale-verified

For the wrist, use the small board:

    .venv/bin/python camera_calibration.py capture --camera wrist --scale-verified

Use Space in the CAMERA window to save a detected board, not the robot
window. Escape finishes. Collect 20–30 distinct sharp views per camera:
center and edges, different distances, and tilts about both axes. Keep the
whole grid in view, pause motion while saving, and avoid fingers covering
corners. Moving only a flat board parallel to the image is insufficient.
The tool rejects missing patterns and near-duplicate corner locations.
If the wrist cannot see the whole small board, hold it farther from the lens;
do not change the camera mounting or lens focus to make it fit.

Capture runs at the project's 640 × 480 settings in raw sensor orientation.
Each camera gets a separate session, square size, USB identity and format.
Use the exact session path printed at the end:

    .venv/bin/python camera_calibration.py solve calibration/camera_views/SESSION

The solver saves a candidate with per-view and held-out reprojection errors
and an undistorted preview. It does not automatically change the MuJoCo
camera. Poor pose diversity or high residuals require additional views.
This estimates intrinsics/distortion only; table-to-camera registration and
wrist hand-eye calibration follow separately. A checkerboard has a 180-degree
pose ambiguity, so fixed-frame registration will need an orientation convention.

Reference: https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html

If both printed check bars measure the same length other than 100 mm, use
that measured length instead of --scale-verified. For example, the user
reported 96 mm on the current overhead sheet:

    .venv/bin/python camera_calibration.py capture --camera overhead --measured-bar-mm 96

This records 24 mm large squares (25 × 0.96). A small sheet printed at the
same scale has 9.6 mm squares, but measure its bars independently before
using --measured-bar-mm 96 for the wrist capture. If horizontal and vertical
bar lengths differ, do not use a single scale factor; correct the print.
