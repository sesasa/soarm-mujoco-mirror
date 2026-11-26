# Camera intrinsics

`overhead_640x480.json` contains the initial reviewed overhead calibration for the device and capture mode recorded in that file. The 27 rigid-backed checkerboard views use measured 24 mm squares (printed 100 mm bars measured 96 mm). Overall RMS is 0.287 px; six held-out view errors range from 0.173 to 0.374 px. Across ten 20-view subsets, focal estimates vary by less than 0.7%.

Coverage is limited to checkerboard corners approximately x=126–526, y=52–285 in the 640×480 image. Bottom and outer-image accuracy is not validated. Lens calibration does not establish the camera pose relative to the table or robot, and is not yet applied to the simulation or live pipeline. Earlier flexible-paper captures remain archived and their result must not be used.

Next: independently verify the small printed board scale, calibrate wrist intrinsics, then establish table/robot extrinsics and wrist hand-eye calibration.

## Wrist provisional result

`wrist_640x480.json` records 37 views with 9.6 mm squares. A single radial coefficient gives RMS 0.711 px and held-out errors 0.437–0.792 px. Focal estimates vary by less than 1% across ten 27-view subsets, but principal-point estimates span about 31 px horizontally and 16 px vertically. This is provisional, with limited edge coverage and an assumed radial-only distortion model; it is not applied to any live pipeline or commands. The unrestricted five-coefficient fit and model comparison remain in the source session for review. Next work can use the overhead calibration to establish the table plane; wrist hand-eye and metric validation remain pending.
