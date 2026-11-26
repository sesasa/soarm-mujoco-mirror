# Table and base registration

The user confirmed the large sheet directly touches the tabletop. `board_plane.json` records that plane using calibrated overhead intrinsics; the most recent fit has 0.343 px RMS and camera-to-table perpendicular distance 0.42217 m. Board axes follow detector order and can flip by 180 degrees between captures.

`base_registration_candidate.json` is a provisional fit to three manually selected landmarks on one visible side of the fixed CAD base plate. Run `python fit_base_registration.py` to reproduce that fit. It assumes the lowest CAD plate surface touches the table, putting the model origin 2.4 mm above it. RMS is 3.78 px; uncertainty includes manual selection, lens extrapolation and CAD/physical differences.

`base_fit_overlay.jpg` shows plate landmarks and projected plate boundaries. `robot_pose_overlay.jpg` shows projected per-link convex outlines at read-only observed encoder positions. The full arm does not align sufficiently: this candidate is NOT accepted for physical commands and the working scene and joint mapping have not been changed. A second distinct, stationary arm pose is the next validation step. Keep the fixed base and overhead camera unchanged.

Second pose captured and compared with `compare_registration_pose.py`. The mismatch remains in both poses; neither independently establishes a unique correction. Existing joint offsets remain unchanged and the base registration stays unaccepted. Blue table lines were erased by the user and are now explicitly retired in config/scale_references.json. The printed checkerboard is the active scale reference.

Third pose captured with clear checkerboard at a new table location. Separate plane fit in pose3_plane gives 0.278 px RMS, camera height 0.42406 m, normal difference 0.056 degrees and height difference 1.88 mm from original plane. Do not combine board XY coordinates across these placements. Three-pose model comparison still shows mismatch; existing base fit is not accepted.
