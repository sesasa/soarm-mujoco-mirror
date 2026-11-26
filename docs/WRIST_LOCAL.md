# Wrist local tracking: commissioning stage

Run `.venv/bin/python src/adaptive_grip.py --wrist-setup`.

This mode uses the existing overhead reach, then holds at the normal hover
position. It deliberately does not descend or close. The right panel is live
wrist video. The ordinary application without this flag is unchanged.

1. A holds; select the object in the overhead view; P prepares; G reaches.
2. At WRIST_SETUP, drag a tight rectangle around a textured part of the object
   in the right panel. Exclude fingertips and background from the rectangle.
3. Right-click the two fingertip contact points. A third click starts a new pair.
4. S saves the current wrist image, tracked patch, fingertip marks, joint
   feedback and command into this session's captures/live_grip directory.
5. Space stops; X releases torque. Support the arm before releasing.

The patch tracker uses OpenCV goodFeaturesToTrack, forward/backward pyramidal
Lucas–Kanade optical flow and RANSAC similarity fitting. It measures patch
center and rotation relative to its initial rectangular selection, not an
absolute object long-axis angle. It rejects stale frames, insufficient features,
poor consensus and clipped patches. No neural network is involved. It is an
observation tool; appearance drift and foreground/background confusion are still
possible. Fingertip marks are manual, fixed image annotations; they do not track
moving jaws.

Saved records explicitly say validated_for_control=false. They are hover
observations, not proven grasp targets. A fingertip image midpoint at hover
is not necessarily the correct projected object center at contact: perspective
and changing camera height matter.

Remaining physical commissioning: establish a successful alignment reference
at the intended grasp height and measure/validate a local image-motion Jacobian
for small lateral and orientation corrections. Validate its direction, scale,
conditioning and prediction on held-out motions. Then implement a wrist
handoff with fresh-target checks, bounded corrections, height-aware references,
and stop-on-loss before enabling descent and closure. The rejected wrist
extrinsic calibration must not be used to generate commands. Neither automatic
wrist steering nor wrist-guided closure is enabled by this change.

Tests: tests.test_wrist_local covers measured synthetic translation/rotation, stale
and reversed frames, blank-image loss and insufficient texture. These tests do
not establish real robot steering accuracy.
