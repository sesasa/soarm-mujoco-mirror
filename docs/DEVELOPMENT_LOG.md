# SOARM MuJoCo Mirror

Initial simulation of the photographed desk with an SO-101, six joint controls,
fixed overhead camera, wrist camera, and approximate tabletop objects.

## Run

```bash
.venv/bin/python src/simulate.py
```

Open **Control** in the MuJoCo viewer's right panel. The six sliders are shoulder
pan, shoulder lift, elbow flex, wrist flex, wrist roll and gripper. Values are
**radians**, including the gripper. Requested positions are ramped at 0.5 rad/s
before reaching the simulated position actuators. Camera selection is under
Rendering. The current implementation has no hardware connection and does not
open the serial port. On desktops with `DISPLAY` available, the launcher uses
GLFW's X11 backend to avoid a native-Wayland shutdown crash observed here.

Record simulated targets, ramped commands, and simulated joint feedback:

```bash
.venv/bin/python src/simulate.py --log logs/session.jsonl
```

Recreate the environment if needed:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Headless physics and preview:

```bash
MUJOCO_GL=egl .venv/bin/python src/simulate.py --headless --seconds 5 --render captures/scene.png
.venv/bin/python src/verify_simulation.py
```

## Geometry and calibration status

`config/scene.json` holds editable scene dimensions and the initial pose;
`src/build_scene.py` regenerates `scene.xml` at startup. Z=0 is approximately the
tabletop and the robot base defines the XY reference. Remote length and width
are measured: 0.22 × 0.05 m. The curved tabletop outline is traced from
`IMG_8795.HEIC`, using an approximate 58 cm reading across the taped section.
This is not the maximum table width. The trace uses uniform image scaling, so
perspective distortion remains; the full outline is an estimate. Source pixel
coordinates and the scale reference are recorded in `config/scene.json`.
Its thickness, object positions, camera positions/FOV, and stand dimensions
are **estimates**. Concave collision geometry is decomposed into triangular
prisms so the table indentation remains open.
The scene uses photos for approximate layout, not a metrically reconstructed
mesh. Robot geometry comes from the vendored model, not inferred photos.
The model's wrist camera transform is also an uncalibrated starting point.
The wire holder is currently fixed; the remote, tape measure proxy, and candle
are free rigid bodies. Mass, friction, and servo dynamics are not identified
from the actual setup. There is no live visual object tracking yet.

## Dual cameras

After the physical camera swap, fresh frames confirmed camera 4 is fixed
overhead and camera 6 is wrist. Stable USB
paths are in `config/cameras.json`; both identify as icSpring USB cameras.
No depth stream was established.

```bash
.venv/bin/python src/capture_cameras.py --seconds 10 --preview
```

This captures both cameras concurrently at 640 × 480, pacing the processing
loop at up to 15 Hz. The devices negotiate 30 FPS; this is separate from the
processing-loop frequency. Latest images and a negotiation report are saved
in `captures/`. Escape closes preview. It never opens the robot bus.

The existing notes at `../action_decoder/dual_camera_notes.txt` request MJPG
on both devices. Live probing on 2026-09-18 found camera 4 supports MJPG,
but camera 6 advertises YUYV only and falls back to it when MJPG is requested.
This project explicitly uses MJPG for camera 4 and YUYV for camera 6 and checks
the negotiated formats. Both returned 67 frames in a five-second simultaneous
capture test. After closing the curtains, fresh live views were checked and both cameras
were returned to automatic exposure and automatic white balance, with 50 Hz
power-line compensation. The old direct-sunlight manual values are no longer
applied. Camera controls use `v4l2-ctl` (from `v4l-utils`) and remain set after
capture exits. Exposure should be checked again after final placement.

The user physically rotated the wrist camera 180 degrees. Its raw image is
now upright, so software rotation is disabled (`rotation_deg: 0`). Raw
`captures/camera_6.jpg` and `camera_6_view.jpg` now have the same orientation. Rotation changes display orientation, not camera aim. Calibrate
against raw frames, or explicitly transform intrinsics and image coordinates
if using rotated frames in a future perception pipeline.

For physical camera positioning, use the labeled center-guide previews:

```bash
.venv/bin/python src/capture_cameras.py --seconds 300 --alignment
```

Escape closes both windows. Center the overhead view on the manipulation
area, with space around the arm and target objects. Aim the wrist camera
along the gripper approach so the fingertips and workspace beyond them are
visible; the latest view after the additional tilt shows both fingertips and the gap
near the bottom, with the remote beyond them. The view is narrow, but can
support an initial grasping-view check. Visibility throughout arm and gripper
motion is not yet verified. The wrist camera advertises only 640 × 480 YUYV
and exposes no V4L2 zoom control; capture does not digitally crop the image.
The latest physical layout reference is `Photos-1-001/IMG_8798.HEIC`.
The object positions and overhead view now have an approximate visual fit
from manually selected landmarks. See `captures/alignment/comparison.jpg`.
The arm pose and wrist camera are provisional and do not yet match exactly;
encoder alignment and camera calibration remain necessary. The recorded
reprojection error describes the selected object landmarks, not metric
accuracy or independent validation.
Camera intrinsics/extrinsics remain uncalibrated. No robot motion was used to adjust these views.
The simulated overhead mount was moved back and tilted toward the work area
as an approximate layout change, not a measured camera-pose fit.

## Hardware calibration and read-only mirror

The existing LeRobot environment is reused at
`/home/saeed/miniforge3/envs/lerobot/bin/python`. The project has a separate
calibration identity, `soarm_mirror`, in `calibration/`; files from other
projects are not assumed valid. `src/robot_setup.py` connects directly to the
LeRobot bus, avoiding the robot connection routine's automatic configuration.
A read-only probe reached all six motors and found torque disabled.
The user completed the midpoint and range sweep; calibration is saved in
`calibration/soarm_mirror.json` and was verified against the motors. All five
swept ranges are nonzero; `calibration/validation.json` records their spans.
The wrist-roll range is LeRobot's full-turn convention, not a measured cable
clearance limit. Motor calibration does not validate the MuJoCo mapping.

```bash
/home/saeed/miniforge3/envs/lerobot/bin/python src/robot_setup.py probe
./calibrate_robot.sh
```

Calibration requires a person at the arm. Support it, clear the sweep area,
place joints roughly at mid-travel when prompted, then move the requested
joints through their ranges by hand. Follow the live prompts; do not force
stops or strain camera cables. LeRobot excludes wrist roll from the range
sweep. This process writes calibration to the motors and project file, and
leaves torque disabled. Probe and observe do not change torque or goals.

After calibration:

```bash
/home/saeed/miniforge3/envs/lerobot/bin/python src/robot_setup.py observe
.venv/bin/python src/mirror_robot.py
```

The mirror reads joint positions at about 15 Hz through one bus process.
It stops on missing/stale feedback and **never sends motor goals**. Directions,
zero offsets and gripper endpoints in `config/joint_mapping.json` are
provisional; use the mirror to validate them one joint at a time. The mapping
is explicitly marked unvalidated for commands. Joint angles use LeRobot's
degrees, converted to MuJoCo radians; gripper feedback is 0–100 and maps to
an estimated hinge range. Out-of-range observations are not silently clipped.
The provisional conversion passed checks and calibrated feedback was read
successfully. Visual direction/offset validation is still pending. The observed
shoulder angle can exceed the generic model limit, so hardware commands must
remain disabled until mapping and limits are reconciled.

The eventual control path is six simulation sliders → bounded joint
trajectory → validated LeRobot mapping → physical arm, with separate encoder
feedback. That outgoing command bridge is not enabled or implemented yet.
Before it is implemented, validate mapping, initialize commands from measured
pose, and implement stop/timeout behavior and serialized robot I/O. The
current simulation trajectory is not collision-aware planning.

Next after joint alignment: camera intrinsics and table registration, then
known-object tracking and IK. Visual fitting currently assumes a pinhole
camera and uses manually picked remote corners and object centers; it is not
camera calibration. `src/align_scene.py` can generate a new rough alignment
proposal, while `config/scene.json` contains the reviewed working scene.

## Asset provenance

`assets/so101/` was copied from the existing local
`../mujoco_menagerie-main/robotstudio_so101/` directory. Its README and Apache-2.0
LICENSE are preserved. Upstream:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101

The builder changes the appearance to white, adds the desk and approximate
camera housing, adjusts rendering, and clips actuator ranges to joint ranges.
Original robot assets are preserved.

## Added scale references

Two 15 cm table lines and a 15 cm ruler are recorded in
`config/scale_references.json`. The thick blue line endpoints are registered in the 640 × 480 capture
`captures/alignment/blue_scale_reference_640.jpg`. The current approximate
camera model maps their stated 15 cm lengths to about 15.73 cm and 14.62 cm,
respectively. This is a local scale check, not full camera calibration; ruler
mark endpoints remain unregistered.
`captures/scale_reference_1280.jpg` is a higher-resolution overhead view; it
has a different field of view than the normal 640 × 480 mode, so its pixel
coordinates must not be reused without accounting for that difference.

Base rotation direction was confirmed by the user in the read-only mirror.
Its offset and range remain unvalidated, as do the other joint mappings.

Wrist-roll direction matches the physical arm. A -90 degree offset is now
applied in the feedback-to-simulation mapping to put the wrist camera on top,
based on the user's observed quarter-turn mismatch and rendered checks of
both signs. Visual confirmation and travel-limit reconciliation are pending.
This does not change motor calibration or send motion commands.

All six motion directions have been confirmed by the user in the read-only
mirror. The -90 degree wrist-roll offset and visually similar gripper opening
were also confirmed. This validates observed behavior only: exact offsets,
travel limits and gripper endpoints are not yet fully measured, so
`validated_for_commands` remains false.

The elbow feedback mapping now includes a -3.5 degree offset, based on the
user's midrange comparison (`captures/midrange.png`). Kinematic checks
confirmed this sign raises the forearm; visual confirmation is pending.
The existing -90 degree wrist-roll correction is retained. No motor homing
or hardware calibration values were changed.

The user confirmed the -3.5 degree elbow correction. Wrist bend now has a
-2.5 degree feedback offset to raise the gripper, pending visual confirmation.
A kinematic check confirmed this sign increases gripper-tip height. These
are simulation-mapping corrections; motor calibration is unchanged.

Latest visual adjustment: one additional upward degree on elbow and wrist
bend. Total feedback offsets are now elbow -4.5 degrees, wrist bend -3.5
degrees, and wrist roll -90 degrees. Visual confirmation is pending; no
motor calibration values were changed.

Latest requested adjustment adds another upward degree to both joints. Total
feedback offsets: elbow -5.5 degrees, wrist bend -4.5 degrees, wrist roll
-90 degrees. Visual confirmation remains pending; motor calibration is unchanged.

The user accepted the final visual alignment. Confirmed feedback offsets are
elbow -5.5 degrees, wrist bend -4.5 degrees, and wrist roll -90 degrees.
All six motion directions and visually similar gripper opening were confirmed.
The outgoing bridge is now implemented for the limited commissioning mode
described below. Full-range validation remains pending.

## First simulation-to-robot control test

```bash
.venv/bin/python src/control_robot.py
```

Close the read-only mirror first: only one process can own the serial bus.
The new viewer starts in MONITOR with torque off; the solid arm is measured
robot feedback. Open **Control** on the right for the six sliders in radians.
Move the real arm into a comfortable middle pose with the gripper partly open
before enabling. The overlay identifies joints outside the permitted range.

- **A**: enable position control at the freshly measured pose, resetting the
  slider targets and the initial test window.
- Move **one slider** a small amount, initially about 1 degree (0.017 radians).
- **Space**: hold the current measured position with torque on.
- **X**: release torque. Support the arm first.
- Closing the window releases torque and disconnects; support the arm first.

This commissioning mode restricts each joint to ±5 degrees from its enable
pose, ramps commands at 5 degrees/s, and intersects measured encoder ranges
(with a 2 motor-degree margin) with model limits. It rejects invalid values,
stale commands, excessive tracking error, and unsupported motor modes. The
wrist's full-turn calibration is not treated as validated unrestricted travel.
The inverse mapping includes the confirmed elbow/wrist offsets. Gripper
commands invert the existing opening mapping and are bounded by measured
encoder endpoints. No modulo wrapping is applied to any joint.

A single LeRobot worker owns all motor reads/writes. It writes the exact
current raw positions as goals before enabling torque. If GUI commands stop
arriving for 0.5 seconds, it latches HOLD. EOF or normal shutdown releases
torque. These are software safeguards; a communication failure can prevent
a final motor write. There is no hardware emergency-stop implementation or
validated collision avoidance. The approximate desk model is not used to
claim physical collision clearance.

The full-range `validated_for_commands` flag remains false: this mode is only
for limited commissioning after the visual checks. Eleven tests passed for
mapping, bounds, enable ordering, rate limits, stale feedback, watchdog hold,
and shutdown release. Live MONITOR feedback was checked without motor writes;
the user subsequently confirmed that all six controls respond on the real
arm. The controller reached ACTIVE. This validates the initial limited
control test, not full-range accuracy or collision clearance.

## Printed camera calibration

Print `calibration/print/camera_calibration_A4.pdf` (two A4 landscape pages)
at 100% Actual size and verify both 100 mm check bars. The large board uses
25 mm squares for the overhead camera; the small board uses 10 mm squares
for the wrist. Both have 9 × 6 internal corners. See
`calibration/print/PRINT_AND_CAPTURE.md` for capture and solving instructions.
PDF dimensions and corner detection were checked, and a synthetic multi-pose
calibration recovered known camera parameters. Real camera calibration awaits
the printed-board captures. No camera calibration values have been applied yet.

### Overhead scene-change preview

Run `.venv/bin/python src/scene_changes.py`. This separate preview reads overhead video
and robot encoders without sending commands or changing torque. The green polygon
is the initial tabletop work area. Press **B** with hands clear to record a baseline;
move or add a pen inside that area and withdraw your hand. After about half a second,
orange boxes mark changed pixels and orange slabs show approximate tabletop
footprints in MuJoCo. **S** saves a screenshot; **Esc** closes the preview.

This is change detection, not object recognition: moving an object can mark both its
old and new locations. Hands and shadows can also trigger it. Broad changes are
suppressed and small global colour/exposure shifts are compensated approximately.
Slab thickness is purely visual; no object height or collision geometry is estimated.
The plane in `captures/scene_changes/plane.json` uses the current 9.6 mm-square sheet
on the tabletop; overhead/base registration remains provisional. Moving the camera
or base invalidates this alignment. Wrist calibration remains deferred and no wrist
mounting candidate was applied. Preview output is not used for reaching commands.

`captures/scene_changes/latest.jpg` and `latest.json` hold the latest preview and
approximate locations; `baseline.png` records the baseline. Run
`PYTHONPATH=src .venv/bin/python -m unittest discover -s src/tests -p test_scene_changes.py -v`
for detector and plane-mapping checks. `--offline` produces a labelled synthetic
rendering check without opening the camera or robot connection.

The change preview now draws the detected contour in cyan and a tighter rotated
footprint in orange. Footprints are fitted after mapping pixels onto the tabletop,
so the MuJoCo slabs follow the observed direction rather than camera-axis boxes.
They still describe changed regions, including vacated positions and shadows—not
verified object dimensions. Use `--reuse-baseline` when restarting with the same
camera and scene alignment to preserve `baseline.png`.

Current-appearance refinement now separates foreground from the nearby tabletop
within each changed region before sizing the footprint. It uses local colour models
and rejects moderate neutral darkening to reduce shadows and empty former positions.
This assumes a locally plain tabletop: low-contrast objects can be missed and strong
shadows or patterned backgrounds can still be included. It does not establish exact
physical dimensions. The original difference detector remains the trigger; this is
not a complete inventory of all stationary objects. Seven checks cover the detector,
metric projection, rotation, and a pen/shadow/vacated-position regression case.

Disconnected interior patches (for example a pen label) now merge when they lie
entirely inside a larger contour's convex envelope. This preserves the outer width
and does not join merely adjacent or end-to-end silhouettes. It is a geometric
heuristic, not semantic object identity; overlapping objects remain ambiguous.
The recorded split-pen case is in `captures/scene_changes/split_pen_regression.json`.
Nine automated checks include fragment merging and separate-neighbour cases.

Adjacent-pen correction: stronger rejection of neutral darkening prevents the
shadow in the narrow gap from becoming a foreground bridge. The actual failure
image (`captures/scene_changes/adjacent_pens.png`) is exercised by the tenth test,
including multiple segmentation seeds. This colour/contrast heuristic may miss
parts of low-contrast objects; it is not general instance segmentation and has not
been validated for touching/overlapping objects or different lighting.
Fresh-frame validation also exposed an intermittently detached pen tip. Small
aligned continuations may now join an elongated body only with limited gap, area,
length and width growth. Similar-size adjacent/end-to-end objects remain separate
in tests. An eleventh test replays the actual three-fragment frame. Small separate
objects aligned with a larger object can still be ambiguous; no grasping commands
use these heuristics.

### Select a preview target

Click inside an object's outline in any of the three panels (away from overlapping
outlines). The selected outline/cross turns green and a green sphere marks its
approximate tabletop center in MuJoCo. **C** clears selection. **B** resets the
baseline and clears selection. A missing/ambiguous match also clears selection;
small motions are associated by footprint overlap, not semantic identity.
`latest.json` records `selected_target_base_m` or null with `selection_status`.
These preview coordinates are not a grasp pose, do not include object height, and
are not connected to any motion command. The next stage needs simulated reaching
and calibration/clearance checks before any real-arm trajectory execution.
Run `PYTHONPATH=src .venv/bin/python -m unittest discover -s src/tests -v` for all 15 checks.

### Simulation-only hover reach

Select an object, then press **R** in the scene-change window. A separate
`SIMULATED REACH - real arm stays still` window animates the virtual arm toward a
point 8 cm above the detected tabletop center. **X** stops the virtual preview;
**Esc** closes the application. A missing, moved or ambiguous target pauses the
virtual animation without closing its window. It resumes after 0.35 seconds of a
unique matching observation. The captured destination remains fixed; re-select
and press R/G to plan for a new location.

The bounded position-only IK solver targets the model's `gripperframe` site, holds
gripper opening, and rejects solutions with position residual above 2 mm. A smooth
joint interpolation stays inside model limits and limits peak animation speed to
25 degrees/s. A measured starting pose outside model limits is clipped for the
virtual preview and the maximum adjustment is shown. This is kinematic animation,
not dynamic tracking or a collision-checked trajectory; grasp orientation, object
height, holder clearance and physical calibration remain unvalidated. Model IK
residual is not physical positioning accuracy. No hardware commands are generated.
`last_simulated_reach.json` records the last preview plan and its limitations.
All 20 detector, selection and IK tests run via the test command above.

Reach-window crash fix: custom MuJoCo footprints and target markers now explicitly
set `segid=-1`, excluding them from segmentation lookup. Previously uninitialized
marker IDs could crash the second renderer. A rendering regression alternates two
preview instances, varies custom markers and poisons unused segmentation IDs with
the recorded crash value. The suite now has 21 checks (rendering requires a display
or EGL/OSMesa context).

### Aligned hover (simulation only)

Select an elongated object and press **G**. The solver aims the gripper's local
-Z axis toward the table and aligns local Y with the object's long axis, so local
X (jaw closing direction) spans its width. It tries both equivalent 180-degree
orientations. **R** still provides the position-only hover. Both modes keep the
reference point 8 cm above the table at the endpoint and hold the jaw opening.
**X** stops. Target loss, >15 mm translation, >15 degrees axis change, or a
substantial size change pauses the animation while keeping the window open.

Five arm joints cannot satisfy every full pose. Acceptable preview solutions must
have <=2 mm model position error, <=8 degrees downward tilt and jaw-axis error,
and <=10 degrees full orientation error. Errors appear in the animation window
and plan JSON; these are model residuals, not real-world accuracy. The current
highlighter pose is approximately 4 degrees tilted and reaches the model's wrist
bend limit. There is no descent, grasp, orientation-constrained path or collision
validation. Small/near-square detections are rejected when their axis is unclear.
The suite now has 23 checks, including jaw-axis orientation, 180-degree symmetry,
limits, invalid targets and rendering.

Reach pause/resume fix: the animation monitors its captured target independently
of the click-selection UI, so a brief loss of selection does not destroy the plan.
A unique geometric match (position, axis and size) must remain stable for 0.35 s
before virtual time advances; time spent paused never causes an animation jump.
This is a preview-only association heuristic, not semantic identity verification.
**C/B** explicitly clear/reset and stop the preview, as does **X**.
`reach_latest.jpg` records the most recent reach display, including pause reasons.
The suite now has 27 checks, including transient loss, recovery, motion, ambiguous
matches and preserved animation time.

### Side view and sampled table clearance

The reach window's middle panel now shows a side view of the virtual pose and
provisional tabletop. The green vertical line marks the hover point's 8 cm height.
Before animation, 81 joint-interpolated poses are checked against an infinite
plane using the moving visual mesh vertices and primitive support distances.
The fixed base and hidden legacy holder are excluded; the current approximate
visual holder is included. The report gives the minimum moving-arm and separate
gripper/holder gaps. A sampled intersection blocks the new preview plan.

The checked aligned hover had approximately 8.9 mm minimum moving-arm clearance
at the shoulder, and 73.5 mm minimum gripper/holder clearance. These are provisional
model estimates, not measured physical gaps. Registration/shape uncertainty,
between-sample motion, tabletop objects, cables and self-collisions remain unchecked.
This does not authorize physical motion or establish a collision-free descent.
Reports are included in `last_simulated_reach.json` under `table_clearance`;
the initial check is saved as `table_clearance_check.json`. All 30 checks include
analytical primitive/mesh distances, a sampled intersection and side rendering.

### Vertical preview steps

After **G** finishes and the target is stable, press **D** to lower the virtual
reference point by 1 cm. Wait for each step to finish; repeat down to a minimum
height of 3 cm above the tabletop. **U** raises it 1 cm, up to 8 cm. Each step takes
at least 2 seconds, keeps jaw opening unchanged, and starts at the previous virtual
endpoint rather than the physical arm pose. **X** stops. This does not grasp or close
the jaws, and tabletop objects still have no measured height/collision geometry.

Before each step, bounded IK must preserve alignment, avoid a >20-degree joint
branch change, and pass 41 intermediate pose checks (<=2 mm lateral error, <=8 deg
down tilt/jaw-axis error, no >0.5 mm reversal). The 81-pose visual/table check must
retain at least 5 mm moving-arm and 10 mm gripper/holder clearance. These thresholds
are preview checks, not physical safety margins or continuous collision validation.
A refused step keeps the previous plan/window and displays the reason.
The checked sequence reached 3 cm reference height with a model gripper/holder gap
of about 24.5 mm. `descent_checks.json` records the results. All 34 tests include
step continuity, ascent, descent limits and clearance refusal without plan mutation.

### Measured highlighter shape

`config/measured_objects.json` records the user's 17 mm height and 27 mm width.
A unique bright pink/red candidate receives a rectangular visual proxy with those
dimensions, resting on the measured table plane. Length and pose remain estimated
from the overhead footprint (including parallax/segmentation uncertainty). Blue pen
and other detections keep flat change markers. No match or multiple pink matches
leaves the measurement unassigned; colour association is not semantic identity.

The proxy appears in the camera-aligned simulation and reach side view. A reach
selected on that highlighter stores the dimensions, and vertical steps retain them.
The reach display subtracts object height from the sampled gripper/holder table gap
to report separation from the object's top plane. At the previous 3 cm reference
height this is about 7.4 mm. This is not a full object-contact check or a grasp:
rectangular proxy dynamics, friction and finger contact are not implemented, and
jaw opening remains unchanged. The suite now has 37 checks, including measured
proxy dimensions, tabletop placement and rejection of ambiguous colour matches.

### Open the virtual gripper for the measured highlighter

After selecting the highlighter and completing **G**, press **O**. The virtual jaw
opens smoothly to an approximate 37 mm fingertip gap (27 mm measured object width
plus 10 mm total allowance); the five arm joints stay fixed. If it is already wider,
O does not close it. D/U retains the opening. The current model gap is displayed.

Gap is the separation along gripper-local X between named fingertip sphere surfaces,
not a measured physical aperture or full finger-body contact check. The opening is
limited by the model joint range, and sampled clearance must retain 5 mm moving-arm
table gap and 5 mm gripper/holder separation above the object's top plane. The
real robot receives no commands. This still does not implement grasping/contact.
The suite has 42 checks, including opening size, monotonic opening, fixed arm pose,
already-open behaviour, invalid dimensions/range, clearance rejection and preserving
the opening on descent. `jaw_opening_check.json` records the initial model check.

### Isolated finger–object contact experiment

Run `.venv/bin/python src/contact_preview.py` to animate a separate, hardware-free
MuJoCo experiment from the last measured/open-jaw snapshot. `--headless` runs and
records without a window; `--source PATH` chooses a saved snapshot. The app does
not open cameras, serial ports, or command workers. **Esc** closes its window.

The highlighter is a free rigid box, 27 mm wide and 17 mm high, with estimated
length. It settles under gravity, the arm descends with the jaws open toward a
12 mm reference height, and the moving jaw closes at an 8 deg/s target rate.
Decomposed finger collision meshes interact with the box and tabletop; approximate
holder geometry also participates. Contacts must produce >0.02 N normal force on
both fingers for >0.25 s before the closing target stops advancing. The test then
holds briefly and records the result. Finger/table or holder/object contact stops
the test. This separate experiment does not alter the normal preview's 3 cm floor.

ASSUMED simulation parameters: 30 g object mass, friction coefficient 0.6, jaw
actuator kp=5, kv=0.08 and torque limit +/-0.08 Nm. No physical force or friction
calibration supports these values. Other tabletop objects, arm-link collisions and
self-collision are not included. No lift is attempted, and contact is not proof of
a stable or physically reproducible grasp. The first completed run established
both contacts after roughly 11 s; the object moved about 8 mm toward the fixed jaw.
`captures/contact_preview/result.json`, `contact_trace.json`, and `latest.jpg`
record the outcome and assumptions. The suite has 44 tests, including free-object
bilateral contact and continued table support in this particular fixture.

### Simulation-only lift test

Run `.venv/bin/python src/contact_preview.py --lift`. After the same bilateral-contact
check and a brief hold, the arm follows an IK path 2 cm upward over 4 seconds while
maintaining the closing target. The highlighter remains a free body: no weld,
attachment or prescribed object motion is used. The experiment then holds for two
seconds. A pass requires both finger forces >0.02 N, object-bottom clearance >10 mm,
and table force <0.01 N continuously for at least 1.9 seconds of the final hold.
Loss of bilateral contact for >0.25 seconds or unexpected contact stops the test.
The lift uses the existing assumed mass/friction/motor parameters without retuning.

The tested fixture passed: the final object/table gap was about 16.7 mm, both fingers
remained in contact and table support was zero. The commanded arm rise is 20 mm;
object clearance differs because of motion and orientation within the grasp.
This is one simulated result, not physical grasp validation or a robustness study.
`lift_result.json` and `lift_trace.json` preserve the lift evidence separately.
The suite has 46 tests, including free-body setup and sustained airborne support.

### Simulation-only pick-up and release cycle

Run `.venv/bin/python src/contact_preview.py --cycle` to watch contact, the 2 cm lift,
set-down, opening and withdrawal. The completed window stays open until **Esc**.
This isolated experiment reads the saved scene and sends no hardware commands.

After the airborne hold passes, the gripper lowers toward its previous height.
Opening starts only after table support exceeds 0.10 N and the object's bottom
is within 1.5 mm of the table continuously for 0.25 seconds. The jaws open over
two seconds. Withdrawal requires both finger forces below 0.02 N and continuing
table support for 0.25 seconds. The open gripper then rises 3 cm; success requires
one second of independent table support, no finger contact and object speed below
1 cm/s. Missing support, lingering finger contact or failure to settle times out.
Unexpected finger/table or holder/object contact also stops the experiment.

The saved fixture completed the cycle in about 27.5 simulated seconds. It ended
with the object on the table, no finger contact and about 8.4 mm displacement
from its initial location. This is a single simulated fixture with the same
assumed contact parameters and collision exclusions described above.
`cycle_result.json` and `cycle_trace.json` in `captures/contact_preview/` preserve
the result and phase evidence. Tests cover support gating, persistent finger
contact, settling requirements and the complete free-object physics cycle.

### First bounded physical approach segment

`src/control_robot.py --step-file calibration/prepared_approach_step.json` retains the
existing ±5-degree worker window. A enables holding at measured position; P runs
one prepared segment once. Space cancels and holds; X releases torque (support
arm first). The prepared segment is at most 2.5 degrees per joint, with up to
0.5-degree start-pose tolerance, and playback rejects any effective excursion
above 3 degrees. Quintic motion is limited to 1 degree/second and keeps the jaw
opening unchanged. It refuses changed poses, mapping changes, invalid values,
and targets outside the active worker limits. Sliders are ignored during playback.

Preparation checks sampled visual-geometry/table clearance (5 mm moving links,
50 mm gripper/holder); this does not validate object, cable or self-collision.
The first segment points toward a position-only 12 cm reference hover, not a
vertically aligned grasp. Full-path execution remains unavailable. Larger paths
must not be executed by silently resetting the commissioning window.
`control_live.json` records current feedback; `approach_tracking.jsonl` records
feedback and worker commands during sessions launched with a step file.

### Supervised real high hover

Run `.venv/bin/python src/real_hover.py`. This is an explicit separate mode; it does
not widen the normal slider controller's ±5-degree window. It owns the overhead
camera and starts one `src/control_worker.py --allow-hover` motor connection.

- A: initialize goals at measured positions and enable holding. Invalid starting
  joint positions are refused. Move hands clear after the arm holds.
- P: detect the stationary pink highlighter and preview a position-only path to a
  12 cm gripper-reference height. The real arm keeps holding during preview.
- G: after the five-second visual preview, execute the single path. The preview
  expires after 30 seconds; use P again if needed before any execution.
- Space: stop and hold. X or closing the window releases torque; support first.

The worker independently checks current calibration mapping, command continuity,
starting feedback within 2 degrees, hardware/model range intersection, unchanged
jaw opening, at most 60 degrees travel per joint and a quintic peak speed of
3 degrees/second. The preparation samples 161 poses for at least 5 mm moving-link
and 80 mm gripper/holder tabletop clearance. Current measured-pose clearance is
checked too. Commands are generated by the worker, not streamed from the preview.

During motion, a 0.5-second missing heartbeat, control interval over 0.2 seconds,
or joint tracking error over 3 degrees stops and holds. The camera UI requests a
hold on lost/ambiguous highlighter detection, scene object count/position/axis
changes or a delayed camera cycle. Completion requires a two-second endpoint
settling period with all measured joints within 2 degrees. A hover is single-use,
including after interruption; no automatic resume, descent, jaw closure or grasp.

This is supervised commissioning using approximate mapping, camera/table
registration and holder geometry. Table clearance is sampled, and vision only
monitors detected objects; neither provides complete collision coverage (notably
cables, self-collision and unobserved objects). The height is the model's reference
point, not a measurement of physical fingertip height. Logs, plan, camera image
and result are under `captures/real_hover/<session>/`; current feedback is in
`calibration/hover_live.json`. No motor gains or calibration offsets are changed.

First physical hover attempt stopped at about 9.1 seconds after the camera's
unique-highlighter classification became false; it did not reach the endpoint.
The diagnostic image showed a split blue-pen footprint and a near-threshold red
label fragment. This is a plausible cause, not a confirmed reconstruction of the
unsaved stop frame. The revised guard tracks the already-selected highlighter by
position and colour, and allows contained fragments to represent an unchanged
footprint when their extents and aggregate area remain consistent. Missing targets,
unmatched objects and changed extents still stop the hover. Future camera stops
save `stop_frame.png` and `stop_diagnostic.json`, and preserve the camera reason in
the worker instead of labeling an automatic camera hold as a user pause.

The pre-start stability timer now follows the selected highlighter, rather than
requiring every scene fragment to remain identical or the highlighter to be the
only red colour candidate. This fixes a remaining start gate where a visible
highlighter could never accumulate 0.7 seconds of stability. The whole-scene
comparison still runs before G and during motion. G now reports which condition
blocks execution, and live telemetry records `target_stable_s`.

### Larger supervised base-only test

`.venv/bin/python src/control_robot.py --base-test` provides an explicit +/-10-degree
shoulder_pan window at 3 degrees/second, intersected with calibrated/model limits.
All other joint targets are locked at the enabling pose. A holds/enables, the
shoulder_pan slider requests rotation, Space stops/holds, X releases torque.
The ordinary six-slider +/-5-degree controller and hover mode are unchanged;
base-test and hover/step modes cannot be combined. This is supervised joint-space
commissioning, not an object-reaching command. Its camera-object scene guard is
not used; the tested swept geometry must be clear and the operator watches motion.
Mapping, stale-command, heartbeat and motor-tracking checks remain active.

### Supervised real grasp attempt

`.venv/bin/python src/grasp_control.py` opens overhead, wrist-camera and model views
with one `src/control_worker.py --allow-grasp` connection. It starts torque-off.
A holds at the measured pose. Each stage has a separate preview key, followed by
G after a five-second preview to command that stage. There is no automatic chain:

- P / G: approach a downward-facing 4 cm reference pose, jaws unchanged.
- D / G: lower to an 18 mm reference height; inspect jaw/object alignment first.
- C / G: close toward an approximate model tip gap of 25 mm. After closure, another
  C / G tightens by 2 degrees if the operator requests it after inspecting the grip.
- L / G: lift the reference by 2 cm; available only after a completed close stage.
- O / G: reopen to the original opening; the operator controls when release is safe.
- Space stops/holds; X or closing releases torque (support the arm first).

Motion limits are stage-specific: approach at most 140 degrees per joint, peak
3 deg/s; lower/lift at most 20 degrees, peak 1 deg/s; jaw-only commands at most
30 degrees, peak 2 deg/s. Command/model range intersection, mapping consistency,
preview freshness, phase order and replay checks are enforced in the motor worker.
Arm tracking error >3 degrees stops; jaw contact stages permit up to 5 degrees
of position error. Stage completion requires two seconds of settling, arm errors
<=2 degrees and jaw error <=2 degrees (<=5 while closing/lifting). This is position
control with existing motor settings, not calibrated force control or grip detection.
Completed close/lift goals remain energized; stop/hold preserves the established
jaw goal while holding the arm. Torque release clears the grasp phase.

Preparation checks 161 sampled tabletop-clearance poses: >=5 mm moving geometry,
>=25 mm gripper/holder on approach, >=8 mm on lowering/lift and >=5 mm during jaw
motion. Lower/lift reference lateral drift is limited to 4 mm in 81 samples.
The model-only sequence passed for the saved starting pose: about 69 s approach,
15 s lowering, 14 s closing, 14 s lift; minimum gripper/table clearance ~9.6 mm
through the modeled sequence. These ideal-feedback results do not prove physical
alignment, grip force, object retention or complete collision safety.

This mode uses the selected highlighter, not unstable outlines of unrelated pens.
Loss of the target outside predicted robot occlusion stops approach/lowering;
expected occlusion is displayed, requiring operator observation (including wrist
video). It does not provide general obstacle avoidance or self-collision checking.
Other objects, hands and cables must remain clear of the supervised path. Existing
approximate calibration is unchanged. Session images/plans/logs are saved beneath
`captures/real_grasp/`; latest feedback is `calibration/grasp_live.json`.
