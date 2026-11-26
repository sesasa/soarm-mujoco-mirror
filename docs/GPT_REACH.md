# GPT object selection and calibrated reaching

Run `.venv/bin/python src/adaptive_grip.py --gpt-reach` with OPENAI_API_KEY in
the process environment. OPENAI_VISION_MODEL optionally overrides gpt-4.1-mini.
No key is stored in source or logs. The Python standard-library HTTPS client
calls the OpenAI Responses API; no additional SDK dependency is required.

A enables holding. Enter opens text entry; type `matchbox` or
`reach for the pink highlighter` and press Enter. GPT selects a currently
detected object; its contour is highlighted and the reach is planned. G executes
the reach and holds above it. With torque off, recognition still works; use
A then P afterward. No manual naming is required. This is typed input, not
microphone/speech input. Space stops and X releases outside text entry.

Each request sends an overhead JPEG with numbered OpenCV contours and the typed
description to api.openai.com. Responses use a strict JSON schema and store=false.
A model selection returns only a candidate ID and description, never a robot
position. Not-found, ambiguous, invalid, refused, incomplete and failed requests
cannot launch motion. Network work runs separately from camera/heartbeat work.
The candidate identities are tracked locally while the request is in flight.
Expired, canceled and no-longer-stable selections are discarded. G remains the
explicit execution step after planning. Robot feedback, joint bounds and sampled
table-clearance checks remain active. Recognition can still be wrong: inspect
the highlighted outline before G.

The detector must provide a separate contour for the requested object. GPT cannot
select an object omitted or merged by that detector. Unfamiliar descriptions may
fail. This does not infer height, grasp force or additional obstacle avoidance;
it uses the existing configured hover geometry and position-only reach mode.

OpenAI documentation used:
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs
https://developers.openai.com/api/docs/models/gpt-4.1-mini

## Grasp mode

Run `.venv/bin/python src/adaptive_grip.py --gpt-grip` for recognition, aligned reach,
descent, closure and hold. G now executes the grasp, not just reaching.
The existing natural-grasp IK evaluates both jaw-axis signs and wrist-roll
branches. Position-only reaching is disabled for this mode because reaching a
point does not ensure that the jaws straddle the object. Full tail feasibility
is checked during initial planning, then recalculated at the actual hover pose.
Missing/moving targets stop descent or prevent closure as in the original grip
controller. No lift, basket or return is added.

Closing gap is measured contour width minus 4 mm (minimum 5 mm); opening is width
plus the configured 8 mm margin. This is a position command, not force control.
Existing profile assumptions remain: 40 mm object height and 25 mm descent.
GPT supplies identity only, not height or safe gripping force. Different-height
objects may require adjustment. Joint, tracking and table-clearance limits remain.

`src/verify_gpt_grip.py` replays the archived selected matchbox through prefix,
600 streaming updates, descent and closure with ideal feedback and no hardware.

Final-grasp occlusion: after a fresh, stable hover has been confirmed, GPT grip
mode latches that target. Object disappearance during the pause, tail planning,
descent or transition to closure no longer cancels the grasp. Visible movement
or ambiguous matching still stops it. The planning latch expires after 15 s;
once accepted, execution is bounded by the planned playback time plus endpoint
settling allowances. Camera-stream loss and motor watchdogs still stop motion.
Occlusion before the confirmed hover still pauses live following. This feature
cannot detect an object moving while completely hidden.

Final approach now also tolerates occlusion: a target observed stable during
approach is remembered for up to 12 seconds. If it disappears within 6 cm of the
hover target, the controller finishes the remaining reach at that fixed location,
confirms the measured arm has arrived, then proceeds through the committed
descent/closure. Visible motion or ambiguous identity clears approach memory.
Loss farther away still pauses. No new target is inferred from the hidden image.

## GPT pick and place

Run `.venv/bin/python src/adaptive_grip.py --gpt-place`.
A holds; Enter opens the prompt; type `matchbox` and Enter to identify/plan;
G executes reach, descent, closure, lift, transfer, release and return to sitting.
Use the object name as the prompt: the destination is the configured plate.

The plate reference now records the reddish plate with a user-confirmed 2 cm
rim height. The yellow reference is backed up under captures/gpt_place.
plate_detection supports explicit red and yellow HSV/ellipse masks. The new
inner opening was transferred from the same-shape yellow plate reference using
fitted outer ellipses, then projected onto the calibrated rim-height plane.
This is approximate camera-derived geometry, not a new physical measurement.

The initial P plan measures the plate from that camera frame and preflights the
complete tail. At hover, it measures the plate again and rebuilds descent,
closure, payload lift, transfer, release and sitting return from the current
command. Old config/basket.json positions are never a detection fallback.
Actual plate motion after the final plan is built is not tracked continuously.

Existing append_basket models the object as attached rigidly to the gripper,
checks carried-object/table/rim clearance and fits its footprint inside the
opening before release. Release is above the rim as in the previous plate mode;
there is no force-controlled set-down or sensor-confirmed possession. The
existing final-approach/grasp occlusion behavior is retained. After closure,
loss of the original object contour does not cancel the transfer.

src/verify_gpt_place.py exercises a saved matchbox observation and the fresh red
plate image through preflight, live-IK/worker-style ideal-feedback replay, all
placement stages and return. Evidence: captures/gpt_place/offline_result.json.
It does not establish physical grasp or placement success.

## Wrist visibility continuation (not wrist visual servoing)

With --wrist-view, an overhead-occlusion pause now remains in TARGET_HIDDEN rather
than repeatedly sending pause commands. Drag a rectangle around a textured patch
of the intended object in the wrist panel (e.g. highlighter lettering), then V.
This is an operator-confirmed association with the previously planned object.
The arm must be within 6 cm of its planned hover. It follows that stored target;
pixels are NOT converted to arm corrections. A fresh LK/RANSAC-tracked patch
must remain visible through approach and descent, within a 20 s continuation
window. Patch loss, expiration, or visible overhead displacement holds motion.
Once closing/contact begins, patch visibility is no longer required for carrying
the object to the plate. This does not prove alignment, detect all object motion
relative to the moving camera, or provide force/grasp confirmation. Wrist-based
position/orientation correction still requires an image-to-motion calibration.

Hover settling update: GPT grip/place accepts model tip error below 8 mm together
with all five arm joint errors below 3 degrees, sustained for the existing 0.8 s.
The previous 4 mm tip gate stalled a recorded case with approximately 6.3 mm
error despite only 1.4 degrees of maximum joint discrepancy. This accepts
settling error; it does not compensate physical alignment. Position-only reach
retains its 4 mm threshold. Clearance and descent checks remain unchanged.
Wrist timeout and patch-loss messages are now distinct. Source: hover_settled()
in src/live_grip.py; regression coverage in test_hover_settling.py.

## Fixed preplanned pick and place (stationary scene)

Run `.venv/bin/python src/adaptive_grip.py --gpt-fixed-place`.
This implies GPT selection and plate placement, and disables wrist modes.
The complete list (clear wrist if needed, open/ready, approach, lower, close,
lift, transfer, release, return sitting) is calculated and checked before G.
Object and plate geometry comes from the current planning observation. G still
checks that the selected object and initial arm command match the prepared plan.
Once started, FIXED replays every segment with motor-feedback settling checks;
object disappearance or association changes do not trigger pauses or replanning.
There is no wrist handoff and no live target-following stage. Keep the scene
stationary after planning. Camera-stream failure, motor faults, tracking limits,
Space and X still apply. This does not verify actual object possession.

Regression evidence: captures/gpt_place/fixed_replay_result.json replays the
latest saved full highlighter plan through 1176 ideal-feedback controller ticks,
all nine stages, without vision queries or any hardware commands.

Fixed-plan speed update: accelerate_fixed_playback() now applies execution_rate=4 to every stage, relative to the original fixed-mode playback. Geometric paths and the additional 10 mm closing-gap reduction remain unchanged. Planning and measured endpoint-settling time are not accelerated. Four-times playback passed ideal-feedback replay (captures/gpt_place/fixed_speed4_replay.json); physical tracking at this speed is not yet validated. Motor tracking-error checks remain enabled.

## One-command execution

In fixed mode, press Enter, type `pick and place the highlighter`, and submit with
Enter. This explicit prefix authorizes automatic holding enable, GPT selection,
full preplanning and execution without A/P/G. Optional `please` and a trailing
`in the plate`/`on the plate` are accepted. The destination remains the configured
plate. Execution intent is parsed locally from the typed command; GPT only
selects the object. A bare object description preserves the manual workflow.

The UI waits for ACTIVE feedback after arming (2 s timeout), still checks the
current target and command before execution, rejects canceled/stale results,
and allows another explicit command from the completed sitting state. Space
stops and X releases outside text entry; Esc cancels text entry. No execution
is triggered by merely opening the window. Planning or recognition failure does
not start a partial sequence. Current stage speeds and gaps are unchanged.

## Voice input

Run `.venv/bin/python src/adaptive_grip.py --gpt-fixed-place --voice`.
While the window shows Listening, say `pick and place the highlighter`.
ALSA arecord captures default-microphone 16 kHz mono PCM. Local energy endpointing
collects a short utterance after at least 0.4 s of voice and 0.8 s of quiet;
clips exceeding 8 s are discarded. A background urllib request uploads a WAV
to OpenAI /v1/audio/transcriptions using gpt-4o-mini-transcribe, with English
language and no command-biasing transcription prompt. Only an explicit
pick-and-place prefix is accepted by the existing local parser. GPT vision then
selects an object and the fixed sequence is planned/executed automatically.

Listening is enabled only while idle, with no text entry, planning or execution
in progress. M toggles microphone mute. Space stops; X releases; both also mute
voice until M re-enables it. Voice is not an emergency-stop channel. Old responses
from before mute/cancellation are discarded using an epoch token. Accepted
transcripts are logged locally; audio clips are not written to project files.
Voice activation is energy based and may include background speech; it is not
speaker authentication. Recognition may mishear words. Watch the displayed
selection and keep Space available.

Environment overrides: SOARM_MIC_DEVICE (default: default), SOARM_VOICE_RMS
(default: 350 PCM units), OPENAI_TRANSCRIBE_MODEL (default: gpt-4o-mini-transcribe).
No added Python audio dependency is needed. Microphone capture and mocked API,
endpointing and stale-result tests passed; live spoken-command accuracy still
requires user testing. Official API source:
https://developers.openai.com/api/docs/guides/speech-to-text
