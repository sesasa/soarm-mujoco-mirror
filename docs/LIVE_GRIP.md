# Live reach and grip — one object, real hardware

Start `.venv/bin/python src/adaptive_grip.py` after releasing torque and closing the previous camera/controller window. The new window starts torque-off. It connects to the physical SO-101 through `src/control_worker.py --direct --visual-servo`; this is not a simulation-only viewer.

1. **A** holds the arm at its measured position.
2. Click the object in the left camera panel if more than one is visible.
3. **P** prepares the initial reach and opening once.
4. **G** opens/prepares, follows the selected object, then descends and grips after alignment and stability.
5. **Space** stops and holds. **X** releases torque after the arm/object are supported.

There is no basket detection, placement, lift, batch or return-to-sitting stage. The final state holds the object in place.

The window shows the live overhead view and a lightweight projection from measured joints. Full 3D rendering was removed from this control loop because it delayed each update by about 300 ms on this desktop. With torque off, the revised live camera processing measured roughly 45 ms; this is not a measurement of end-to-end moving-robot latency.

## Live loop

- A capture thread continuously reads the overhead camera and retains only the latest delivered frame.
- The UI estimates the selected footprint/appearance and updates a warm-started local IK solution, aiming for about 10 updates/second. The prior solution is the next seed; the expensive multi-branch solve runs only for initial P preparation.
- Fresh angle goals go to the real motor worker. Its existing nominal 30 Hz loop follows them with a 20-degree/second joint speed cap and 60-degree/second-squared acceleration limit. This is a best-effort desktop loop, not a hard-real-time guarantee.
- The worker checks the actual incremental command against the table/holder geometry each tick, along with joint limits and tracking feedback. These remain sampled checks and do not cover cables, other objects or complete self-collision.
- Camera-target timestamps prevent heartbeats alone from keeping an obsolete visual target active: updates older than 250 ms are refused and a target stream older than 300 ms stops/holds. User stop/release clears the stream.
- Missing or ambiguous target detection pauses following. A visible compatible target can resume it. A failed local IK update needs a new P plan rather than an unannounced wrist branch change.

After measured alignment within 4 mm of the model target and 3 degrees of the desired arm angles, plus target stability, the app holds briefly to prepare the checked descent and closure. The target must remain visible/stationary during descent; a detected change stops the attempt. This is continuous adaptation during reaching, followed by a committed final grasp, not interception of a continuously moving object during contact.

The existing empirical 20 mm camera-left correction, 40 mm height assumption and 25 mm descent are retained. The packet closing gap remains 35 mm; a selected short box uses its estimated width minus 4 mm (minimum 5 mm). Grasp pressure is not force-calibrated. Wrist video calibration is still unaccepted, so it does not supply automated corrections.

## Validation and limits

`src/verify_live_grip.py` uses an archived box scene, shifts the target smoothly 10 mm over four seconds, then holds it still. It runs the streaming controller with ideal feedback at 30 Hz and warm IK at 10 Hz, followed by descent and closure. No camera or motor connection is opened.

Recorded numerical test: 200 IK updates, median 1.01 ms / 95th percentile 1.78 ms per update; worker geometry checks median 1.26 ms / 95th percentile 2.07 ms. Final model hover error was 0.40 mm. All streaming commands and both grip stages passed. These do not measure live camera latency, actual serial-loop frequency, physical accuracy or grip success.

Evidence: `captures/live_grip_test/result.json`. Regression tests include changed target direction, bounded streaming speed/acceleration, stale/out-of-order images, explicit hardware-mode opt-in, segment/stream exclusion, and clearing the stream on hold/pause. The existing batch/grasp behavior is retained in its separate app.

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_live_grip tests.test_adaptive_reach tests.test_batch_grasp tests.test_direct_segment tests.test_replay_settling tests.test_table_clearance tests.test_control tests.test_playback_speed tests.test_plate_occlusion
MUJOCO_GL=egl .venv/bin/python src/verify_live_grip.py
```
