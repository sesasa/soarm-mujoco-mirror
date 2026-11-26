# Adaptive approach: first implementation

Run `.venv/bin/python src/mirror_grasp.py --adaptive` for the opt-in mode. Without that flag the previous workflow remains the default. A holds, P previews, G executes, Space cancels the entire sequence, X releases torque. As before, startup does not enable torque automatically.

The first version adapts to an object that is moved and then becomes stationary. It pauses and replans; it does not continuously intercept a moving object. The recommended long-term architecture is a slower global planner plus a faster local visual-servo loop, but the current full grasp planner takes seconds, and wrist optical registration is not accepted. This implementation uses the calibrated overhead view.

## What happens

1. A fresh plan binds the selected object's metric footprint and HSV appearance. Motion monitoring uses quicker change contours (120 ms persistence), while normal mode retains 500 ms.
2. During wrist clearance, ready and approach, displacement greater than 8 mm, long-axis rotation greater than 8 degrees, or a substantial size change triggers a pause. Missing detection for over 250 ms or ambiguous identity also pauses. These thresholds are engineering defaults, not independently measured accuracy bounds.
3. The worker freezes its current commanded pose and acknowledges a pause token. It stays torque-enabled; there is no automatic release/rearm or jump to a stale UI pose. This command freeze is not a jerk-limited braking trajectory.
4. Once the same object has been clearly stationary for 600 ms and the arm is within 3 degrees of its held command, a fresh plan uses the current object, plate and held arm pose. Continuously moving targets keep the arm waiting.
5. Late plans are checked against the newest target and arm command. Stale results are discarded and refreshed. During an adaptive first-object refresh, the unused second-object preview is omitted to reduce latency; the second object is still freshly planned after the first placement.
6. Before descent, a fresh stable observation is required. Motion or loss of visibility during descent stops the attempt for inspection; it does not automatically chase sideways close to the table. After closure, the picked target is not retargeted from its moving image.
7. Space, X, worker faults, missing feedback and the heartbeat watchdog cancel continuation. Pause acknowledgement has a 2-second deadline; waiting plus adaptive planning has a 30-second limit. No valid solution leaves the arm holding, rather than resuming the old trajectory.

A target moved before G requires a new P preview. Matching uses size, appearance and distance within 12 cm of the last matched location. Two similar objects, hands covering the target, or gripper occlusion can make tracking ambiguous. This version waits/stops in those cases. It does not infer a hidden object's motion. It does not provide moving-obstacle avoidance or adapt the plate continuously during transfer; the plate is refreshed when planning runs.

## Files

- `src/adaptive_reach.py`: pure observation association, stability, moved-target detection and stale-result checks.
- `src/mirror_grasp.py --adaptive`: optional monitoring and asynchronous pause/replan integration.
- `src/control_worker.py`: acknowledged `direct_pause` operation that cancels only the segment and holds its current command.
- `src/batch_grasp.py`: optional omission of speculative second-leg preview during adaptive refresh.
- `test_adaptive_reach.py`: target-motion, rotation, loss, ambiguity, cancellation, stale-plan and fake-bus pause tests.
- `src/verify_adaptive_reach.py`: offline saved-scene test that moves the box synthetically, interrupts a simulated reach, replans and replays all replacement stages through the trajectory validators.

## Validation

The offline test uses `captures/mirror_grasp/1789857210115922940/planning_5/`. A synthetic 10-pixel leftward box movement corresponded to about 10.9 mm in the estimated table plane. The original approach was interrupted halfway through with ideal joint feedback. The old-target result was rejected; all eight replacement stages passed trajectory replay. In the recorded run, replanning took about 1.64 seconds. This is a single model/saved-image case, not a live responsiveness benchmark or physical validation.

Evidence: `captures/adaptive_reach_test/result.json`, original and synthetically changed images, and `replacement_plan.json`. No hardware movement or live camera capture was used for implementation tests. The normal gripper setting remains the restored width-minus-4-mm rule for the second box.

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_adaptive_reach tests.test_batch_grasp tests.test_direct_segment tests.test_replay_settling tests.test_table_clearance tests.test_control tests.test_playback_speed tests.test_plate_occlusion
MUJOCO_GL=egl .venv/bin/python src/verify_adaptive_reach.py
```

For a future continuous visual-servo loop, update a short-horizon Cartesian target from fresh observations, warm-start local IK, enforce velocity/acceleration continuity, and keep a global fallback for larger changes. That needs separate latency, occlusion and physical tracking validation. IK memory helps computation but cannot replace fresh observation.

Architecture reference: [MoveIt Hybrid Planning](https://moveit.picknik.ai/main/doc/concepts/hybrid_planning/hybrid_planning.html) describes combining slower global planning with faster local feedback; [MoveIt Servo](https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html) documents Cartesian servoing. These informed the recommendation; no ROS/MoveIt dependency was added.
