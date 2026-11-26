# Named-object reach

Start `.venv/bin/python src/adaptive_grip.py --named-reach`.

Place separated objects in the overhead workspace. Click a detected outline,
press N, type a name (e.g. highlighter or matchbox), and press Enter. Repeat for
other objects. Labels belong to this session and track appearance, dimensions
and nearby location using the existing TargetWatch. They are manually taught
names, not automatic semantic recognition. No neural network or language model
is used. Large jumps, occlusion and similar objects may require relabeling.

Press A to enable holding. Press Enter to open the command field, type
`reach for the highlighter`, and press Enter again. The current tracked location
is selected and planning starts. G executes the prepared reach. The arm follows
the target until aligned and then holds at hover: no descent, closure, basket or
return. Another command can select another object after reaching. If torque is
off when a command is submitted, selection works; A then P prepares the reach.

Space stops and X releases torque outside text entry. Text entry is available
only while stationary; Esc cancels text entry. Support the arm before releasing.
If recognition is missing or ambiguous, separate the objects or relabel them.

This reuses existing tabletop/arm clearance checks; it does not add complete
collision avoidance against all other scene objects. Keep reach paths clear.
Tests cover prompt parsing, updated positions, two different appearances,
unknown names, missing/ambiguous objects and stale observations.
