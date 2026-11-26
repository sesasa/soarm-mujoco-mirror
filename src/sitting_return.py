"""Return the empty, open gripper to the calibrated sitting working pose."""
import json
import numpy as np
from command_mapping import ROOT
from simulated_reach import trajectory


def build_sitting_return(start,mapping,checker,basket):
    saved=json.loads((ROOT/'calibration/seated_start.json').read_text())
    if saved['joint_mapping']!=mapping.mapping:raise ValueError('Sitting pose calibration changed')
    # The torque-off resting pose lies outside working limits. Its recorded
    # inward goal is the matching seated pose available under position control.
    goal=np.asarray(saved['goal'],float);goal[5]=start[5]
    mapping.check(goal)
    duration=max(4.,1.875*float(max(abs(goal-start)))/np.deg2rad(6))
    plan={'start':np.asarray(start),'goal':goal,'duration':duration}
    clearance=checker.path(plan,161)
    if clearance['path_minimum_m']<.005:raise ValueError('Sitting return crosses tabletop')
    for t in np.linspace(0,duration,161):basket.check_robot(trajectory(plan,t))
    return {'stage':'return_sitting','plan':{'start':list(start),'goal':goal.tolist(),'duration':duration,
            'speedup_until_s':duration,'speedup_factor':4.,'playback_duration':duration/4,'hold_gripper':False},
            'clearance':clearance,'source':'calibration/seated_start.json: goal','gripper_remains_open':True}
