"""Prepare one bounded commissioning step; no hardware I/O."""
import numpy as np

def prepare_step(report, measured, mapping, clearance):
    start=np.asarray(measured,float)
    mapping.check(start)
    original=np.asarray(report['plan']['start'],float)
    goal=np.asarray(report['plan']['goal'],float)
    if original.shape!=(6,) or goal.shape!=(6,) or not np.isfinite(np.r_[original,goal]).all():
        raise ValueError('Invalid approach plan')
    if np.max(np.abs(start-original))>np.deg2rad(.5):
        raise ValueError('Arm moved since planning; prepare a new step')
    delta=goal-start;delta[5]=0.
    scale=min(1.,np.deg2rad(2.5)/max(np.max(np.abs(delta)),1e-12))
    target=start+scale*delta;mapping.check(target)
    plan={'start':start,'goal':target,'duration':max(4.,1.875*np.max(np.abs(target-start))/np.deg2rad(1.))}
    check=clearance.path(plan)
    if check['path_minimum_m']<.005 or check['gripper_holder_path_minimum_m']<.05:
        raise ValueError('Step refused: insufficient modeled table clearance')
    plan['clearance']=check
    return plan

def validate_step(plan, measured, limits):
    start=np.asarray(plan['start'],float);goal=np.asarray(plan['goal'],float)
    measured=np.asarray(measured,float);limits=np.asarray(limits,float)
    duration=float(plan['duration'])
    if start.shape!=(6,) or goal.shape!=(6,) or measured.shape!=(6,) or limits.shape!=(6,2):
        raise ValueError('Invalid step dimensions')
    if not np.isfinite(np.r_[start,goal,measured,limits.ravel(),duration]).all():
        raise ValueError('Invalid step values')
    if np.max(np.abs(start-measured))>np.deg2rad(.5):raise ValueError('Arm moved since planning; replan first')
    if np.max(np.abs(goal-start))>np.deg2rad(3)+1e-9 or abs(goal[5]-start[5])>1e-9:
        raise ValueError('Step must stay within 3 degrees and leave jaw unchanged')
    if duration<4 or 1.875*np.max(np.abs(goal-start))/duration>np.deg2rad(1)+1e-9:
        raise ValueError('Step exceeds 1 degree/second')
    if np.any(np.minimum(start,goal)<limits[:,0]) or np.any(np.maximum(start,goal)>limits[:,1]):
        raise ValueError('Step exceeds active commissioning window')
    return {'start':start,'goal':goal,'duration':duration}
