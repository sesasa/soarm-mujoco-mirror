"""Reverse the approach with the jaws held closed and a carried-object proxy."""
import numpy as np
from itertools import product
from simulated_reach import ReachSolver,trajectory

def append_return(reports,checker,model,region,up):
    approach=next(r for r in reports if r['stage']=='approach')['plan']
    lower=next(r for r in reports if r['stage']=='lower')['plan']
    closed=np.array(reports[-1]['plan']['goal']);q=closed.copy();solver=ReachSolver(model);body=model.body('gripper').id
    # Provisional 20 mm packet height; footprint comes from calibrated camera.
    center=np.array(region[0])+up*.01;size=np.array(region[1]);size[2]=.01
    corners=np.array(list(product([-1,1],repeat=3)))*size
    world=corners@np.asarray(region[2]).T+center
    solver.position(closed);origin=solver.data.xpos[body].copy();rotation=solver.data.xmat[body].reshape(3,3).copy()
    attached=(world-origin)@rotation
    for stage,end,speed in [('withdraw',lower['start'],1.5),('return',approach['start'],6.)]:
        selected=None
        for wrist_degrees in ([0,5,10,15,20,25,30] if stage=='return' else [0]):
            goal=np.array(end).copy();goal[5]=closed[5];goal[3]-=np.deg2rad(wrist_degrees)
            if not model.jnt_range[model.joint('wrist_flex').id,0]<=goal[3]<=model.jnt_range[model.joint('wrist_flex').id,1]:continue
            plan={'start':q,'goal':goal,'duration':max(4.,1.875*max(abs(goal-q))/np.deg2rad(speed))}
            clearance=checker.path(plan,161)
            if clearance['path_minimum_m']<.005:continue
            minimum=float('inf')
            for t in np.linspace(0,plan['duration'],161):
                sample=trajectory(plan,t);solver.position(sample)
                pts=attached@solver.data.xmat[body].reshape(3,3).T+solver.data.xpos[body]
                minimum=min(minimum,float(np.min((pts-checker.point)@up)))
            if minimum<(-.001 if stage=='withdraw' else .003):continue
            selected=(goal,plan,clearance,minimum,wrist_degrees);break
        if selected is None:raise ValueError(f'{stage}: no return path keeps the held packet clear of the table')
        goal,plan,clearance,minimum,wrist_degrees=selected
        reports.append({'stage':stage,'plan':{'start':q.tolist(),'goal':goal.tolist(),'duration':plan['duration'],'hold_gripper':True},'clearance':clearance,'carried_object_minimum_m':minimum,'assumed_object_height_m':.02,'return_wrist_adjustment_deg':wrist_degrees})
        q=goal
