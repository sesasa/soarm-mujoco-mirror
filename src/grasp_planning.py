"""Model checks for the explicitly requested real grasp stages."""
import time
import numpy as np
from scipy.optimize import brentq
from simulated_reach import ReachSolver,hover_point,fingertip_gap
from table_clearance import TableClearance
from grasp_motion import GraspMotion,STAGES,STAGE_HEIGHTS

def make_grasp_report(model,mapping,plane,state,region,stage,opening):
    start=np.asarray(state['command'],float);mapping.check(start)
    solver=ReachSolver(model);solver.bounds=np.asarray(mapping.limits)
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    surface,size,rotation=region;axis=np.asarray(rotation)[:,np.argmax(size[:2])]
    height=STAGE_HEIGHTS.get(stage)
    if height is not None:
        plan=solver.solve_grasp(start,hover_point(surface,up,height),up,axis)
        if plan['start_adjustment_deg']>.001:raise ValueError('Starting command outside planning limits')
        plan['start']=start.copy();plan['goal'][5]=start[5]
    else:
        goal=start.copy()
        if stage=='close':
            def gap(angle):
                q=start.copy();q[5]=angle;return fingertip_gap(solver,q)-.025
            goal[5]=max(mapping.limits[5][0],start[5]-np.deg2rad(2)) if state.get('grasp_stage')=='close' else min(start[5],brentq(gap,mapping.limits[5][0],min(1.,mapping.limits[5][1])))
        elif stage=='open':goal[5]=max(start[5],opening)
        else:raise ValueError('Unknown stage')
        plan={'start':start.copy(),'goal':goal}
    plan['duration']=max(4.,1.875*np.max(np.abs(plan['goal']-start))/np.deg2rad(STAGES[stage][1]))
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    clearance=checker.path(plan,samples=161)
    initial=checker.at(np.array(state['q']))
    if initial['minimum_m']<.005:raise ValueError('Measured pose too close to modeled table')
    if stage in ('lower','lift'):
        # Verify that a joint-space segment remains over the object during descent/lift.
        from simulated_reach import trajectory
        for t in np.linspace(0,plan['duration'],81):
            point=solver.position(trajectory(plan,t));delta=point-surface;side=delta-up*np.dot(delta,up)
            if np.linalg.norm(side)>.004:raise ValueError('Vertical stage drifts more than 4 mm sideways')
    result={'mode':'supervised_grasp','stage':stage,'prepared_at_unix':time.time(),
            'joint_mapping':mapping.mapping,'reference_height_m':height,
            'plan':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()},
            'clearance':clearance,'initial_clearance':initial,'surface_center_base_m':np.asarray(surface).tolist(),
            'region':[np.asarray(v).tolist() for v in region],
            'original_opening_rad':opening,'hardware_commands':False,
            'note':'Supervised experimental grasp; model gap is approximate, contact/force not measured.'}
    GraspMotion(result,mapping,state['command'],state['q'],state.get('grasp_stage'),time.time())
    return result
