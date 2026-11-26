"""A configurable local descent, jaw closure and optional vertical lift from a proven reach."""
import numpy as np
from scipy.optimize import brentq,least_squares
from scipy.spatial.transform import Rotation
from simulated_reach import ReachSolver,fingertip_gap,trajectory
from table_clearance import TableClearance

def build_tail(model,mapping,plane,approach,closing_gap=.020,lift_height=.02,descent_m=.015):
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    solver=ReachSolver(model);q=np.array(approach,float);reports=[];body=model.body('gripper').id
    if not np.isfinite(descent_m) or descent_m<=0:raise ValueError('Invalid grasp descent')
    stages=[('lower',-float(descent_m),1.5),('close',0.,2.5)]
    if lift_height>0:stages.append(('lift',lift_height,1.5))
    for stage,dz,speed in stages:
        start=q.copy();position=solver.position(start);rotation=solver.data.xmat[body].reshape(3,3).copy();details={}
        if stage=='close':
            def gap(angle):
                candidate=start.copy();candidate[5]=angle;return fingertip_gap(solver,candidate)-closing_gap
            q=start.copy();q[5]=min(q[5],brentq(gap,mapping.limits[5][0],min(1.,mapping.limits[5][1])))
            details={'commanded_gap_m':closing_gap,'grip_force_calibrated':False}
        else:
            solver.bounds=np.array(mapping.limits)
            excursion=np.deg2rad(35 if stage=='lift' else 20)
            solver.bounds[:5,0]=np.maximum(solver.bounds[:5,0],start[:5]-excursion)
            solver.bounds[:5,1]=np.minimum(solver.bounds[:5,1],start[:5]+excursion)
            solved=solver.solve(start,position+up*dz);q=solved['goal'];q[5]=start[5]
            solver.position(q);change=float(np.rad2deg(Rotation.from_matrix(rotation.T@solver.data.xmat[body].reshape(3,3)).magnitude()))
            if change>10:
                target=position+up*dz
                # Prefer the existing gripping orientation when position-only IK twists the wrist.
                for weight in (.01,.02,.03):
                    def residual(x):
                        pos=solver.position(np.r_[x,start[5]])
                        actual=solver.data.xmat[body].reshape(3,3)
                        return np.r_[pos-target,weight*Rotation.from_matrix(rotation.T@actual).as_rotvec()]
                    fit=least_squares(residual,q[:5],bounds=solver.bounds[:5].T,max_nfev=250,gtol=1e-10,ftol=1e-10,xtol=1e-10)
                    candidate=np.r_[fit.x,start[5]];error=float(np.linalg.norm(solver.position(candidate)-target))
                    angle=float(np.rad2deg(Rotation.from_matrix(rotation.T@solver.data.xmat[body].reshape(3,3)).magnitude()))
                    if error<=.002 and angle<=10:
                        q=candidate;change=angle;solved['error_m']=error;break
                else:raise ValueError(f'{stage}: cannot keep grip orientation within limits')
            details={'vertical_travel_m':dz,'position_error_m':solved['error_m'],'orientation_change_deg':change}
        mapping.check(q)
        plan={'start':start,'goal':q,'duration':max(4.,1.875*max(abs(q-start))/np.deg2rad(speed))}
        clearance=checker.path(plan,161)
        if clearance['path_minimum_m']<.005 or clearance['gripper_holder_path_minimum_m']<.005:
            raise ValueError(f"{stage}: table clearance {clearance['path_minimum_m']*1000:.1f} mm; gripper/holder {clearance['gripper_holder_path_minimum_m']*1000:.1f} mm")
        if stage!='close':
            for t in np.linspace(0,plan['duration'],81):
                delta=solver.position(trajectory(plan,t))-position
                if np.linalg.norm(delta-up*np.dot(delta,up))>.004:raise ValueError(f'{stage}: path moves sideways more than 4 mm')
        playback={'start':start.tolist(),'goal':q.tolist(),'duration':plan['duration']}
        if stage=='lower':playback.update(speedup_until_s=plan['duration'],speedup_factor=2.,playback_duration=plan['duration']/2)
        if stage=='close':playback.update(speedup_until_s=plan['duration'],speedup_factor=4.,playback_duration=plan['duration']/4,hold_gripper=True)
        reports.append({'stage':stage,'plan':playback,'clearance':clearance,**details})
    return reports
