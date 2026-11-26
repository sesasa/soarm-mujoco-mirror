"""Geometry and streaming motion for one-object visual reach and grip.

No device access. Shared by the vision process, hardware worker and offline tests.
"""
import json,math
import xml.etree.ElementTree as ET
import mujoco,numpy as np
from scipy.optimize import least_squares,brentq
from command_mapping import ROOT
from simulated_reach import ReachSolver,fingertip_gap
from table_clearance import TableClearance
from reach_retarget import camera_left_offset


def load_model():
    root=ET.parse(ROOT/'assets/so101/so101.xml').getroot()
    root.find('compiler').set('meshdir',str(ROOT/'assets/so101/assets'))
    from wrist_mount_visual import apply_wrist_mount_visual
    apply_wrist_mount_visual(root)
    return mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


def table_frame():
    reg=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
    board=json.loads((ROOT/'captures/scene_changes/plane.json').read_text())
    transform=np.linalg.inv(np.array(reg['T_camera_from_base']))@np.array(board['T_camera_from_board'])
    up=transform[:3,2].copy()
    if up[2]<0:up=-up
    return transform[:3,3],up


class WarmGraspIK:
    """One warm-started branch per update; full search is only for initial planning."""
    def __init__(self,model,mapping,plane,profile,reference=None):
        self.solver=ReachSolver(model);self.solver.bounds=np.asarray(mapping.limits)
        self.mapping=mapping;self.plane=plane;self.profile=dict(profile)
        self.reference=reference or json.loads((ROOT/'calibration/reach_reference.json').read_text())
        if self.reference['joint_mapping']!=mapping.mapping:raise ValueError('Reach reference mapping changed')
        self.up=np.asarray(plane.base_from_board[:3,2]).copy()
        if self.up[2]<0:self.up=-self.up
        self.body=model.body('gripper').id
        self.seed=None;self.axis_sign=None

    def target(self,region):
        ref=self.reference
        q=np.asarray(ref['plan']['goal'],float)
        position=self.solver.position(q)
        center=np.asarray(region[0]);shift=center-np.asarray(ref['surface_center_base_m'])
        previous=ref.get('retarget',{})
        height=self.profile.get('height_m',.04)-previous.get('object_height_m',.03)
        offset=camera_left_offset(self.plane,center,self.profile.get('camera_left_offset_m',0.))
        return position+shift+self.up*height+offset-np.asarray(previous.get('grasp_offset_base_m',[0.,0.,0.]))

    def reset(self,goal,region):
        self.seed=np.asarray(goal,float).copy()
        self.solver.position(self.seed)
        axis=np.asarray(region[2])[:,np.argmax(region[1][:2])]
        self.axis_sign=1 if self.solver.data.xmat[self.body].reshape(3,3)[:,1]@axis>=0 else -1
        self.previous_axis=axis*self.axis_sign

    def solve(self,region):
        if self.seed is None:raise ValueError('Initial reach must be planned first')
        target=self.target(region)
        if self.profile.get('position_only_reach',False):
            bounds=self.solver.bounds.copy()
            self.solver.bounds[:5,0]=np.maximum(bounds[:5,0],self.seed[:5]-np.deg2rad(12))
            self.solver.bounds[:5,1]=np.minimum(bounds[:5,1],self.seed[:5]+np.deg2rad(12))
            try:fit=self.solver.solve(self.seed,target)
            finally:self.solver.bounds=bounds
            q=fit['goal'];self.mapping.check(q);self.seed=q.copy()
            return q,{'target':target.tolist(),'error_m':fit['error_m'],'evaluations':None}
        axis=np.asarray(region[2])[:,np.argmax(region[1][:2])].copy()
        axis-=self.up*np.dot(axis,self.up);axis/=np.linalg.norm(axis)
        if axis@self.previous_axis<0:axis=-axis
        seed=self.seed.copy();bounds=self.solver.bounds[:5].copy()
        # A fast local update must not silently jump to an unrelated wrist-roll branch.
        bounds[:,0]=np.maximum(bounds[:,0],seed[:5]-np.deg2rad(12))
        bounds[:,1]=np.minimum(bounds[:,1],seed[:5]+np.deg2rad(12))
        def residual(x):
            pos=self.solver.position(np.r_[x,seed[5]])
            r=self.solver.data.xmat[self.body].reshape(3,3)
            return np.r_[pos-target,.03*(r[:,1]-axis),.003*(r[:,2]-self.up)]
        fit=least_squares(residual,seed[:5],bounds=bounds.T,max_nfev=25,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=np.r_[fit.x,seed[5]];error=float(np.linalg.norm(self.solver.position(q)-target))
        r=self.solver.data.xmat[self.body].reshape(3,3)
        align=np.degrees(np.arccos(np.clip(r[:,1]@axis,-1,1)))
        tilt=np.degrees(np.arccos(np.clip(r[:,2]@self.up,-1,1)))
        if error>.003 or align>15 or tilt>30:
            raise ValueError(f'Local reach needs a new P plan: {error*1000:.1f} mm error, {align:.0f} deg alignment')
        self.mapping.check(q);self.seed=q.copy();self.previous_axis=axis
        return q,{'target':target.tolist(),'error_m':error,'evaluations':fit.nfev}


class JointServo:
    """Acceleration-limited target following; independent of the vision rate."""
    speed=np.deg2rad(20.)
    acceleration=np.deg2rad(60.)
    def __init__(self,q):
        self.q=np.asarray(q,float).copy();self.velocity=np.zeros(6);self.goal=self.q.copy();self.frame_time=None
    def update(self,goal,frame_time,now):
        goal=np.asarray(goal,float)
        if goal.shape!=(6,) or not np.isfinite(goal).all():raise ValueError('Invalid servo goal')
        if not math.isfinite(frame_time) or not 0<=now-frame_time<=.25:raise ValueError('Stale visual target')
        if self.frame_time is not None and frame_time<=self.frame_time:raise ValueError('Repeated/out-of-order camera observation')
        self.goal=goal.copy();self.frame_time=frame_time
    def step(self,dt,now,limits=None):
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Stale servo control interval')
        if self.frame_time is None or now-self.frame_time>.3:raise ValueError('Visual target stream lost; holding')
        delta=self.goal-self.q
        wanted=np.sign(delta)*np.minimum(self.speed,np.sqrt(2*self.acceleration*abs(delta)))
        velocity=self.velocity+np.clip(wanted-self.velocity,-self.acceleration*dt,self.acceleration*dt)
        step=velocity*dt
        # Stop exactly at a close target instead of oscillating around it.
        crossing=(step*delta>=0)&(abs(step)>=abs(delta))&(abs(self.velocity)<=self.acceleration*dt)
        step=np.where(crossing,delta,step);velocity=np.where(crossing,0.,velocity)
        self.q+=step;self.velocity=velocity
        if limits is not None:
            limits=np.asarray(limits,float)
            if limits.shape!=(6,2) or not np.isfinite(limits).all():raise ValueError('Invalid servo limits')
            if np.any(self.goal<limits[:,0]-1e-9) or np.any(self.goal>limits[:,1]+1e-9):raise ValueError('Servo target outside working limits')
            # The integrator may pass a nearby target while braking after an
            # update. It must never pass the controller's actual joint bounds.
            bounded=np.clip(self.q,limits[:,0],limits[:,1])
            self.velocity=np.where(bounded!=self.q,0.,self.velocity)
            self.q=bounded
        return self.q.copy()


class ServoClearance:
    """Worker-side checks on each actual incremental command, not an old preview."""
    def __init__(self,model=None):
        point,up=table_frame();self.checker=TableClearance(model if model is not None else load_model(),point,up)
    def check(self,start,goal):
        for f in (0.,.5,1.):
            report=self.checker.at(np.asarray(start)+(np.asarray(goal)-start)*f)
            if report['minimum_m']<.005 or report['gripper_holder_minimum_m']<.015:
                raise ValueError('Live approach too close to tabletop; holding')


def hover_settled(position_error_m,measured,goal,position_tolerance_m=.008):
    """Allow measured SO-101 settling error while retaining the joint gate."""
    measured=np.asarray(measured,float);goal=np.asarray(goal,float)
    return bool(measured.shape==(6,) and goal.shape==(6,)
                and np.isfinite(np.r_[measured,goal,position_error_m]).all()
                and 0<=position_error_m<position_tolerance_m
                and np.max(abs(measured[:5]-goal[:5]))<np.deg2rad(3))
