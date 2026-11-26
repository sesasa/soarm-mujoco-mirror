"""Kinematic hover previews. No hardware I/O or motion commands."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

class ReachSolver:
    def __init__(self, model):
        self.model=model
        self.data=mujoco.MjData(model)
        self.addresses=np.array([model.joint(n).qposadr[0] for n in JOINTS])
        self.dofs=np.array([model.joint(n).dofadr[0] for n in JOINTS[:5]])
        ids=[model.joint(n).id for n in JOINTS]
        self.bounds=model.jnt_range[ids].copy()
        self.site=model.site('gripperframe').id

    def position(self,q):
        self.data.qpos[self.addresses]=q
        mujoco.mj_forward(self.model,self.data)
        return self.data.site_xpos[self.site].copy()

    def solve(self, start, target):
        start=np.asarray(start,float);target=np.asarray(target,float)
        if start.shape!=(6,) or target.shape!=(3,) or not np.isfinite(np.r_[start,target]).all():
            raise ValueError('Invalid start or target')
        limited=np.clip(start,self.bounds[:,0]+1e-6,self.bounds[:,1]-1e-6)
        def residual(x):
            return np.r_[self.position(np.r_[x,limited[5]])-target,.0002*(x-limited[:5])]
        def jacobian(x):
            self.position(np.r_[x,limited[5]])
            jac=np.zeros((3,self.model.nv))
            mujoco.mj_jacSite(self.model,self.data,jac,None,self.site)
            return np.vstack([jac[:,self.dofs],.0002*np.eye(5)])
        result=least_squares(residual,limited[:5],jac=jacobian,bounds=(self.bounds[:5,0],self.bounds[:5,1]),max_nfev=180,gtol=1e-10,ftol=1e-10,xtol=1e-10)
        goal=np.r_[result.x,limited[5]]
        error=float(np.linalg.norm(self.position(goal)-target))
        if error>.002:raise ValueError(f'No reach found within limits (position error {error*1000:.1f} mm)')
        return {'start':limited,'goal':goal,'target':target,'error_m':error,
                'start_adjustment_deg':float(np.max(np.abs(limited-start))*180/np.pi),
                'duration':max(2.,float(np.max(np.abs(goal-limited)))*1.875/np.deg2rad(25))}

    def solve_grasp(self, start, target, table_normal, object_axis):
        """Approximate downward hover: jaw closing direction across the long axis.

        Five arm joints cannot satisfy every six-dimensional pose. Try both
        equivalent 180-degree jaw orientations and report residuals explicitly.
        """
        start=np.asarray(start,float);target=np.asarray(target,float)
        up=np.asarray(table_normal,float);along=np.asarray(object_axis,float)
        if (start.shape!=(6,) or target.shape!=(3,) or up.shape!=(3,) or along.shape!=(3,)
                or not np.isfinite(np.r_[start,target,up,along]).all() or np.linalg.norm(up)<1e-8):
            raise ValueError('Invalid grasp target or axes')
        up=up/np.linalg.norm(up)
        if up[2]<0:up=-up
        along=along-up*np.dot(along,up)
        if np.linalg.norm(along)<1e-8:raise ValueError('Object direction must lie along tabletop')
        along/=np.linalg.norm(along)
        limited=np.clip(start,self.bounds[:,0]+1e-6,self.bounds[:,1]-1e-6)
        body=self.model.body('gripper').id
        seeds=[limited[:5],(self.bounds[:5,0]+self.bounds[:5,1])/2]
        try:seeds.append(self.solve(start,target)['goal'][:5])
        except ValueError:pass
        candidates=[]
        for sign in (1,-1):
            y=sign*along
            desired=np.column_stack([np.cross(y,up),y,up])
            def residual(x):
                position=self.position(np.r_[x,limited[5]])
                actual=self.data.xmat[body].reshape(3,3)
                return np.r_[position-target,.03*Rotation.from_matrix(desired.T@actual).as_rotvec()]
            for seed in seeds:
                result=least_squares(residual,seed,bounds=self.bounds[:5].T,max_nfev=250,
                                     gtol=1e-10,ftol=1e-10,xtol=1e-10)
                goal=np.r_[result.x,limited[5]]
                error=float(np.linalg.norm(self.position(goal)-target))
                actual=self.data.xmat[body].reshape(3,3)
                tilt=float(np.rad2deg(np.arccos(np.clip(np.dot(actual[:,2],up),-1,1))))
                alignment=float(np.rad2deg(np.arccos(np.clip(abs(np.dot(actual[:,1],along)),0,1))))
                orientation=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(desired.T@actual).as_rotvec())))
                candidates.append((error,tilt,alignment,orientation,goal))
        valid=[c for c in candidates if c[0]<=.002 and c[1]<=10 and c[2]<=10 and c[3]<=12]
        if not valid:
            best=min(candidates,key=lambda c:c[0]/.002+c[3]/10)
            raise ValueError(f'No aligned hover found: {best[0]*1000:.1f} mm, tilt {best[1]:.1f} deg, alignment {best[2]:.1f} deg')
        error,tilt,alignment,orientation,goal=min(valid,key=lambda c:c[0]/.002+c[3]/10+.01*np.linalg.norm(c[4]-limited))
        return {'start':limited,'goal':goal,'target':target,'error_m':error,
                'downward_tilt_deg':tilt,'jaw_alignment_error_deg':alignment,
                'orientation_error_deg':orientation,'object_axis':along,'table_up':up,
                'mode':'aligned_hover','grasp_executed':False,
                'start_adjustment_deg':float(np.max(np.abs(limited-start))*180/np.pi),
                'duration':max(2.,float(np.max(np.abs(goal-limited)))*1.875/np.deg2rad(25))}


    def solve_natural_grasp(self,start,target,table_normal,object_axis,return_candidates=False):
        """Search both equivalent jaw orientations and both roll directions."""
        start=np.asarray(start,float);target=np.asarray(target,float);up=np.asarray(table_normal,float).copy();up/=np.linalg.norm(up)
        axis=np.asarray(object_axis,float).copy();axis-=up*np.dot(axis,up);axis/=np.linalg.norm(axis)
        reference=np.clip(start,self.bounds[:,0]+1e-6,self.bounds[:,1]-1e-6)
        body=self.model.body('gripper').id;seeds=[reference[:5],np.mean(self.bounds[:5],axis=1)]
        try:seeds.append(self.solve(reference,target)['goal'][:5])
        except ValueError:pass
        valid=[]
        for sign in (1,-1):
            for direction in (-1,1):
                bounds=self.bounds[:5].copy()
                if direction<0:bounds[4,1]=min(bounds[4,1],reference[4])
                else:bounds[4,0]=max(bounds[4,0],reference[4])
                if bounds[4,1]-bounds[4,0]<1e-5:continue
                for seed in seeds:
                    seed=np.clip(seed,bounds[:,0]+1e-7,bounds[:,1]-1e-7)
                    def residual(x):
                        pos=self.position(np.r_[x,reference[5]]);rot=self.data.xmat[body].reshape(3,3)
                        return np.r_[pos-target,.03*(rot[:,1]-sign*axis),.003*(rot[:,2]-up)]
                    fit=least_squares(residual,seed,bounds=bounds.T,max_nfev=250,gtol=1e-10,ftol=1e-10,xtol=1e-10)
                    goal=np.r_[fit.x,reference[5]];error=float(np.linalg.norm(self.position(goal)-target));rot=self.data.xmat[body].reshape(3,3)
                    align=float(np.rad2deg(np.arccos(np.clip(abs(rot[:,1]@axis),0,1))))
                    tilt=float(np.rad2deg(np.arccos(np.clip(rot[:,2]@up,-1,1))))
                    if error>.002 or align>15 or tilt>30:continue
                    if any(np.max(np.abs(goal-r['goal']))<np.deg2rad(.5) for r in valid):continue
                    motion=np.abs(goal[:5]-reference[:5]);score=2*motion[4]+.25*np.linalg.norm(motion)
                    valid.append({'goal':goal,'error_m':error,'jaw_alignment_error_deg':align,'downward_tilt_deg':tilt,
                                  'roll_change_deg':float(np.rad2deg(goal[4]-reference[4])),'jaw_orientation_sign':sign,'motion_cost':float(score)})
        if not valid:raise ValueError('Neither wrist-roll direction gives a reachable grasp at this target')
        valid.sort(key=lambda r:r['motion_cost'])
        return valid if return_candidates else valid[0]


def trajectory(plan,elapsed):
    t=float(np.clip(elapsed/plan['duration'],0,1))
    blend=10*t**3-15*t**4+6*t**5
    return plan['start']+blend*(plan['goal']-plan['start'])


def hover_point(surface, normal, height=.08):
    normal=np.asarray(normal,float)
    normal=normal/np.linalg.norm(normal)
    if normal[2]<0:normal=-normal
    return np.asarray(surface,float)+height*normal

class ReachProgress:
    """Pause virtual time on uncertain observations; never retarget the saved plan."""
    def __init__(self, region):
        center,size,rotation=region
        self.center=np.asarray(center).copy()
        self.size=np.sort(np.asarray(size)[:2])
        self.axis=np.asarray(rotation)[:,np.argmax(size[:2])].copy()
        self.elapsed=0.
        self.stable=0.
        self.paused=True
        self.message='Waiting for stable target'

    def update(self, regions, dt):
        matches=[]
        for index,(center,size,rotation) in enumerate(regions):
            ratios=np.sort(np.asarray(size)[:2])/self.size
            axis=np.asarray(rotation)[:,np.argmax(size[:2])]
            if (np.linalg.norm(np.asarray(center)-self.center)<=.015
                    and np.all((ratios>.65)&(ratios<1.45))
                    and abs(np.dot(axis,self.axis))>=np.cos(np.deg2rad(15))):
                matches.append(index)
        dt=float(np.clip(dt,0,.2))
        if len(matches)!=1:
            self.stable=0.;self.paused=True
            self.message='PAUSED: target missing, moved or ambiguous - waiting for original target'
            return
        self.stable+=dt
        self.paused=self.stable<.35
        self.message='PAUSED: confirming target' if self.paused else 'Simulation-only reach active | X: stop'
        if not self.paused:self.elapsed+=dt


def vertical_step(solver,previous,delta,clearance):
    """Plan a 1 cm virtual step with sampled alignment/clearance checks."""
    if previous.get('mode') not in ('aligned_hover','vertical_step','jaw_opening'):
        raise ValueError('Run G aligned hover before stepping vertically')
    if not np.isfinite(delta) or not np.isclose(abs(delta),.01):
        raise ValueError('Vertical preview steps must be 1 cm')
    up=np.asarray(previous['table_up'],float)
    surface=np.asarray(previous['surface'],float)
    height=float((np.asarray(previous['target'])-surface)@up)
    destination_height=height+delta
    if destination_height<.03-1e-6 or destination_height>.08+1e-6:
        raise ValueError('Preview height limit reached (3 to 8 cm above table)')
    destination=hover_point(surface,up,destination_height)
    start=np.asarray(previous['goal'],float)
    plan=solver.solve_grasp(start,destination,up,previous['object_axis'])
    if np.max(np.abs(plan['goal']-start))>np.deg2rad(20):
        raise ValueError('Vertical step blocked: large joint change')
    body=solver.model.body('gripper').id
    heights=[];lateral=[];tilts=[];alignments=[]
    for t in np.linspace(0,plan['duration'],41):
        point=solver.position(trajectory(plan,t));offset=point-surface
        heights.append(float(offset@up))
        lateral.append(float(np.linalg.norm(offset-up*(offset@up))))
        rotation=solver.data.xmat[body].reshape(3,3)
        tilts.append(float(np.degrees(np.arccos(np.clip(rotation[:,2]@up,-1,1)))))
        alignments.append(float(np.degrees(np.arccos(np.clip(abs(rotation[:,1]@plan['object_axis']),0,1)))))
    if max(lateral)>.002 or max(tilts)>8 or max(alignments)>8:
        raise ValueError('Vertical step blocked: path alignment outside preview limits')
    if np.min(np.diff(heights)*np.sign(delta))<-.0005:
        raise ValueError('Vertical step blocked: path reverses direction')
    report=clearance.path(plan)
    if report['path_minimum_m']<.005 or report['gripper_holder_path_minimum_m']<.01:
        raise ValueError('Vertical step blocked: insufficient sampled table clearance')
    if 'measured_object' in previous:plan['measured_object']=previous['measured_object']
    if 'jaw_opening' in previous:plan['jaw_opening']=previous['jaw_opening']
    plan.update(mode='vertical_step',surface=surface,height_m=destination_height,
                table_clearance=report,vertical_path_checks={
                    'samples':41,'maximum_lateral_error_m':max(lateral),
                    'maximum_downward_tilt_deg':max(tilts),'maximum_jaw_alignment_error_deg':max(alignments),
                    'minimum_reference_height_m':min(heights),'grasp_executed':False})
    return plan


def fingertip_gap(solver,q):
    """Approximate surface gap along gripper-local X using named fingertip spheres."""
    solver.position(q)
    model,data=solver.model,solver.data
    fixed=model.geom('fixed_jaw_sph_tip1').id
    moving=model.geom('moving_jaw_sph_tip1').id
    closing_axis=data.xmat[model.body('gripper').id].reshape(3,3)[:,0]
    return float((data.geom_xpos[moving]-data.geom_xpos[fixed])@closing_axis
                 -model.geom_size[fixed,0]-model.geom_size[moving,0])


def open_for_object(solver,previous,clearance,extra_gap=.010):
    """Open the virtual jaw, retaining the arm pose. No closing or contact."""
    from scipy.optimize import brentq
    if previous.get('mode') not in ('aligned_hover','vertical_step','jaw_opening'):
        raise ValueError('Run G aligned hover before opening the virtual jaw')
    obj=previous.get('measured_object')
    if not obj:raise ValueError('Select the measured highlighter and run G first')
    width=float(obj['width_m']);height=float(obj['height_m'])
    if not np.isfinite([width,height,extra_gap]).all() or min(width,height)<=0 or extra_gap<.005:
        raise ValueError('Invalid object dimensions or opening clearance')
    required=width+extra_gap
    start=np.asarray(previous['goal'],float).copy();goal=start.copy()
    initial=fingertip_gap(solver,start)
    if initial<required:
        # Find the first opening solution inside the joint limits, never close.
        angles=np.linspace(start[5],solver.bounds[5,1],100)
        def residual(angle):
            q=start.copy();q[5]=angle
            return fingertip_gap(solver,q)-required
        bracket=None
        for lo,hi in zip(angles[:-1],angles[1:]):
            if residual(lo)<=0<=residual(hi):bracket=(lo,hi);break
        if bracket is None:raise ValueError('Required opening exceeds model gripper range')
        goal[5]=brentq(residual,*bracket,xtol=1e-12)
    plan=dict(previous)
    plan.update(start=start,goal=goal,mode='jaw_opening',start_adjustment_deg=0.,
                duration=max(2.,abs(goal[5]-start[5])*1.875/np.deg2rad(25)),grasp_executed=False)
    report=clearance.path(plan)
    if report['path_minimum_m']<.005 or report['gripper_holder_path_minimum_m']-height<.005:
        raise ValueError('Opening blocked: insufficient sampled table/object-top clearance')
    plan['table_clearance']=report
    plan['jaw_opening']={'object_width_m':width,'requested_gap_m':required,
                         'initial_gap_m':initial,'model_gap_m':fingertip_gap(solver,goal),
                         'extra_gap_m':extra_gap,'method':'projected named fingertip sphere surfaces',
                         'physical_gap_validated':False,'finger_body_contact_checked':False}
    return plan
