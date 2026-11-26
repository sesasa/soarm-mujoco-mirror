"""Plan a held packet over a measured basket, then open above its rim."""
import json
from itertools import product
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares, brentq
from command_mapping import ROOT
from simulated_reach import ReachSolver, trajectory as joint_trajectory, fingertip_gap


def trajectory(plan,t):
    return joint_trajectory({**plan,"start":np.asarray(plan["start"]),"goal":np.asarray(plan["goal"])},t)

def basket_visuals(config):
    up=-np.asarray(config['rotation'])[:,2]
    if up[2]<0:up=-up
    points=np.asarray(config['rim_polygon_base_m']);height=config['rim_height_m']
    result=[]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        x=b-a;x/=np.linalg.norm(x);y=np.cross(up,x)
        result.append(((a+b)/2-up*height/2,np.array([np.linalg.norm(b-a)/2,.003,height/2]),np.column_stack([x,y,up])))
    return result


class BasketClearance:
    def __init__(self,checker,config):
        self.checker=checker;self.origin=np.asarray(config['center_base_m'])
        self.rotation=np.asarray(config['rotation']);self.up=checker.normal
        self.polygon=((np.asarray(config['rim_polygon_base_m'])-self.origin)@self.rotation)[:,:2].astype(np.float32)
    def inside(self,points,margin=.002):
        xy=((points-self.origin)@self.rotation)[:,:2]
        return all(cv2.pointPolygonTest(self.polygon,tuple(map(float,p)),True)>=margin for p in xy)
    def check_robot(self,q):
        c=self.checker;c.set_pose(q);m=c.model;d=c.data
        for i in c.geoms:
            if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                mesh=int(m.geom_dataid[i]);pts=c.mesh_hulls[mesh]
            else:
                size=m.geom_size[i].copy()
                if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE:size[:]=size[0]
                elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE:size=np.array([size[0],size[0],size[0]+size[1]])
                elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER:size=np.array([size[0],size[0],size[1]])
                pts=np.array(list(product([-1,1],repeat=3)))*size
            world=pts@d.geom_xmat[i].reshape(3,3).T+d.geom_xpos[i]
            if np.min((world-self.origin)@self.up)>=.008:continue
            xy=((world-self.origin)@self.rotation)[:,:2].astype(np.float32)
            # Conservative projection test against basket footprint, including rim thickness.
            hull=cv2.convexHull(xy).reshape(-1,2)
            if cv2.intersectConvexConvex(hull,self.polygon)[0]>1e-8 or any(cv2.pointPolygonTest(self.polygon,tuple(map(float,p)),True)>=-.008 for p in hull):
                raise ValueError('Robot geometry crosses basket footprint below rim clearance')


def append_basket(reports,checker,model,mapping,region,up,config,target_profile=None,return_to_sitting=True):
    if config.get('occupied_height_m',0)>0:
        config=dict(config);occupied=float(config['occupied_height_m'])
        config['center_base_m']=(np.asarray(config['center_base_m'])+up*occupied).tolist()
        config['rim_polygon_base_m']=(np.asarray(config['rim_polygon_base_m'])+up*occupied).tolist()
        config['rim_height_m']+=occupied
    solver=ReachSolver(model);solver.bounds=np.asarray(mapping.limits);body=model.body('gripper').id
    closed=np.asarray(reports[-1]['plan']['goal']);solver.position(closed)
    rotation=solver.data.xmat[body].reshape(3,3).copy();origin=solver.data.xpos[body].copy()
    height=float((target_profile if target_profile is not None else json.loads((ROOT/'config/grasp_target.json').read_text())).get('height_m',.03))
    if not np.isfinite(height) or height<=0:raise ValueError('Invalid object thickness')
    size=np.asarray(region[1]).copy();size[2]=height/2
    world=np.array(list(product([-1,1],repeat=3)))*size@np.asarray(region[2]).T+np.asarray(region[0])+up*height/2
    attached=(world-origin)@rotation;local_center=attached.mean(axis=0)
    long_axis=np.asarray(region[2])[:,np.argmax(size[:2])]@rotation
    basket=BasketClearance(checker,config);center=np.asarray(config['center_base_m'])
    axis=np.asarray(config['rotation'])[:,np.argmax(config['opening_size_m'])]
    def payload(q):
        solver.position(q);r=solver.data.xmat[body].reshape(3,3);return attached@r.T+solver.data.xpos[body]
    def solve(q,target,desired_axis,axis_weight=.035,fit_basket=False):
        candidates=[]
        for sign in (1,-1):
            def residual(x):
                solver.position(np.r_[x,q[5]]);r=solver.data.xmat[body].reshape(3,3)
                delta=local_center@r.T+solver.data.xpos[body]-target
                if fit_basket:
                    # Fit the carried box itself, with analytic polygon distances
                    # so finite-difference IK retains precision near a narrow rim.
                    xy=((attached@r.T+solver.data.xpos[body]-center)@basket.rotation)[:,:2]
                    polygon=basket.polygon.astype(float);edges=np.roll(polygon,-1,axis=0)-polygon
                    inward=np.column_stack([-edges[:,1],edges[:,0]])
                    inward/=np.linalg.norm(inward,axis=1)[:,None]
                    signs=np.sign(np.sum((polygon.mean(axis=0)-polygon)*inward,axis=1))
                    inward*=signs[:,None]
                    distances=np.sum((xy[:,None,:]-polygon[None,:,:])*inward[None,:,:],axis=2)
                    penalty=3*np.maximum(.003-distances,0).ravel()
                    delta=.2*(delta-up*np.dot(delta,up))+up*np.dot(delta,up)
                    return np.r_[delta,.01*(r@long_axis-sign*desired_axis),penalty,.0001*(x-q[:5])]
                return np.r_[delta,axis_weight*(r@long_axis-sign*desired_axis),.001*(x-q[:5])]
            for seed in (q[:5],np.mean(solver.bounds[:5],axis=1)):
                fit=least_squares(residual,np.clip(seed,solver.bounds[:5,0]+1e-6,solver.bounds[:5,1]-1e-6),bounds=solver.bounds[:5].T,max_nfev=250,gtol=1e-10,ftol=1e-10,xtol=1e-10)
                goal=np.r_[fit.x,q[5]]
                delta=payload(goal).mean(axis=0)-target
                if (np.linalg.norm(delta)<.002 if not fit_basket else np.linalg.norm(delta)<.025 and abs(delta@up)<.003 and basket.inside(payload(goal))):candidates.append(goal)
        return sorted(candidates,key=lambda g:np.linalg.norm(g-q))
    def report(stage,q,goal,carrying=True):
        mapping.check(goal);duration=max(4.,1.875*max(abs(goal-q))/np.deg2rad(2.5 if stage=='basket_open' else 1.5 if stage=='basket_lift' else 6.))
        plan={'start':q.tolist(),'goal':goal.tolist(),'duration':duration,'hold_gripper':carrying}
        if stage in ('basket_lift','basket_transfer','basket_open'):plan.update(speedup_until_s=duration,speedup_factor=4.,playback_duration=duration/4)
        clearance=checker.path({**plan,"start":q,"goal":goal},121)
        if clearance['path_minimum_m']<.005:raise ValueError(stage+': table clearance')
        minimum=float('inf')
        for t in np.linspace(0,duration,121):
            sample=trajectory(plan,t);basket.check_robot(sample)
            if carrying:
                pts=payload(sample);minimum=min(minimum,float(min((pts-checker.point)@up)))
                if minimum<-.001:raise ValueError(stage+': packet crosses tabletop')
                if stage=='basket_transfer' and min((pts-center)@up)<.008:raise ValueError('Packet transfer too low for basket rim')
        return {'stage':stage,'plan':plan,'clearance':clearance,'basket_rim_height_m':config['rim_height_m'],'object_height_m':height,'carried_object_minimum_m':minimum if carrying else None}
    # Check the complete incoming reach as well as the loaded transfer.
    for r in reports:
        for t in np.linspace(0,r['plan']['duration'],81):basket.check_robot(trajectory(r['plan'],t))
    errors=[]
    initial_center=payload(closed).mean(axis=0);initial_axis=np.asarray(region[2])[:,np.argmax(size[:2])]
    for extra in (.055,.075,.095):
        desired_height=config['rim_height_m']+extra
        lift_target=initial_center+up*(desired_height-(initial_center-checker.point)@up)
        lifts=solve(closed,lift_target,initial_axis)
        lifts+=solve(closed,lift_target,initial_axis,axis_weight=.005)
        if not lifts:errors.append(f'No lift pose at {desired_height*1000:.0f} mm object center height')
        for lifted in lifts:
            try:lift=report('basket_lift',closed,lifted)
            except ValueError as exc:errors.append(str(exc));continue
            destinations=solve(lifted,center+up*extra,axis,fit_basket=True)
            if not destinations:errors.append('No basket release pose for held orientation')
            for destination in destinations:
                try:
                    pts=payload(destination)
                    if not basket.inside(pts):raise ValueError('Held packet does not fit projected basket opening')
                    transfer=report('basket_transfer',lifted,destination)
                    opened=destination.copy();gap=min(.075,2*min(size[:2])+.015)
                    def f(a):
                        q=destination.copy();q[5]=a;return fingertip_gap(solver,q)-gap
                    opened[5]=brentq(f,mapping.limits[5][0],min(1.,mapping.limits[5][1]))
                    release=report('basket_open',destination,opened,False)
                    import importlib,sitting_return
                    if return_to_sitting:
                        finish=importlib.reload(sitting_return).build_sitting_return(opened,mapping,checker,basket)
                    else:
                        from camera_visibility import CameraVisibility
                        visibility=CameraVisibility(checker)
                        polygons=[np.asarray(config['inner_rim_pixels'])]
                        if config.get('next_target_pixels') is not None:polygons.append(np.asarray(config['next_target_pixels']))
                        occupied=dict(config)
                        occupied['center_base_m']=(center+up*height).tolist()
                        occupied['rim_polygon_base_m']=(np.asarray(config['rim_polygon_base_m'])+up*height).tolist()
                        occupied_check=BasketClearance(checker,occupied)
                        finish=None
                        for degrees in (0,-15,15,-30,30,-45,45,-60,60,-90,90):
                            clear=lifted.copy();clear[5]=opened[5];clear[0]+=np.deg2rad(degrees)
                            try:mapping.check(clear)
                            except ValueError:continue
                            coverage=visibility.coverage(clear,polygons)
                            if max(coverage)>.15:continue
                            try:
                                candidate=report('plate_exit',opened,clear,False)
                                for t in np.linspace(0,candidate['plan']['duration'],121):occupied_check.check_robot(trajectory(candidate['plan'],t))
                            except ValueError:continue
                            candidate['plan'].update(speedup_until_s=candidate['plan']['duration'],speedup_factor=4.,playback_duration=candidate['plan']['duration']/4)
                            candidate['camera_target_occlusion']=coverage
                            candidate['view_clearance_base_turn_deg']=degrees
                            finish=candidate;break
                        if finish is None:raise ValueError('No intermediate pose leaves the plate and next object visible')
                    reports.extend([lift,transfer,release,finish]);return
                except ValueError as exc:errors.append(str(exc))
    raise ValueError('Basket placement: '+ '; '.join(dict.fromkeys(errors or ['No basket pose reachable within calibrated limits'])))
