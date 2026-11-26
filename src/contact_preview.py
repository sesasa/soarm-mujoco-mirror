"""Isolated MuJoCo contact experiment. No cameras, serial ports or robot commands."""
import argparse,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.spatial.transform import Rotation
from simulated_reach import ReachSolver,JOINTS,hover_point,fingertip_gap
from wrist_mount_visual import apply_wrist_mount_visual
from contact_cycle import make_lift_waypoints,SetDownCycle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'captures/contact_preview'

def quat(matrix):
    q=Rotation.from_matrix(matrix).as_quat();return ' '.join(map(str,np.r_[q[3],q[:3]]))
def vec(values):return ' '.join(map(str,values))

def build(source):
    measured=source.get('measured_object')
    if not measured or not source.get('jaw_opening'):raise ValueError('Measured-object/open-jaw preview required')
    root=ET.parse(ROOT/'assets/so101/so101.xml').getroot()
    root.find('compiler').set('meshdir',str(ROOT/'assets/so101/assets'))
    apply_wrist_mount_visual(root)
    for material in root.findall('asset/material'):
        if 'sts3215' not in material.get('name',''):material.set('rgba','.88 .89 .87 1')
    option=root.find('option');option.set('timestep','.001');option.set('iterations','60')
    ET.SubElement(root.find('visual'),'global',offwidth='960',offheight='720')
    # Contact experiment: highlighter, table, decomposed finger meshes, and holder.
    # Other arm links/self-collisions remain outside this isolated test.
    for i,g in enumerate(root.findall('.//body/geom')):
        g.set('contype','0');g.set('conaffinity','0')
        if g.get('class')=='collision_gripper_mesh':
            g.set('name',f'finger_contact_{i}');g.set('contype','4');g.set('conaffinity','3')
            g.set('friction','.6 .005 .0001');g.set('solref','.01 1')
        elif g.get('name','').startswith('photo_holder_'):
            g.set('contype','8');g.set('conaffinity','3')
    for actuator in root.findall('actuator/position'):
        if actuator.get('name')=='gripper':
            actuator.set('kp','5');actuator.set('kv','.08');actuator.set('forcerange','-.08 .08')
    up=np.array(source['table_up']);along=np.array(source['object_axis']);across=np.cross(along,up)
    basis=np.column_stack([across,along,up]);surface=np.array(source['surface'])
    world=root.find('worldbody')
    ET.SubElement(world,'geom',name='test_table',type='plane',pos=vec(surface),quat=quat(basis),size='.5 .5 .01',
                  rgba='.42 .40 .35 1',contype='2',conaffinity='13',friction='.6 .005 .0001')
    body=ET.SubElement(world,'body',name='test_highlighter',pos=vec(surface+up*(measured['height_m']/2+.0002)),quat=quat(basis))
    ET.SubElement(body,'freejoint',name='object_free')
    ET.SubElement(body,'geom',name='test_object',type='box',size=vec([measured['width_m']/2,measured['estimated_length_m']/2,measured['height_m']/2]),
                  mass='.03',rgba='1 .12 .25 1',contype='1',conaffinity='14',friction='.6 .005 .0001',condim='4',solref='.01 1')
    ET.SubElement(world,'light',pos='0 0 1',dir='0 0 -1',directional='true',diffuse='.8 .8 .8',ambient='.4 .4 .4')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    return model,basis


def contact_state(model,data):
    object_id=model.geom('test_object').id;table=model.geom('test_table').id
    forces={'fixed_finger':0.,'moving_finger':0.,'table':0.,'holder':0.}
    unwanted=False
    for i in range(data.ncon):
        c=data.contact[i];ids=[int(c.geom1),int(c.geom2)]
        if object_id in ids:
            other=ids[1] if ids[0]==object_id else ids[0]
            name=model.geom(other).name;body=model.body(int(model.geom_bodyid[other])).name
            category='table' if other==table else 'holder' if name.startswith('photo_holder_') else 'moving_finger' if body=='moving_jaw_so101_v1' else 'fixed_finger' if body=='gripper' else None
            if category:
                force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force);forces[category]+=max(0.,float(force[0]))
        if table in ids:
            other=ids[1] if ids[0]==table else ids[0]
            if model.geom(other).name.startswith(('finger_contact_','photo_holder_')):unwanted=True
    return forces,unwanted


def object_clearance(model,data,surface,up):
    geom=model.geom('test_object').id
    normal_local=data.geom_xmat[geom].reshape(3,3).T@up
    return float((data.geom_xpos[geom]-surface)@up-np.abs(normal_local)@model.geom_size[geom])



def run(headless=False,source_path=None,lift=False,cycle=False):
    lift=lift or cycle
    OUT.mkdir(exist_ok=True)
    source=json.loads(Path(source_path or ROOT/'captures/scene_changes/last_simulated_reach.json').read_text())
    model,basis=build(source);data=mujoco.MjData(model);solver=ReachSolver(model)
    qstart=np.asarray(source['goal'],float).copy();qstart[5]=source['goal'][5]
    up=basis[:,2];surface=np.array(source['surface']);axis=np.array(source['object_axis'])
    start_height=float((solver.position(qstart)-surface)@up)
    heights=np.linspace(start_height,.012,max(2,int(abs(start_height-.012)/.002)+1))
    waypoints=[qstart];errors=[]
    for h in heights[1:]:
        plan=solver.solve_grasp(waypoints[-1],hover_point(surface,up,h),up,axis)
        if np.max(np.abs(plan['goal']-waypoints[-1]))>np.deg2rad(15):raise ValueError('Descent branch discontinuity')
        waypoints.append(plan['goal']);errors.append(plan['error_m'])
    addresses=[model.joint(n).qposadr[0] for n in JOINTS]
    actuators=[model.actuator(n).id for n in JOINTS]
    data.qpos[addresses]=qstart;data.ctrl[actuators]=qstart;mujoco.mj_forward(model,data)
    initial_object=data.xpos[model.body('test_highlighter').id].copy()
    renderer=mujoco.Renderer(model,height=720,width=960)
    camera=mujoco.MjvCamera();camera.lookat[:]=surface+up*.10;camera.distance=.48
    camera.azimuth=np.degrees(np.arctan2(surface[1],surface[0]))+100;camera.elevation=-22
    options=mujoco.MjvOption();options.geomgroup[:]=0;options.geomgroup[2]=1;options.geomgroup[0]=1
    window='MUJOCO PICK AND RELEASE - no hardware' if cycle else 'MUJOCO LIFT TEST - no hardware' if lift else 'MUJOCO CONTACT TEST - no hardware'
    if not headless:cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,960,720)
    records=[];bilateral_time=0.;held=False;outcome='No bilateral contact';last_frame=None;phase='settle';wall=time.monotonic()
    closing_start=None;stopped=False;setdown=None
    lift_started=None;lift_points=None;lift_passed=False;airborne_hold=0.;lost_contact=0.;maximum_clearance=0.
    try:
        for step in range(45000 if cycle else 26000 if lift else 18000):
            t=step*model.opt.timestep
            if t<.7:q=qstart.copy();phase='Settling object on table'
            elif t<4.7:
                u=(t-.7)/4* (len(waypoints)-1);index=min(int(u),len(waypoints)-2);blend=u-index
                q=(1-blend)*waypoints[index]+blend*waypoints[index+1];phase='Lowering OPEN virtual fingers'
            else:
                q=waypoints[-1].copy();phase='Closing virtual jaw slowly'
                if closing_start is None:closing_start=float(data.qpos[addresses[5]])
                q[5]=max(solver.bounds[5,0],closing_start-np.deg2rad(8)*(t-4.7))
                if held:q[5]=hold_angle;phase='Holding contact - no lift'
            if lift and held and t-held_at>.8:
                if lift_started is None:
                    try:
                        lift_points=make_lift_waypoints(solver,data.qpos[addresses].copy(),up,axis)
                        lift_started=t
                    except ValueError as error:
                        outcome='Lift refused: '+str(error);stopped=True
                if lift_points is not None:
                    progress=np.clip((t-lift_started)/4.,0.,1.)
                    smooth=10*progress**3-15*progress**4+6*progress**5
                    u=smooth*(len(lift_points)-1);index=min(int(u),len(lift_points)-2);blend=u-index
                    q=(1-blend)*lift_points[index]+blend*lift_points[index+1]
                    q[5]=hold_angle
                    phase='Lifting virtual object 2 cm' if progress<1 else 'Testing airborne hold (2 seconds)'
            if setdown is not None:
                q=setdown.control(t);phase=setdown.message
            data.ctrl[actuators]=q
            mujoco.mj_step(model,data)
            forces,unwanted=contact_state(model,data)
            bilateral=forces['fixed_finger']>.02 and forces['moving_finger']>.02
            bilateral_time=bilateral_time+model.opt.timestep if bilateral else 0.
            if bilateral_time>.25 and not held:
                held=True;hold_angle=float(q[5]);held_at=t
                outcome='Both fingers established simulated contact'
            if unwanted or forces['holder']>.01:
                outcome='Stopped: finger/table or holder contact';stopped=True
            bottom=object_clearance(model,data,surface,up)
            maximum_clearance=max(maximum_clearance,bottom)
            if lift_started is not None and setdown is None:
                lost_contact=0. if bilateral else lost_contact+model.opt.timestep
                airborne=(bottom>.010 and forces['table']<.01 and bilateral)
                if t-lift_started>=4:
                    airborne_hold=airborne_hold+model.opt.timestep if airborne else 0.
                if lost_contact>.25:
                    outcome='Lift failed: finger contact lost';stopped=True
                elif t-lift_started>=6:
                    lift_passed=airborne_hold>=1.9
                    outcome='Simulated 2 cm lift and airborne hold passed' if lift_passed else 'Lift failed: object not held clear of table'
                    if cycle:
                        if lift_passed:
                            try:setdown=SetDownCycle(solver,data.qpos[addresses].copy(),up,axis,hold_angle,qstart[5],t)
                            except ValueError as error:outcome='Set-down refused: '+str(error);stopped=True
                        else:stopped=True
            if setdown is not None:
                object_dof=model.joint('object_free').dofadr[0]
                speed=float(np.linalg.norm(data.qvel[object_dof:object_dof+3]))
                try:setdown.observe(t,model.opt.timestep,data.qpos[addresses].copy(),forces,bottom,speed)
                except ValueError as error:setdown.failed=True;setdown.message='Withdraw refused: '+str(error)
                phase=setdown.message
                if setdown.done or setdown.failed:outcome=setdown.message
                if setdown.failed:stopped=True
            finished=(stopped or (setdown is not None and setdown.done)
                      or (not lift and held and t-held_at>.8)
                      or (not cycle and lift_started is not None and t-lift_started>=6))
            if step%20==0:
                records.append({'time_s':t,'phase':phase,'forces_N':forces,'joint_angle_rad':float(data.qpos[addresses[5]]),'bilateral':bilateral,'object_bottom_clearance_m':bottom,'lift_elapsed_s':None if lift_started is None else t-lift_started,'cycle_stage':None if setdown is None else setdown.stage})
            if step%40==0 or finished:
                renderer.update_scene(data,camera=camera,scene_option=options)
                last_frame=cv2.cvtColor(renderer.render(),cv2.COLOR_RGB2BGR)
                cv2.rectangle(last_frame,(0,0),(960,64),(20,20,20),-1)
                cv2.putText(last_frame,'PICK AND RELEASE SIMULATION - REAL ARM STAYS STILL' if cycle else 'ISOLATED LIFT SIMULATION - REAL ARM STAYS STILL' if lift else 'ISOLATED CONTACT SIMULATION - REAL ARM STAYS STILL',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,0),1)
                cv2.putText(last_frame,phase,(10,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
                cv2.rectangle(last_frame,(0,660),(960,720),(20,20,20),-1)
                cv2.putText(last_frame,f"Fixed finger {forces['fixed_finger']:.2f} N | moving finger {forces['moving_finger']:.2f} N | assumed mass/friction",(10,685),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
                if lift:cv2.putText(last_frame,f'Object/table gap: {bottom*1000:.1f} mm | table support: {forces["table"]:.2f} N',(10,85),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
                cv2.putText(last_frame,outcome+' | Esc: close',(10,709),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
                cv2.imwrite(str(OUT/'latest.jpg'),last_frame)
                if not headless:
                    cv2.imshow(window,last_frame)
                    if cv2.waitKey(1)&255==27:outcome='Closed by user';break
                    time.sleep(max(0,t-(time.monotonic()-wall)))
            if finished:break
        displacement=float(np.linalg.norm(data.xpos[model.body('test_highlighter').id]-initial_object))
        result={'outcome':outcome,'bilateral_contact_established':held,'unexpected_contact':stopped,
                'object_displacement_m':displacement,'final_forces_N':forces,'simulated_time_s':float(data.time),
                'source_snapshot':source,'assumptions':{'mass_kg':.03,'friction':.6,'gripper_torque_limit_Nm':.08,
                  'contact_height_m':.012,'object_shape':'box','robot_contacts':'decomposed finger meshes and approximate holder'},
                'limitations':['Mass, friction, actuator response and contact compliance are assumed.','No other tabletop objects or arm self-collisions included.','Simulation contact is not a validated physical grasp.'],
                'hardware_commands':False,'lift_attempted':lift_started is not None,
                'lift_passed':lift_passed,'lift_requested_m':.02 if lift else None,
                'final_object_bottom_clearance_m':bottom,'maximum_object_bottom_clearance_m':maximum_clearance,
                'airborne_hold_s':airborne_hold,'cycle_requested':cycle,
                'cycle_passed':bool(setdown and setdown.done),'supported_before_release':bool(setdown and setdown.supported_before_release),
                'final_table_support_hold_s':setdown.final_support_hold if setdown else 0.,
                'release_started_at_s':setdown.release_started_at if setdown else None}
        if cycle:
            (OUT/'cycle_result.json').write_text(json.dumps(result,indent=2))
            (OUT/'cycle_trace.json').write_text(json.dumps(records,indent=2))
        if lift:
            (OUT/'lift_result.json').write_text(json.dumps(result,indent=2))
            (OUT/'lift_trace.json').write_text(json.dumps(records,indent=2))
        (OUT/'result.json').write_text(json.dumps(result,indent=2));(OUT/'contact_trace.json').write_text(json.dumps(records,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_snapshot',)},indent=2),flush=True)
        if not headless:
            while cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>0:
                if cv2.waitKey(50)&255==27:break
        return result
    finally:renderer.close();cv2.destroyAllWindows()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--headless',action='store_true');parser.add_argument('--source',type=Path);parser.add_argument('--lift',action='store_true');parser.add_argument('--cycle',action='store_true');args=parser.parse_args();run(args.headless,args.source,args.lift,args.cycle)
