"""Single-owner LeRobot I/O loop. Starts torque-off; only explicit UI arm enables it."""
import argparse
import json
import math
import os
import select
import sys
import time
from command_mapping import Mapping,Trajectory,BaseTestTrajectory,DirectTrajectory,DirectSegment,ROOT,JOINTS
from robot_setup import make_robot

class Controller:
    def __init__(self,bus,mapping,allow_hover=False,base_test=False,allow_grasp=False,direct=False,visual_servo=False):
        self.bus=bus;self.mapping=mapping;self.state='MONITOR';self.reason='Press A to enable the initial ±5 degree control window'
        self.trajectory=None;self.target=None;self.enabled=False
        self.allow_hover=allow_hover;self.hover=None;self.hover_used=False;self.base_test=base_test
        self.allow_grasp=allow_grasp;self.grasp=None;self.grasp_stage=None;self.grip_goal_raw=None;self.used_grasp_plans=set()
        self.seated_profile=None
        if allow_grasp and (ROOT/'calibration/seated_start.json').exists():
            self.seated_profile=json.loads((ROOT/'calibration/seated_start.json').read_text())
        self.direct=direct
        self.segment=None;self.segment_id=None
        self.direct_pause_id=None
        self.visual_servo=visual_servo;self.servo=None;self.servo_clearance=None
        if visual_servo:
            if not direct:raise ValueError('Visual servo requires direct mode')
            from live_grip import ServoClearance
            self.servo_clearance=ServoClearance()
        self.direct_grip_hold=None
        if sum(map(bool,(allow_hover,base_test,allow_grasp,direct)))>1:raise ValueError("Choose only one controller mode")
    def read(self):
        raw=self.bus.sync_read('Present_Position',normalize=False)
        return raw,self.mapping.from_raw(raw)
    def hold(self,reason):
        self.servo=None
        self.direct_pause_id=None
        self.segment=None;self.segment_id=None
        if self.enabled:
            raw,q=self.read()
            if self.direct and self.direct_grip_hold is not None:
                raw['gripper']=self.direct_grip_hold;q[5]=self.mapping.raw_scalar(5,self.direct_grip_hold)
            if self.allow_grasp and self.grasp_stage in ('close','lift') and self.grip_goal_raw is not None:
                raw['gripper']=self.grip_goal_raw;q[5]=self.mapping.raw_scalar(5,self.grip_goal_raw)
            self.bus.sync_write('Goal_Position',raw,normalize=False)
            self.target=q
            if self.trajectory:self.trajectory.command=list(q)
            self.hover=None;self.grasp=None
            self.state='HOLD'
        self.reason=reason
    def disable_all_torque(self):
        try:self.bus.disable_torque()
        except Exception as original:
            # A servo status fault must not prevent release of the remaining motors.
            if not hasattr(self.bus,'_write'):raise
            failures=[]
            for name,motor in self.bus.motors.items():
                try:
                    self.bus._write(40,1,motor.id,0,raise_on_error=False)
                    value,comm,error=self.bus._read(40,1,motor.id,raise_on_error=False)
                    if comm!=0 or value!=0:failures.append(name)
                except Exception:failures.append(name)
            if failures:raise RuntimeError('Torque release unconfirmed: '+', '.join(failures)) from original
    def release(self):
        self.servo=None
        self.direct_pause_id=None
        self.direct_grip_hold=None
        self.segment=None;self.segment_id=None
        if self.enabled:self.disable_all_torque()
        self.hover=None;self.grasp=None;self.grasp_stage=None;self.grip_goal_raw=None
        self.enabled=False;self.state='MONITOR';self.reason='Torque released; support the arm'
    def arm(self):
        self.servo=None
        self.direct_pause_id=None
        if self.state in ('HOVER','GRASP','RISING'):raise ValueError('Stop motion before rearming')
        if self.allow_grasp and self.enabled:raise ValueError('Already holding; use the grasp stage controls')
        raw,q=self.read();trajectory=(DirectTrajectory if self.direct else BaseTestTrajectory if self.base_test else Trajectory)(self.mapping,q)
        if any(v!=0 for v in self.bus.sync_read('Operating_Mode',normalize=False).values()):raise ValueError('All motors must already be in position mode')
        if not self.enabled and any(self.bus.sync_read('Torque_Enable',normalize=False).values()):raise ValueError('Unexpected enabled torque; another controller may own the arm')
        # Initialize exact raw current positions BEFORE enabling torque.
        self.bus.sync_write('Goal_Position',raw,normalize=False)
        self.enabled=True # ensure cleanup attempts release if enabling only partially succeeds
        self.bus.enable_torque()
        self.direct_grip_hold=None
        self.trajectory=trajectory;self.target=q;self.state='ACTIVE';self.reason='BASE ONLY: +/-10 degrees, 3 deg/s. Space: hold; X: release' if self.base_test else 'Move one slider at a time. Space: hold; X: release torque'
        if self.direct:self.reason='Direct joint control: full working range, 7 deg/s; Space holds, X releases'
    def handle(self,msg,now):
        if not isinstance(msg,dict):raise ValueError('Expected command object')
        age=now-float(msg.get('time',0))
        if not 0<=age<=.5:raise ValueError('Stale UI command')
        op=msg.get('op')
        if op=='seated_start':
            if not self.allow_grasp or self.state!='MONITOR' or self.enabled:raise ValueError('Seated start requires torque-off grasp mode')
            from seated_start import SeatedStart
            raw,q=self.read()
            motion=SeatedStart(self.mapping,q,json.loads((ROOT/'calibration/seated_start.json').read_text()))
            if any(self.bus.sync_read('Torque_Enable',normalize=False).values()):raise ValueError('Unexpected enabled torque')
            if any(self.bus.sync_read('Operating_Mode',normalize=False).values()):raise ValueError('Position mode required')
            self.bus.sync_write('Goal_Position',raw,normalize=False)
            self.enabled=True;self.bus.enable_torque()
            self.trajectory=motion;self.state='RISING';self.reason='Rising from sitting; Space stops, X releases'
        elif op=='arm':self.arm()
        elif op=='hold':self.hold(str(msg.get('reason','Paused by user; torque holds current position'))[:200])
        elif op=='release':self.release()
        elif op=='hover':
            if not self.allow_hover or self.hover_used or self.state!='ACTIVE':
                raise ValueError('Hover requires an enabled, unused hover controller')
            from hover_trajectory import HoverTrajectory
            raw,q=self.read()
            self.hover=HoverTrajectory(msg['report'],self.mapping,self.trajectory.command,q,time.time())
            self.hover_used=True;self.state='HOVER';self.reason='Slow hover running; Space holds, X releases'
        elif op=='grasp':
            if not self.allow_grasp or self.state not in ('ACTIVE','HOLD'):raise ValueError('Enable grasp controller holding first')
            from grasp_motion import GraspMotion
            report=msg['report'];key=(report['stage'],report['prepared_at_unix'])
            if key in self.used_grasp_plans:raise ValueError('Prepare a new stage; this preview was already used')
            raw,q=self.read()
            self.grasp=GraspMotion(report,self.mapping,self.trajectory.command,q,self.grasp_stage,time.time())
            self.used_grasp_plans.add(key);self.state='GRASP';self.reason='Grasp stage: '+self.grasp.stage+'; Space stops'
        elif op=='direct_pause':
            if not self.direct or self.state!='ACTIVE' or not self.enabled:
                raise ValueError('Adaptive pause requires active direct control')
            token=msg.get('pause_id')
            if not isinstance(token,int) or isinstance(token,bool) or token<0:
                raise ValueError('Invalid adaptive pause token')
            # Freeze the command at the worker's current position in the path.
            # No release/rearm and no new joint target from an old UI frame.
            self.segment=None;self.segment_id=None
            self.target=list(self.trajectory.command);self.direct_pause_id=token
            self.servo=None
            self.reason='Adaptive pause: holding current command for fresh planning'
        elif op=='servo_target':
            if not self.visual_servo or not self.direct or self.state!='ACTIVE' or not self.enabled:
                raise ValueError('Visual servo requires explicitly enabled servo controller')
            if self.segment is not None:raise ValueError('Pause trajectory before starting visual servo')
            from live_grip import JointServo
            self.mapping.check(msg['q'])
            if self.servo is None:self.servo=JointServo(self.trajectory.command)
            self.servo.update(msg['q'],float(msg['frame_time']),now)
            self.direct_pause_id=None;self.direct_grip_hold=None
            self.reason='Live object following; Space holds, X releases'
        elif op=='direct_segment':
            if not self.direct or self.state!='ACTIVE':raise ValueError('Segment requires active direct control')
            self.segment=DirectSegment(self.trajectory,msg['plan']);self.segment_id=msg['segment_id']
            self.servo=None
            self.direct_pause_id=None
            self.direct_grip_hold=self.trajectory.command_mapping.to_raw(self.trajectory.command)['gripper'] if msg['plan'].get('hold_gripper') else None
        elif op=='target':
            if self.state=='ACTIVE':
                self.servo=None
                self.direct_pause_id=None
                self.direct_grip_hold=None
                self.segment=None;self.segment_id=None
                q=msg['q'];(self.trajectory.command_mapping if self.direct else self.mapping).check(q);self.target=list(q)
        elif op!='heartbeat':raise ValueError('Unknown operation')


def run(bus,mapping,seconds,allow_hover=False,base_test=False,allow_grasp=False,direct=False,visual_servo=False):
    c=Controller(bus,mapping,allow_hover,base_test,allow_grasp,direct,visual_servo);last=time.monotonic();last_message=last;buffer=b'';deadline=last+seconds
    diagnostics=None;last_diagnostics=0.
    try:
        while time.monotonic()<deadline:
            start=time.monotonic()
            readable,_,_=select.select([sys.stdin],[],[],0)
            if readable:
                chunk=os.read(sys.stdin.fileno(),65536)
                if not chunk:break
                buffer+=chunk
                if len(buffer)>65536:raise ValueError('Command buffer overflow')
                while b'\n' in buffer:
                    line,buffer=buffer.split(b'\n',1)
                    try:c.handle(json.loads(line),time.monotonic());last_message=time.monotonic()
                    except (ValueError,KeyError,TypeError,OverflowError) as exc:
                        # Late in-flight targets must not overwrite the fault
                        # that already stopped the arm.
                        if c.state!='HOLD':c.hold(str(exc))
            now=time.monotonic()
            if c.state in ('ACTIVE','HOVER','GRASP','RISING') and now-last_message>.5:c.hold('UI heartbeat lost; holding current position')
            raw,q=c.read()
            if c.state in ('RISING','GRASP','ACTIVE') and now-last_diagnostics>=.5:
                diagnostics={'time':time.monotonic(),'raw_units':True}
                for register in ('Present_Load','Present_Current','Present_Voltage','Status'):
                    try:diagnostics[register]=bus.sync_read(register,normalize=False)
                    except (ValueError,KeyError,AssertionError) as exc:diagnostics[register]={'unavailable':str(exc)}
                last_diagnostics=now
            if c.state=='RISING':
                try:
                    cmd=c.trajectory.step(q,now-last)
                    bus.sync_write('Goal_Position',c.trajectory.mapping.to_raw(cmd),normalize=False)
                    if c.trajectory.done:
                        c.trajectory=Trajectory(mapping,cmd);c.target=list(cmd);c.state='HOLD'
                        c.reason='Seated startup complete; P previews approach'
                except ValueError as exc:c.hold(str(exc))
            elif c.state=='ACTIVE':
                try:
                    if c.servo is not None:
                        c.target=c.servo.step(now-last,now,limits=c.trajectory.limits)
                        c.servo_clearance.check(c.trajectory.command,c.target)
                    elif c.segment is not None:c.target=c.segment.step(now-last)
                    speed=c.servo.speed if c.servo is not None else c.segment.speed_limit if c.segment is not None else None
                    cmd=c.trajectory.step(c.target,q,now-last,**({'speed_limit':speed} if speed is not None else {}))
                    bus.sync_write('Goal_Position',(c.trajectory.command_mapping if direct else mapping).to_raw(cmd),normalize=False)
                    if direct and c.direct_grip_hold is not None:
                        c.direct_grip_hold=c.trajectory.command_mapping.to_raw(cmd)['gripper']
                except ValueError as exc:c.hold(str(exc))
            elif c.state=='HOVER':
                try:
                    cmd=c.hover.step(q,now-last)
                    bus.sync_write('Goal_Position',mapping.to_raw(cmd),normalize=False)
                    c.trajectory.command=list(cmd)
                    if c.hover.done:c.hold('Hover complete; holding. No descent or grasp.')
                except ValueError as exc:c.hold(str(exc))
            elif c.state=='GRASP':
                try:
                    cmd=c.grasp.step(q,now-last);raw_command=mapping.to_raw(cmd)
                    bus.sync_write('Goal_Position',raw_command,normalize=False);c.trajectory.command=list(cmd)
                    if c.grasp.stage in ('close','lift','open'):c.grip_goal_raw=raw_command['gripper']
                    if c.grasp.done:
                        stage=c.grasp.stage;c.grasp_stage=stage;c.grasp=None
                        c.state='HOLD';c.target=list(cmd)
                        c.reason='Stage complete: '+stage+'; inspect before the next action'
                except ValueError as exc:c.hold(str(exc))
            last=now
            if c.state=='MONITOR':
                outside=[n for n,v,(lo,hi) in zip(JOINTS,q,mapping.limits) if not lo<=v<=hi]
                c.reason=('Move away from end stops: '+', '.join(outside)) if outside else 'Ready: A enables torque at the measured pose; move one slider at a time'
                if direct:
                    try:DirectTrajectory(mapping,q);c.reason='Ready: A holds this pose; sliders control all six joints'
                    except ValueError as exc:c.reason=str(exc)
                if c.seated_profile and c.seated_profile['joint_mapping']==mapping.mapping and max(abs(a-b) for a,b in zip(q,c.seated_profile['start']))<=math.radians(.75):
                    c.reason='Sitting pose recognized: hands clear, S starts the slow inward rise'
            result={'state':c.state,'reason':c.reason,'q':q,'command':c.trajectory.command if c.trajectory else q,
                    'segment_id':c.segment_id,'segment_elapsed':c.segment.elapsed if c.segment else None,
                    'direct_pause_id':c.direct_pause_id,
                    'visual_servo_running':c.servo is not None,
                    'motor_diagnostics':diagnostics,
                    'limits':c.trajectory.limits if c.trajectory else mapping.limits,'time':time.monotonic(),
                    'grasp_stage':c.grasp_stage,'grasp_running':c.grasp.stage if c.grasp else None,'grasp_elapsed':c.grasp.elapsed if c.grasp else None,
                    'hover_used':c.hover_used,'hover_elapsed':c.hover.elapsed if c.hover else None}
            print(json.dumps(result),flush=True)
            time.sleep(max(0,1/(30 if direct else 15)-(time.monotonic()-start)))
    finally:
        if c.enabled:c.disable_all_torque()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=3600);modes=p.add_mutually_exclusive_group();modes.add_argument('--allow-hover',action='store_true');modes.add_argument('--base-test',action='store_true');modes.add_argument('--allow-grasp',action='store_true');modes.add_argument('--direct',action='store_true');p.add_argument('--visual-servo',action='store_true');args=p.parse_args()
    if args.visual_servo and not args.direct:p.error('--visual-servo requires --direct')
    robot=make_robot();bus=robot.bus
    try:
        bus.connect()
        if not robot.calibration or not bus.is_calibrated:raise RuntimeError('Project and motor calibration must match')
        run(bus,Mapping(),args.seconds,args.allow_hover,args.base_test,args.allow_grasp,args.direct,args.visual_servo)
    finally:
        if bus.is_connected:bus.disconnect(disable_torque=False)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
