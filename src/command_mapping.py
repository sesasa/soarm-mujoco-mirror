"""Pure joint mapping and bounded commissioning trajectories; no hardware I/O."""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

class Mapping:
    def __init__(self, mapping=None, calibration=None):
        self.mapping=mapping or json.loads((ROOT/'config/joint_mapping.json').read_text())
        self.calibration=calibration or json.loads((ROOT/'calibration/soarm_mirror.json').read_text())
        model=ET.parse(ROOT/'assets/so101/so101.xml')
        self.model_limits=[tuple(map(float,model.find(f".//joint[@name='{n}']").get('range').split())) for n in JOINTS]
        self.limits=[]
        for i,n in enumerate(JOINTS):
            c=self.calibration[n]
            if c['range_max']<=c['range_min']:raise ValueError('Invalid calibration range')
            # Stay 2 physical motor degrees inside the measured encoder endpoints.
            endpoints=[self.raw_scalar(i,c['range_min']+23),self.raw_scalar(i,c['range_max']-23)]
            low=max(min(endpoints),self.model_limits[i][0]);high=min(max(endpoints),self.model_limits[i][1])
            if low>=high:raise ValueError(f'No valid command range for {n}')
            self.limits.append((low,high))

    def raw_scalar(self,i,raw):
        c=self.calibration[JOINTS[i]];lo,hi=c['range_min'],c['range_max']
        if i==5:
            return self.mapping['gripper_closed_rad']+(raw-lo)/(hi-lo)*(self.mapping['gripper_open_rad']-self.mapping['gripper_closed_rad'])
        return math.radians((raw-(lo+hi)/2)*360/4095)*self.mapping['sign'][i]+self.mapping['offset_rad'][i]

    def from_raw(self,raw):return [self.raw_scalar(i,raw[n]) for i,n in enumerate(JOINTS)]

    def to_raw(self,q):
        self.check(q)
        values={}
        for i,n in enumerate(JOINTS):
            c=self.calibration[n];lo,hi=c['range_min'],c['range_max']
            if i==5:
                raw=lo+(q[i]-self.mapping['gripper_closed_rad'])/(self.mapping['gripper_open_rad']-self.mapping['gripper_closed_rad'])*(hi-lo)
            else:raw=(lo+hi)/2+math.degrees((q[i]-self.mapping['offset_rad'][i])/self.mapping['sign'][i])*4095/360
            values[n]=round(raw)
        return values

    def check(self,q):
        if len(q)!=6 or not all(math.isfinite(v) for v in q):raise ValueError('Expected six finite angles')
        for n,v,(lo,hi) in zip(JOINTS,q,self.limits):
            if not lo<=v<=hi:raise ValueError(f'{n}: {math.degrees(v):.1f} deg is outside commissioning limits [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]')

class Trajectory:
    SPEED=math.radians(5)
    EXCURSION=math.radians(5)
    TRACKING=math.radians(8)
    def __init__(self,mapping,q):
        mapping.check(q)
        self.mapping=mapping;self.origin=list(q);self.command=list(q)
        self.limits=[(max(lo,v-self.EXCURSION),min(hi,v+self.EXCURSION)) for v,(lo,hi) in zip(q,mapping.limits)]
    def step(self,target,measured,dt):
        self.mapping.check(target)
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Control interval is stale')
        for n,v,(lo,hi) in zip(JOINTS,target,self.limits):
            if not lo-1e-9<=v<=hi+1e-9:raise ValueError(f'{n}: target exceeds the initial ±5 degree window')
        if len(measured)!=6 or not all(math.isfinite(v) for v in measured):raise ValueError('Invalid feedback')
        for name,command,actual in zip(JOINTS,self.command,measured):
            if abs(command-actual)>self.TRACKING:
                raise ValueError(f'{name}: tracking error {math.degrees(abs(command-actual)):.1f} deg exceeds {math.degrees(self.TRACKING):.0f} deg')
        step=self.SPEED*dt
        self.command=[a+max(-step,min(step,b-a)) for a,b in zip(self.command,target)]
        return self.command.copy()

class BaseTestTrajectory(Trajectory):
    """Explicit larger test: base only, +/-10 degrees at 3 degrees/second."""
    SPEED=math.radians(3)
    EXCURSION=math.radians(10)
    def __init__(self,mapping,q):
        super().__init__(mapping,q)
        # Tiny display range avoids degenerate GUI sliders; other joints stay fixed.
        self.limits[1:]=[(max(lo,v-1e-9),min(hi,v+1e-9)) for v,(lo,hi) in zip(q[1:],mapping.limits[1:])]

class DirectTrajectory:
    """Full-range manual control with inward-only escape from resting margins."""
    SPEED=math.radians(7)
    TRACKING=math.radians(8)
    def __init__(self,mapping,q):
        self.mapping=mapping
        self.command_mapping=Mapping(mapping.mapping,mapping.calibration)
        self.command_mapping.limits=[tuple(sorted(mapping.raw_scalar(i,c[k]) for k in ('range_min','range_max'))) for i,n in enumerate(JOINTS) for c in [mapping.calibration[n]]]
        self.command_mapping.check(q)
        for n,v,(lo,hi) in zip(JOINTS,q,mapping.limits):
            if v<lo-math.radians(6) or v>hi+math.radians(6):raise ValueError(f'{n}: resting pose too far outside working range')
        self.command=list(q)
        self.limits=[(min(lo,v),max(hi,v)) for v,(lo,hi) in zip(q,mapping.limits)]
    def step(self,target,measured,dt,speed_limit=None):
        self.command_mapping.check(target)
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Control interval is stale')
        if len(measured)!=6 or not all(math.isfinite(v) for v in measured):raise ValueError('Invalid feedback')
        for name,command,actual in zip(JOINTS,self.command,measured):
            if abs(command-actual)>self.TRACKING:
                raise ValueError(f'{name}: tracking error {math.degrees(abs(command-actual)):.1f} deg exceeds {math.degrees(self.TRACKING):.0f} deg')
        for n,v,(lo,hi) in zip(JOINTS,target,self.limits):
            if not lo-1e-9<=v<=hi+1e-9:raise ValueError(f'{n}: outside direct-control limits')
        speed=self.SPEED if speed_limit is None else float(speed_limit)
        if not math.isfinite(speed) or not 0<speed<=32*self.SPEED:raise ValueError('Invalid playback speed limit')
        step=speed*dt
        self.command=[a+max(-step,min(step,b-a)) for a,b in zip(self.command,target)]
        # Once a joint leaves its resting margin, that margin cannot be re-entered.
        self.limits=[(min(lo,v),max(hi,v)) for v,(lo,hi) in zip(self.command,self.mapping.limits)]
        return self.command.copy()

class DirectSegment:
    """Evaluate one previewed smooth segment at the motor-loop frequency."""
    def __init__(self,trajectory,plan):
        self.start=list(plan['start']);self.goal=list(plan['goal']);self.duration=float(plan['duration'])
        trajectory.command_mapping.check(self.start);trajectory.command_mapping.check(self.goal)
        if max(abs(a-b) for a,b in zip(self.start,trajectory.command))>1e-6:raise ValueError('Segment start differs from current command')
        if not math.isfinite(self.duration) or not 4<=self.duration<=180:raise ValueError('Invalid segment duration')
        if 1.875*max(abs(a-b) for a,b in zip(self.start,self.goal))/self.duration>trajectory.SPEED+1e-9:raise ValueError('Segment exceeds direct-control speed')
        for v,(lo,hi) in zip(self.goal,trajectory.limits):
            if not lo<=v<=hi:raise ValueError('Segment goal outside working range')
        self.fast_until=float(plan.get('speedup_until_s',0.))
        if not math.isfinite(self.fast_until) or not 0<=self.fast_until<=self.duration:raise ValueError('Invalid speedup boundary')
        self.speedup_factor=float(plan.get('speedup_factor',2.))
        if not math.isfinite(self.speedup_factor) or not 1<=self.speedup_factor<=4:raise ValueError('Invalid speedup factor')
        self.tail_speedup_factor=float(plan.get('tail_speedup_factor',1.))
        if not math.isfinite(self.tail_speedup_factor) or not 1<=self.tail_speedup_factor<=4:raise ValueError('Invalid tail speedup factor')
        self.playback_rate=float(plan.get('playback_rate',1.))
        if not math.isfinite(self.playback_rate) or not 1<=self.playback_rate<=2:raise ValueError('Invalid sequence playback rate')
        self.execution_rate=float(plan.get('execution_rate',1.))
        if not math.isfinite(self.execution_rate) or not 1<=self.execution_rate<=4:raise ValueError('Invalid execution rate')
        self.speed_limit=self.execution_rate*self.playback_rate*trajectory.SPEED*max(self.speedup_factor if self.fast_until else 1,self.tail_speedup_factor)
        self.elapsed=0.
    def step(self,dt):
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Segment control interval stale')
        dt*=self.playback_rate*self.execution_rate
        fast_dt=min(dt,max(0.,self.fast_until-self.elapsed)/self.speedup_factor)
        self.elapsed+=self.speedup_factor*fast_dt+self.tail_speedup_factor*(dt-fast_dt)
        t=min(1.,self.elapsed/self.duration);b=10*t**3-15*t**4+6*t**5
        return [a+b*(g-a) for a,g in zip(self.start,self.goal)]
