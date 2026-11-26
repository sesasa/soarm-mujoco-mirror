"""Validated staged grasp commands; no automatic stage advancement."""
import math

STAGE_HEIGHTS={'approach':.04,'lower':.03,'lift':.05}

STAGES={'approach':(None,4.5,140.,.025),'lower':('approach',1.5,20.,.008),
        'close':('lower',2.5,30.,.005),'lift':('close',1.5,20.,.008),
        'open':(None,2.,30.,.005)}

class GraspMotion:
    def __init__(self,report,mapping,command,measured,completed,now):
        if report.get('mode')!='supervised_grasp' or report.get('joint_mapping')!=mapping.mapping:
            raise ValueError('Wrong grasp mode or changed mapping')
        self.stage=report['stage']
        if self.stage not in STAGES:raise ValueError('Unknown grasp stage')
        previous,speed,travel,gap=STAGES[self.stage]
        if self.stage=='open':
            if completed not in ('approach','lower','close','lift'):raise ValueError('No grasp stage to open from')
        elif self.stage=='close' and completed in ('lower','close'):pass
        elif completed!=previous:raise ValueError('Complete the previous grasp stage first')
        age=now-float(report['prepared_at_unix'])
        if not math.isfinite(age) or not 0<=age<=60:raise ValueError('Preview expired; prepare it again')
        plan=report['plan'];self.start=list(plan['start']);self.goal=list(plan['goal']);self.duration=float(plan['duration'])
        mapping.check(self.start);mapping.check(self.goal);mapping.check(command)
        if len(measured)!=6 or not all(math.isfinite(v) for v in measured):raise ValueError('Invalid feedback')
        if max(abs(a-b) for a,b in zip(self.start,command))>1e-6:raise ValueError('Command changed since preview')
        if max(abs(a-b) for a,b in zip(self.start[:5],measured[:5]))>math.radians(3) or abs(self.start[5]-measured[5])>math.radians(5 if completed in ('close','lift') else 3):raise ValueError('Starting tracking error too large')
        delta=max(abs(a-b) for a,b in zip(self.start,self.goal))
        if not math.isfinite(self.duration) or not 4<=self.duration<=120:raise ValueError('Invalid stage duration')
        if delta>math.radians(travel)+1e-8 or 1.875*delta/self.duration>math.radians(speed)+1e-8:
            raise ValueError('Stage exceeds travel or speed limit')
        if self.stage in ('close','open'):
            if max(abs(a-b) for a,b in zip(self.start[:5],self.goal[:5]))>1e-9:raise ValueError('Jaw stage must keep the arm still')
            if (self.stage=='close' and self.goal[5]>self.start[5]) or (self.stage=='open' and self.goal[5]<self.start[5]):raise ValueError('Jaw command has wrong direction')
        elif abs(self.start[5]-self.goal[5])>1e-9:raise ValueError('Arm stage must leave jaw unchanged')
        expected_height=STAGE_HEIGHTS.get(self.stage)
        if expected_height is not None and report.get('reference_height_m')!=expected_height:raise ValueError('Unexpected stage height')
        check=report['clearance']
        for key,threshold in [('path_minimum_m',.005),('gripper_holder_path_minimum_m',gap)]:
            value=float(check[key])
            if not math.isfinite(value) or value<threshold:raise ValueError('Stage lacks modeled table clearance')
        if int(check['samples'])<161:raise ValueError('Insufficient path samples')
        self.elapsed=0.;self.command=self.start.copy();self.done=False
    def step(self,measured,dt):
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Grasp control interval stale')
        if len(measured)!=6 or not all(math.isfinite(v) for v in measured):raise ValueError('Invalid feedback')
        error=[abs(a-b) for a,b in zip(measured,self.command)]
        if max(error[:5])>math.radians(3) or error[5]>math.radians(5 if self.stage in ('close','lift') else 3):
            raise ValueError('Grasp tracking error too large; stopped')
        self.elapsed+=dt;t=min(1.,self.elapsed/self.duration);b=10*t**3-15*t**4+6*t**5
        self.command=[a+b*(g-a) for a,g in zip(self.start,self.goal)]
        if self.elapsed>=self.duration+2:
            err=[abs(a-b) for a,b in zip(measured,self.goal)]
            if max(err[:5])>math.radians(2) or err[5]>math.radians(5 if self.stage in ('close','lift') else 2):
                raise ValueError('Stage did not settle; inspect before continuing')
            self.done=True
        return self.command.copy()
