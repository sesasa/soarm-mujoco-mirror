"""Restricted inward recovery from the recorded, supported sitting pose."""
import math
from command_mapping import Mapping,JOINTS

class SeatedStart:
    def __init__(self,mapping,q,profile):
        if profile['joint_mapping']!=mapping.mapping:raise ValueError('Seated pose mapping changed')
        self.mapping=Mapping(mapping.mapping,mapping.calibration)
        self.mapping.limits=[tuple(sorted(mapping.raw_scalar(i,c[k]) for k in ('range_min','range_max'))) for i,n in enumerate(JOINTS) for c in [mapping.calibration[n]]]
        self.mapping.check(q)
        if len(profile['start'])!=6 or max(abs(a-b) for a,b in zip(q,profile['start']))>math.radians(.75):raise ValueError('Arm is not in the recorded sitting pose')
        self.start=list(q);self.command=list(q);self.goal=list(profile['goal']);mapping.check(self.goal)
        for i,(a,b) in enumerate(zip(q,self.goal)):
            lo,hi=mapping.limits[i]
            if a<lo and not a<=b<=hi:raise ValueError('Startup must move inward')
            if a>hi and not lo<=b<=a:raise ValueError('Startup must move inward')
            if lo<=a<=hi:self.goal[i]=a
        if max(abs(a-b) for a,b in zip(q,self.goal))>math.radians(7):raise ValueError('Startup travel too large')
        self.duration=14.;self.elapsed=0.;self.done=False
        self.limits=[tuple(sorted((a,b))) for a,b in zip(q,self.goal)]
    def step(self,q,dt):
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Startup control interval stale')
        if len(q)!=6 or not all(math.isfinite(x) for x in q):raise ValueError('Invalid startup feedback')
        if max(abs(a-b) for a,b in zip(q,self.command))>math.radians(3):raise ValueError('Startup tracking error; stopped')
        self.elapsed+=dt;t=min(1,self.elapsed/self.duration);b=10*t**3-15*t**4+6*t**5
        self.command=[a+b*(g-a) for a,g in zip(self.start,self.goal)]
        if self.elapsed>=self.duration+2:
            if max(abs(a-b) for a,b in zip(q,self.goal))>math.radians(1):raise ValueError('Startup did not settle')
            self.done=True
        return self.command.copy()
