"""One bounded position-only hover trajectory. Pure math; no hardware I/O."""
import math

class HoverTrajectory:
    SPEED=math.radians(3)
    TRACKING=math.radians(3)
    def __init__(self,report,mapping,command,measured,now):
        if report.get('mode')!='position_only_high_hover' or report.get('height_above_table_m')!=.12:
            raise ValueError('Only the 12 cm position-only hover is supported')
        if report.get('joint_mapping')!=mapping.mapping:raise ValueError('Hover mapping changed')
        age=now-float(report['prepared_at_unix'])
        if not math.isfinite(age) or not 0<=age<=30:raise ValueError('Hover plan is stale; preview again')
        plan=report['plan'];self.start=list(plan['start']);self.goal=list(plan['goal']);self.duration=float(plan['duration'])
        mapping.check(self.start);mapping.check(self.goal);mapping.check(measured);mapping.check(command)
        if not math.isfinite(self.duration) or not 5<=self.duration<=90:raise ValueError('Invalid hover duration')
        delta=max(abs(a-b) for a,b in zip(self.start,self.goal))
        if delta>math.radians(60) or 1.875*delta/self.duration>self.SPEED+1e-9:raise ValueError('Hover exceeds travel or speed limit')
        if abs(self.goal[5]-self.start[5])>1e-9:raise ValueError('Hover must not move the jaw')
        if max(abs(a-b) for a,b in zip(self.start,command))>1e-6:raise ValueError('Command changed since preview')
        if max(abs(a-b) for a,b in zip(self.start,measured))>math.radians(2):raise ValueError('Starting feedback differs by more than 2 degrees')
        check=report['clearance']
        for key,threshold in [('path_minimum_m',.005),('gripper_holder_path_minimum_m',.08)]:
            value=float(check[key])
            if not math.isfinite(value) or value<threshold:raise ValueError('Insufficient modeled table clearance')
        if int(check['samples'])<161:raise ValueError('Insufficient path samples')
        self.elapsed=0.;self.command=self.start.copy();self.done=False
    def step(self,measured,dt):
        if not math.isfinite(dt) or not 0<dt<=.2:raise ValueError('Hover control interval stale')
        if len(measured)!=6 or not all(math.isfinite(v) for v in measured):raise ValueError('Invalid hover feedback')
        if max(abs(a-b) for a,b in zip(measured,self.command))>self.TRACKING:raise ValueError('Hover tracking error exceeds 3 degrees')
        self.elapsed+=dt;t=min(1.,self.elapsed/self.duration);blend=10*t**3-15*t**4+6*t**5
        self.command=[a+blend*(b-a) for a,b in zip(self.start,self.goal)]
        if self.elapsed>=self.duration+2:
            if max(abs(a-b) for a,b in zip(measured,self.goal))>math.radians(2):raise ValueError('Hover endpoint did not settle within 2 degrees')
            self.done=True
        return self.command.copy()
