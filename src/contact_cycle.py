"""Set-down/release/withdraw state machine for the isolated physics experiment."""
import numpy as np

def make_lift_waypoints(solver,start,up,axis,height=.02):
    origin=solver.position(start)
    points=[np.asarray(start,float).copy()]
    for distance in np.linspace(0,height,11)[1:]:
        plan=solver.solve_grasp(points[-1],origin+distance*up,up,axis)
        if np.max(np.abs(plan['goal'][:5]-points[-1][:5]))>np.deg2rad(15):
            raise ValueError('Lift branch discontinuity')
        points.append(plan['goal'])
    return points


class SetDownCycle:
    def __init__(self,solver,start,up,axis,hold_angle,open_angle,now):
        self.solver=solver;self.up=up;self.axis=axis
        self.hold_angle=hold_angle;self.open_angle=open_angle
        self.stage='lower';self.since=now;self.stable=0.;self.done=False;self.failed=False
        self.supported_before_release=False;self.release_started_at=None
        self.points=make_lift_waypoints(solver,start,up,axis,height=-.020)
        self.message='Lowering onto table; waiting for support'
        self.command=np.asarray(start).copy()
        self.final_support_hold=0.

    def _path(self,now,duration):
        t=float(np.clip((now-self.since)/duration,0,1));blend=10*t**3-15*t**4+6*t**5
        u=blend*(len(self.points)-1);i=min(int(u),len(self.points)-2);a=u-i
        return (1-a)*self.points[i]+a*self.points[i+1]

    def control(self,now):
        if self.stage=='lower':
            self.command=self._path(now,4.);self.command[5]=self.hold_angle
        elif self.stage=='release':
            t=float(np.clip((now-self.since)/2.,0,1));blend=10*t**3-15*t**4+6*t**5
            self.command=self.anchor.copy();self.command[5]=self.hold_angle+blend*(self.open_angle-self.hold_angle)
        elif self.stage=='withdraw':
            self.command=self._path(now,4.);self.command[5]=self.open_angle
        return self.command.copy()

    def observe(self,now,dt,q,forces,bottom,speed):
        supported=forces['table']>.10 and abs(bottom)<.0015
        clear=forces['fixed_finger']<.02 and forces['moving_finger']<.02
        if self.stage=='lower':
            self.stable=self.stable+dt if supported else 0.
            if self.stable>=.25:
                self.supported_before_release=True;self.release_started_at=now
                self.anchor=np.asarray(q).copy();self.stage='release';self.since=now;self.stable=0.
                self.message='Table support confirmed; opening virtual jaw'
            elif now-self.since>5.:
                self.failed=True;self.message='Set-down stopped: table support not confirmed'
        elif self.stage=='release':
            self.stable=self.stable+dt if supported and clear else 0.
            if now-self.since>=2. and self.stable>=.25:
                self.points=make_lift_waypoints(self.solver,q,self.up,self.axis,height=.03)
                self.stage='withdraw';self.since=now;self.stable=0.
                self.message='Object released; withdrawing open gripper'
            elif now-self.since>5.:
                self.failed=True;self.message='Release stopped: object still contacting fingers or unsupported'
        elif self.stage=='withdraw':
            if now-self.since>=4.:
                self.final_support_hold=self.final_support_hold+dt if supported and clear and speed<.01 else 0.
                if self.final_support_hold>=1.:
                    self.done=True;self.message='Pick-up, set-down, release and withdrawal passed'
                elif now-self.since>6.:
                    self.failed=True;self.message='Cycle failed: object not settled independently on table'
