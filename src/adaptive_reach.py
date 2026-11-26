"""Selected-object monitoring for optional pause/replan reach adaptation.

Pure observation/state logic: never opens devices or sends robot commands.
This is stop-and-replan, not continuous moving-target interception.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np

@dataclass
class Observation:
    region: tuple
    histogram: np.ndarray

    @property
    def center(self):return np.asarray(self.region[0],float)
    @property
    def size(self):return np.sort(2*np.asarray(self.region[1],float)[:2])
    @property
    def axis(self):
        a=np.asarray(self.region[2],float)[:,np.argmax(np.asarray(self.region[1])[:2])]
        return a/np.linalg.norm(a)


def observe_regions(frame,contours,regions):
    """Describe each current contour, preserving its detector index."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    output=[]
    for c,r in zip(contours,regions):
        if not all(np.isfinite(v).all() for v in map(np.asarray,r)):
            raise ValueError('Non-finite adaptive observation')
        mask=np.zeros(frame.shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.asarray(c,np.int32)],255)
        mask=cv2.erode(mask,np.ones((3,3),np.uint8))
        hist=cv2.calcHist([hsv],[0,1,2],mask,[12,8,8],[0,180,0,256,0,256])
        if hist.sum():hist/=hist.sum()
        output.append(Observation(tuple(np.asarray(v).copy() for v in r),hist.astype(np.float32)))
    return output


def axis_angle(a,b):
    # A rectangular object's long axis has no distinguished forward direction.
    return math.degrees(math.acos(float(np.clip(abs(a.axis@b.axis),0,1))))


def pose_matches(a,b,distance=.008,angle=8.,size_fraction=.20):
    return (np.linalg.norm(a.center-b.center)<=distance and axis_angle(a,b)<=angle
            and np.max(abs(a.size-b.size)/np.maximum(b.size,.005))<=size_fraction)


class TargetWatch:
    """Bind one appearance/shape; fail closed on missing or ambiguous matches."""
    stable_seconds=.6
    lost_seconds=.25
    freshness_seconds=.20

    def __init__(self,reference,now):
        self.reference=reference;self.identity=reference;self.current=reference
        self.last_seen=now;self.still_since=now;self.still_anchor=reference
        self.problem=None

    def update(self,observations,now):
        candidates=[]
        for obs in observations:
            shape=float(np.max(abs(obs.size-self.identity.size)/np.maximum(self.identity.size,.005)))
            distance=float(np.linalg.norm(obs.center-self.current.center))
            appearance=float(cv2.compareHist(obs.histogram,self.identity.histogram,cv2.HISTCMP_BHATTACHARYYA))
            if shape<=.35 and distance<=.12 and appearance<=.45:
                candidates.append((appearance+shape*.3+distance,obs))
        candidates.sort(key=lambda v:v[0])
        ambiguous=len(candidates)>1 and candidates[1][0]-candidates[0][0]<.12
        if not candidates or ambiguous:
            self.problem='ambiguous target' if ambiguous else 'target not visible'
            self.still_since=None
            return
        self.problem=None;obs=candidates[0][1]
        if (self.still_since is None or now-self.last_seen>self.lost_seconds
                or not pose_matches(obs,self.still_anchor,.002,3.,.08)):
            self.still_since=now;self.still_anchor=obs
        self.current=obs;self.last_seen=now

    def fresh(self,now):return self.problem is None and -1e-9<=now-self.last_seen<=self.freshness_seconds
    def stable(self,now):return self.fresh(now) and self.still_since is not None and now-self.still_since>=self.stable_seconds
    def moved(self):return not pose_matches(self.current,self.reference)
    def action(self,stage,now):
        if stage not in ('clear_wrist','ready','approach','lower'):return None
        bad=self.moved() or self.problem=='ambiguous target' or now-self.last_seen>self.lost_seconds
        if bad:return 'stop' if stage=='lower' else 'pause'
        return None
    def can_descend(self,now):return self.stable(now) and not self.moved()
    def accepts_plan(self,planned,now):return self.stable(now) and pose_matches(self.current,planned)
    def commit(self,planned):self.reference=planned


def planning_result_current(state,start,watch,planned,now):
    """A result computed while holding must still match both arm and target."""
    q=np.asarray(state.get('command',[]),float);start=np.asarray(start,float)
    return (state.get('state')=='ACTIVE' and q.shape==(6,) and start.shape==(6,)
            and np.isfinite(q).all() and np.max(abs(q-start))<=1e-6
            and watch.accepts_plan(planned,now))


def pause_acknowledged(state,token):
    return (state.get('state')=='ACTIVE' and state.get('direct_pause_id')==token
            and state.get('segment_id') is None)


def committed_grip_valid(watch,now,deadline):
    """Allow a latched final grasp to finish when the object disappears.

    The caller sets a deadline only after a fresh stable hover is committed.
    Missing visual evidence is not proof of motion. Visible displacement and
    ambiguous association remain reasons to stop; no camera-loss override here.
    """
    if watch is None or deadline is None or not now<=deadline:return False
    if watch.moved():return False
    if watch.fresh(now):return True
    return watch.problem=='target not visible'


class FinalApproachMemory:
    """Short-lived stable target for finishing the last 6 cm of a grasp reach."""
    def __init__(self):self.observation=None;self.seen=None
    def update(self,watch,now):
        if watch is None:return
        if watch.stable(now):self.observation=watch.current;self.seen=now
        elif watch.fresh(now) or watch.problem=='ambiguous target':
            self.observation=None;self.seen=None
    def permits(self,watch,now,remaining_m):
        return (self.observation is not None and self.seen is not None
                and watch is not None and watch.problem=='target not visible'
                and 0<=now-self.seen<=12. and math.isfinite(remaining_m)
                and 0<=remaining_m<=.06
                and pose_matches(watch.current,self.observation))
