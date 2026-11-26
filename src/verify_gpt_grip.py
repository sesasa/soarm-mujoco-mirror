"""Offline saved-matchbox grasp replay. No hardware access."""
import json
import numpy as np
from command_mapping import Mapping,DirectTrajectory,DirectSegment
from preview_camera_alignment import Preview
from scene_changes import Plane
from mirror_grasp_plan import build_sequence
from live_grip import WarmGraspIK,JointServo,ServoClearance
from grasp_tail import build_tail

p=Preview()
try:
 m=Mapping();plane=Plane(p.K,p.T,np.array(json.load(open('captures/scene_changes/plane.json'))['T_camera_from_board']))
 f='captures/live_grip/1789923501036923903/'
 r=tuple(np.array(x) for x in json.load(open(f+'gpt_selection.json'))['region'])
 q=json.load(open(f+'initial_plan.json'))[0]['plan']['start']
 profile=json.load(open('config/grasp_target.json'));profile.update(preflight_grip=True,after_grasp='hold',closing_gap_m=max(.005,float(2*min(r[1][:2]))-.004))
 plans=build_sequence(p.model,m,plane,{'q':q,'command':q},r,reach_only=True,target_profile_override=profile)
 tr=DirectTrajectory(m,q)
 for report in plans[:-1]:
  seg=DirectSegment(tr,report['plan'])
  while seg.elapsed<seg.duration+.1:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
 w=WarmGraspIK(p.model,m,plane,profile);w.reset(plans[-1]['plan']['goal'],r)
 servo=JointServo(q);guard=ServoClearance(p.model)
 for i in range(1,601):
  t=i/30
  if i%3==1:
   goal,info=w.solve(r);servo.update(goal,t,t)
  cmd=servo.step(1/30,t,limits=tr.limits);guard.check(q,cmd);q=tr.step(cmd,q,1/30,speed_limit=servo.speed)
 error=np.linalg.norm(w.solver.position(q)-w.target(r));assert error<.004
 tail=build_tail(p.model,m,plane,q,closing_gap=profile['closing_gap_m'],lift_height=0,descent_m=profile['descent_m'])
 for report in tail:
  seg=DirectSegment(tr,report['plan'])
  while seg.elapsed<seg.duration+.1:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
 print(json.dumps({'hardware_commands':False,'hover_error_mm':error*1000,'stages':[r['stage'] for r in tail],'gap_mm':profile['closing_gap_m']*1000,'status':'passed ideal-feedback replay'}))
finally:p.close()
