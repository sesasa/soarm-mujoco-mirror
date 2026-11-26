"""Offline integration replay: saved object and freshly measured red plate."""
import json,time
import cv2,numpy as np
from command_mapping import Mapping,DirectTrajectory,DirectSegment
from preview_camera_alignment import Preview
from scene_changes import Plane
from mirror_grasp_plan import build_sequence
from gpt_pick_place import current_plate,build_pick_place_tail
from live_grip import WarmGraspIK,JointServo,ServoClearance
p=Preview()
try:
 m=Mapping();plane=Plane(p.K,p.T,np.array(json.load(open('captures/scene_changes/plane.json'))['T_camera_from_board']))
 f='captures/live_grip/1789923501036923903/'
 r=tuple(np.array(x) for x in json.load(open(f+'gpt_selection.json'))['region']);q=json.load(open(f+'initial_plan.json'))[0]['plan']['start']
 profile=json.load(open('config/grasp_target.json'));profile.update(preflight_grip=True,after_grasp='hold',closing_gap_m=max(.005,float(2*min(r[1][:2]))-.004))
 plate=current_plate(cv2.imread('captures/gpt_place/red_plate_reference.jpg'),plane,p.T)
 t=time.monotonic();plans=build_sequence(p.model,m,plane,{'q':q,'command':q},r,reach_only=True,target_profile_override=profile)
 preflight=build_pick_place_tail(p.model,m,plane,plans[-1]['plan']['goal'],r,profile,plate)
 print('Preflight',time.monotonic()-t,[x['stage'] for x in preflight],flush=True)
 tr=DirectTrajectory(m,q)
 for report in plans[:-1]:
  seg=DirectSegment(tr,report['plan'])
  while seg.elapsed<seg.duration+.1:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
 w=WarmGraspIK(p.model,m,plane,profile);w.reset(plans[-1]['plan']['goal'],r);servo=JointServo(q);guard=ServoClearance(p.model)
 for i in range(1,601):
  now=i/30
  if i%3==1:
   goal,info=w.solve(r);servo.update(goal,now,now)
  cmd=servo.step(1/30,now,limits=tr.limits);guard.check(q,cmd);q=tr.step(cmd,q,1/30,speed_limit=servo.speed)
 tail=build_pick_place_tail(p.model,m,plane,q,r,profile,plate)
 for report in tail:
  seg=DirectSegment(tr,report['plan'])
  while seg.elapsed<seg.duration+.1:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
 result={'hardware_commands':False,'status':'passed ideal-feedback replay','stages':[p['stage'] for p in tail],'seconds':time.monotonic()-t,'plate_height_m':plate['rim_height_m']}
 open('captures/gpt_place/offline_result.json','w').write(json.dumps(result,indent=2));print(result)
finally:p.close()
