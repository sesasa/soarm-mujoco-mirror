"""Saved-scene benchmark of streaming real-worker logic; never opens hardware."""
import json,time
import cv2,numpy as np
from command_mapping import ROOT,Mapping,DirectTrajectory,DirectSegment
from live_grip import WarmGraspIK,JointServo,ServoClearance
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from grasp_target import refine_packet_contours
from mirror_grasp_plan import build_sequence
from grasp_tail import build_tail


def run():
 p=Preview();mapping=Mapping();out=ROOT/'captures/live_grip_test';out.mkdir(exist_ok=True)
 try:
  src=ROOT/'captures/mirror_grasp/1789857210115922940/planning_5'
  state=json.loads((src/'captured_state.json').read_text())['state'];frame=cv2.imread(str(src/'planning_scene.jpg'))
  plane=Plane(p.K,p.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
  roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),p.K,p.dist,P=p.K).reshape(-1,2).round().astype(np.int32)
  mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
  detector=Changes(mask);detector.reset(cv2.imread(str(ROOT/'captures/scene_changes/baseline.png')))
  for _ in range(5):cs,_=detector.update(frame,.2)
  profile=json.loads((ROOT/'config/grasp_target.json').read_text());profile['after_grasp']='hold'
  cs=refine_packet_contours(frame,refine_current_contours(frame,cs),profile);rs=[plane.footprint(c) for c in cs]
  choices=[r for r in rs if .025<=min(2*r[1][:2])<=.08 and .04<=max(2*r[1][:2])<.08]
  if len(choices)!=1:raise ValueError(f'Expected one archived small box, got {len(choices)}')
  region=choices[0];profile.update(name='small box',closing_gap_m=max(.005,float(min(2*region[1][:2]))-.004))
  plans=build_sequence(p.model,mapping,plane,state,region,reach_only=True,target_profile_override=profile)
  approach=next(r for r in plans if r['stage']=='approach')
  warm=WarmGraspIK(p.model,mapping,plane,profile);warm.reset(approach['plan']['goal'],region)
  tr=DirectTrajectory(mapping,state['command']);q=list(state['command'])
  for report in plans:
   if report['stage']=='approach':break
   segment=DirectSegment(tr,report['plan'])
   while segment.elapsed<segment.duration+.1:q=tr.step(segment.step(1/30),q,1/30,speed_limit=segment.speed_limit)
  servo=JointServo(q);guard=ServoClearance(p.model);timings=[];checks=[];tracking=[]
  # Move the tabletop target smoothly by up to 10 mm during the reach, then settle.
  direction=plane.base_from_board[:3,0];last_goal=None
  for i in range(1,601):
   now=i/30
   shift=.01*min(1.,now/4.)
   current=(region[0]+direction*shift,region[1].copy(),region[2].copy())
   if i%3==1:
    began=time.perf_counter();last_goal,info=warm.solve(current);timings.append(time.perf_counter()-began)
    servo.update(last_goal,now,now)
   command=servo.step(1/30,now,limits=tr.limits);began=time.perf_counter();guard.check(q,command);checks.append(time.perf_counter()-began)
   q=tr.step(command,q,1/30,speed_limit=servo.speed)
   tracking.append(float(np.linalg.norm(warm.solver.position(q)-warm.target(current))))
  if tracking[-1]>.004:raise ValueError(f'Did not reach moving target: {tracking[-1]}')
  tail=build_tail(p.model,mapping,plane,np.asarray(q),closing_gap=profile['closing_gap_m'],lift_height=0,descent_m=profile['descent_m'])
  assert [r['stage'] for r in tail]==['lower','close']
  for report in tail:
   segment=DirectSegment(tr,report['plan'])
   while segment.elapsed<segment.duration+.1:q=tr.step(segment.step(1/30),q,1/30,speed_limit=segment.speed_limit)
  report={'hardware_commands':False,'scenario':'Archived second box, target shifted smoothly 10 mm over 4 seconds; ideal-feedback 20-second streaming replay',
   'ik_queries':len(timings),'ik_median_ms':float(np.median(timings)*1000),'ik_p95_ms':float(np.percentile(timings,95)*1000),
   'worker_clearance_median_ms':float(np.median(checks)*1000),'worker_clearance_p95_ms':float(np.percentile(checks,95)*1000),
   'final_hover_error_mm':tracking[-1]*1000,'grip_stages':[r['stage'] for r in tail],
   'basket_required':False,'all_streaming_commands_and_final_grip_validated':True,
   'limitations':'No live image-latency, servo-bus timing, physical tracking or grasp-success measurement.'}
  (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 finally:p.close()

if __name__=='__main__':run()
