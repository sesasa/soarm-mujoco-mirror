"""Read-only preflight from the running UI's saved overhead image and feedback."""
import json,time
from pathlib import Path
import cv2,numpy as np
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from grasp_target import select_target,refine_packet_contours
from mirror_grasp_plan import build_sequence
from command_mapping import Mapping,DirectTrajectory,DirectSegment,ROOT

def main():
 p=Preview()
 try:
  path=max((ROOT/'captures/mirror_grasp').glob('*/latest.jpg'),key=lambda x:x.stat().st_mtime)
  if time.time()-path.stat().st_mtime>3:raise RuntimeError('Camera snapshot is stale')
  for _ in range(10):
   data=path.read_bytes()
   if data.endswith(b'\xff\xd9'):break
  else:raise RuntimeError('Incomplete camera snapshot')
  im=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)[:,:640]
  pl=Plane(p.K,p.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
  roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),p.K,p.dist,P=p.K).reshape(-1,2).round().astype(np.int32)
  mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255);mask[:65]=0;mask[420:]=0
  d=Changes(mask);d.reset(cv2.imread(str(ROOT/'captures/scene_changes/baseline.png')));profile=json.loads((ROOT/'config/grasp_target.json').read_text())
  for _ in range(5):cs,_=d.update(im,.2)
  cs=refine_packet_contours(im,refine_current_contours(im,cs),profile);rs=[pl.footprint(c) for c in cs];_,i,_=select_target(im,cs,rs,profile)
  if i is None:raise RuntimeError('Packet not uniquely visible')
  s=json.loads((ROOT/'calibration/mirror_grasp_live.json').read_text())
  if time.time()-s['recorded_at_unix']>3:raise RuntimeError('Feedback snapshot stale')
  if s.get('replay_running'):raise RuntimeError('Wait for active motion to finish')
  if s['state']=='MONITOR':s['command']=s['q']
  (ROOT/'calibration/current_packet_check_input.json').write_text(json.dumps({'state':s,'region':[v.tolist() for v in rs[i]]},indent=2))
  print('Packet center:',rs[i][0].tolist(),flush=True)
  from basket_tracking import measure_basket
  basket=measure_basket(im,json.loads((ROOT/'config/basket_reference.json').read_text()),pl,p.T)
  print('Fresh basket:',basket['center_base_m'],basket['detection'],flush=True)
  plans=build_sequence(p.model,Mapping(),pl,s,rs[i],basket_config=basket);q=s['command'];t=DirectTrajectory(Mapping(),q)
  for r in plans:
   seg=DirectSegment(t,r['plan'])
   while seg.elapsed<seg.duration+.1:
    q=t.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit);t.command_mapping.to_raw(q)
   print(r['stage'],'passes',r.get('orientation_change_deg',''),flush=True)
  (ROOT/'calibration/new_object_grasp_preflight.json').write_text(json.dumps(plans,indent=2))
 finally:p.close()
if __name__=='__main__':main()
