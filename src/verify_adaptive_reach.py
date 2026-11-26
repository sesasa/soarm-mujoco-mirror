"""Hardware-free replay from an archived scene with a synthetically moved box.

No camera or motor connection. Produces evidence under captures/adaptive_reach_test.
"""
import json,time
from pathlib import Path
import cv2,numpy as np
from preview_camera_alignment import Preview,ROOT
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from grasp_target import refine_packet_contours
from batch_grasp import prepare_batch,table_targets
from basket_tracking import measure_basket
from command_mapping import Mapping,DirectTrajectory,DirectSegment
from adaptive_reach import observe_regions,TargetWatch,planning_result_current


def run():
    source=ROOT/'captures/mirror_grasp/1789857210115922940/planning_5/captured_state.json'
    capture=json.loads(source.read_text());state=capture['state']
    if capture['completed']!=['packet']:raise ValueError('Expected archived second-object scene')
    frame=cv2.imread(str(source.parent/'planning_scene.jpg'))
    p=Preview();out=ROOT/'captures/adaptive_reach_test';out.mkdir(exist_ok=True)
    try:
        plane=Plane(p.K,p.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
        roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),p.K,p.dist,P=p.K).reshape(-1,2).round().astype(np.int32)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
        detector=Changes(mask,stable_seconds=.12);detector.reset(cv2.imread(str(ROOT/'captures/scene_changes/baseline.png')))
        def detect(image):
            for _ in range(5):cs,_=detector.update(image,.2)
            cs=refine_packet_contours(image,refine_current_contours(image,cs),json.loads((ROOT/'config/grasp_target.json').read_text()))
            return cs,[plane.footprint(c) for c in cs]
        cs,rs=detect(frame);mapping=Mapping()
        start=time.monotonic()
        answer=prepare_batch(p.model,mapping,plane,state,frame,cs,rs,p.T,capture['completed'],preview_remaining=False)
        original_seconds=time.monotonic()-start
        selected=min(range(len(rs)),key=lambda i:np.linalg.norm(rs[i][0]-np.array(answer['region'][0])))
        original=observe_regions(frame,cs,rs)[selected];watch=TargetWatch(original,0.)
        # Generate a changed scene fixture from the actual box pixels, not a new live frame.
        contour=np.asarray(cs[selected],np.int32);object_mask=np.zeros(mask.shape,np.uint8);cv2.fillPoly(object_mask,[contour],255)
        erase=cv2.dilate(object_mask,np.ones((5,5),np.uint8))
        moved=cv2.inpaint(frame,erase,5,cv2.INPAINT_TELEA)
        translation=np.float32([[1,0,-10],[0,1,0]])
        shifted_mask=cv2.warpAffine(object_mask,translation,(640,480))
        shifted=cv2.warpAffine(frame,translation,(640,480))
        moved[shifted_mask>0]=shifted[shifted_mask>0]
        new_cs,new_rs=detect(moved);observations=observe_regions(moved,new_cs,new_rs)
        for i in range(1,20):watch.update(observations,i*.05)
        if not watch.stable(.95) or not watch.moved():raise AssertionError(f'Moved box not stably tracked: {watch.problem}')
        assert watch.action('approach',.95)=='pause'
        # Ideal-feedback replay up to half of the original approach.
        tr=DirectTrajectory(mapping,state['command']);q=list(state['command'])
        for report in answer['plans']:
            seg=DirectSegment(tr,report['plan'])
            limit=seg.duration*.5 if report['stage']=='approach' else seg.duration+.1
            while seg.elapsed<limit:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
            if report['stage']=='approach':break
        held={**state,'state':'ACTIVE','q':list(q),'command':list(q),'segment_id':None}
        start=time.monotonic()
        replacement=prepare_batch(p.model,mapping,plane,held,moved,new_cs,new_rs,p.T,capture['completed'],preview_remaining=False)
        replan_seconds=time.monotonic()-start
        selected=min(range(len(new_rs)),key=lambda i:np.linalg.norm(new_rs[i][0]-np.array(replacement['region'][0])))
        planned=observations[selected]
        assert planning_result_current(held,replacement['plans'][0]['plan']['start'],watch,planned,.95)
        assert not planning_result_current(held,replacement['plans'][0]['plan']['start'],watch,original,.95)
        for report in replacement['plans']:
            seg=DirectSegment(tr,report['plan'])
            while seg.elapsed<seg.duration+.1:q=tr.step(seg.step(1/30),q,1/30,speed_limit=seg.speed_limit)
        report={'source':str(source.relative_to(ROOT)),'hardware_commands':False,'image_modification':'Synthetic box shift -10 pixels in overhead x; actual scene was not modified',
                'target_shift_m':float(np.linalg.norm(planned.center-original.center)),
                'original_planning_seconds':original_seconds,'adaptive_replanning_seconds':replan_seconds,
                'interrupted_at':'halfway through original approach, ideal feedback',
                'stale_target_plan_rejected':True,'replacement_replay_passed':True,
                'replacement_stages':[r['stage'] for r in replacement['plans']],
                'limitations':'Saved-scene synthetic image and ideal joint feedback; no live perception or physical-motion validation.'}
        cv2.imwrite(str(out/'original.jpg'),frame);cv2.imwrite(str(out/'synthetically_moved.jpg'),moved)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'replacement_plan.json').write_text(json.dumps(replacement['plans'],indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:p.close()

if __name__=='__main__':run()
