"""One object: real-camera following, real motor streaming, then grasp and hold.

A enables holding. Click an object if needed; P prepares once; G follows + grips.
Space stops. X releases. No basket, placement, lift or return sequence.
"""
import argparse,json,queue,re,signal,subprocess,threading,time
import cv2,numpy as np
from command_mapping import ROOT,Mapping
from mirror_robot import PYTHON
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from grasp_target import refine_packet_contours
from adaptive_reach import TargetWatch,observe_regions,pose_matches,pause_acknowledged,committed_grip_valid,FinalApproachMemory
from mirror_grasp_plan import build_sequence
from grasp_tail import build_tail
from replay_settling import endpoint_settled
from live_grip import WarmGraspIK,ServoClearance,hover_settled
from camera_visibility import CameraVisibility

WINDOW='LIVE REACH AND GRIP'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrist-setup',action='store_true',help='Dual view; hold at hover for wrist commissioning, without descent/closure')
    parser.add_argument('--named-reach',action='store_true',help='Label objects and request reaches by name; no automatic grasp')
    parser.add_argument('--gpt-reach',action='store_true',help='Identify an overhead object from text using OpenAI, then prepare reach')
    parser.add_argument('--gpt-grip',action='store_true',help='GPT selection followed by reach, descent, close and hold')
    parser.add_argument('--gpt-place',action='store_true',help='GPT selection, grasp, fresh plate placement and sitting return')
    parser.add_argument('--wrist-view',action='store_true',help='Observe wrist camera alongside overhead control; no wrist steering')
    parser.add_argument('--gpt-fixed-place',action='store_true',help='Preplan and replay the entire GPT pick/place; ignore object occlusion during execution')
    parser.add_argument('--voice',action='store_true',help='Listen for spoken pick-and-place commands while idle')
    args=parser.parse_args()
    if args.voice and not args.gpt_fixed_place:parser.error('--voice requires --gpt-fixed-place')
    voice=None;voice_muted=False
    if args.gpt_fixed_place:
        args.gpt_place=True;args.wrist_view=False;args.wrist_setup=False
    if args.gpt_place:args.gpt_grip=True
    if args.gpt_grip:args.gpt_reach=True
    if args.gpt_reach:args.named_reach=True
    recognition_answers=queue.Queue();recognition_pending=None;recognition_busy=False;auto_plan=False;auto_execute=False;auto_go=False;arm_requested_at=None
    if args.named_reach and args.wrist_setup:parser.error('Choose named reach or wrist setup')
    from named_targets import NamedTargets
    names=NamedTargets();entry=None;entry_text='';active_name=None
    wrist=None;wrist_tracker=None;wrist_sample=None;wrist_features=None;tips=[];drag=[];wrist_events=[]
    preview=Preview();mapping=Mapping();cap=None;proc=None;log=None;camera_thread=None;stop=threading.Event()
    folder=ROOT/'captures/live_grip'/str(time.time_ns());folder.mkdir(parents=True)
    frames=queue.Queue(maxsize=1);states=queue.Queue(maxsize=1);answers=queue.Queue()
    phase='IDLE';notice='A holds. Click an object, P prepares, G follows and grips.'
    selected=None;watch=None;planned_obs=None;warm=None;profile=None;plans=None;index=0
    busy=False;generation=0;pause_token=0;pause_after=None;pause_time=0.
    last_send=last_save=last_solve=0.;last_frame_time=None;tracking_since=None;tail_started=None
    from wrist_local import WristContinuation
    wrist_continuation=WristContinuation()
    grip_deadline=None;approach_memory=FinalApproachMemory()
    click=[];last_ik_ms=None;target_position=None;detector=None
    try:
        plane=Plane(preview.K,preview.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
        visibility=CameraVisibility(ServoClearance(preview.model).checker)
        roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),preview.K,preview.dist,P=preview.K).reshape(-1,2).round().astype(np.int32)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
        detector=Changes(mask,stable_seconds=.06);detector.reset(cv2.imread(str(ROOT/'captures/scene_changes/baseline.png')))
        base_profile=json.loads((ROOT/'config/grasp_target.json').read_text())
        if args.wrist_setup or args.wrist_view:
            from wrist_local import WristView,PatchTracker
            cameras=json.loads((ROOT/'config/cameras.json').read_text())['cameras']
            wrist=WristView(next(c for c in cameras if c['role']=='wrist'));wrist_tracker=PatchTracker()
        spec=preview.meta['camera'];cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Close the other camera window first')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,640);cap.set(4,480);cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        def camera_reader():
            while not stop.is_set():
                ok,frame=cap.read();sample=(time.monotonic(),frame if ok else None)
                if frames.full():
                    try:frames.get_nowait()
                    except queue.Empty:pass
                frames.put(sample)
                if not ok:break
        camera_thread=threading.Thread(target=camera_reader,daemon=True);camera_thread.start()
        proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'src/control_worker.py'),'--direct','--visual-servo'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def state_reader():
            for line in proc.stdout:
                try:
                    value=json.loads(line)
                    if states.full():states.get_nowait()
                    states.put_nowait(value)
                except (ValueError,queue.Empty,queue.Full):pass
        threading.Thread(target=state_reader,daemon=True).start()
        def send(op,**kw):
            proc.stdin.write(json.dumps({'op':op,'time':time.monotonic(),**kw})+'\n');proc.stdin.flush()
        def pause(after):
            nonlocal phase,pause_token,pause_after,pause_time
            pause_token+=1;pause_after=after;pause_time=time.monotonic();phase='PAUSING'
            send('direct_pause',pause_id=pause_token)
        def halt(reason):
            nonlocal phase,generation,plans,tracking_since,wrist_continuation,auto_execute,auto_go,auto_plan,arm_requested_at
            wrist_continuation=WristContinuation()
            auto_execute=False;auto_go=False;auto_plan=False;arm_requested_at=None
            send('hold',reason=reason);phase='STOPPED';generation+=1;plans=None;tracking_since=None
        def launch(kind,obs,st):
            nonlocal busy,generation,phase
            busy=True;generation+=1;job=generation;captured=json.loads(json.dumps(st));job_profile=dict(profile)
            scene_image=real.copy() if args.gpt_place else None
            def calculate():
                try:
                    if args.gpt_place:
                        from gpt_pick_place import current_plate,build_pick_place_tail
                        cv2.imwrite(str(folder/f'{job}_plate_scene.jpg'),scene_image)
                        plate=current_plate(scene_image,plane,preview.T)
                        (folder/f'{job}_plate.json').write_text(json.dumps(plate,indent=2))
                    if kind=='prepare':
                        value=build_sequence(preview.model,mapping,plane,captured,obs.region,reach_only=True,target_profile_override=job_profile)
                        if args.gpt_place:
                            approach=next(p for p in value if p['stage']=='approach')
                            tail=build_pick_place_tail(preview.model,mapping,plane,approach['plan']['goal'],obs.region,job_profile,plate)
                            from basket_place import BasketClearance,trajectory
                            from table_clearance import TableClearance
                            up=plane.base_from_board[:3,2].copy()
                            if up[2]<0:up=-up
                            guard=BasketClearance(TableClearance(preview.model,plane.base_from_board[:3,3],up),plate)
                            for report in value:
                                for t in np.linspace(0,report['plan']['duration'],81):guard.check_robot(trajectory(report['plan'],t))
                            complete=value+tail
                            if args.gpt_fixed_place:
                                from gpt_pick_place import accelerate_fixed_playback
                                complete=accelerate_fixed_playback(complete);value=complete
                            (folder/'complete_pick_place_preview.json').write_text(json.dumps(complete,indent=2))
                    elif args.gpt_place:
                        value=build_pick_place_tail(preview.model,mapping,plane,captured['command'],obs.region,job_profile,plate)
                    else:
                        value=build_tail(preview.model,mapping,plane,np.asarray(captured['command']),closing_gap=job_profile['closing_gap_m'],lift_height=0,descent_m=job_profile.get('descent_m',.025))
                    answers.put((job,kind,obs,captured,value))
                except Exception as e:answers.put((job,kind,obs,captured,e))
            threading.Thread(target=calculate,daemon=True).start()
            phase='PLANNING' if kind=='prepare' else 'GRIP_PLANNING'
        state=states.get(timeout=15);previous_state=state['state']
        if args.voice:
            from voice_commands import VoiceCommands
            voice=VoiceCommands()
        log=(folder/'tracking.jsonl').open('a',buffering=1)
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1440,420)
        def mouse(event,x,y,flags,param):
            if x<640 and event==cv2.EVENT_LBUTTONDOWN:click.append((x,y))
            elif wrist is not None and x>=640:
                point=(x-640,y)
                if event==cv2.EVENT_RBUTTONDOWN:wrist_events.append(('tip',point))
                if event==cv2.EVENT_LBUTTONDOWN:drag[:]=[point]
                if event==cv2.EVENT_LBUTTONUP and drag:
                    wrist_events.append(('select',(*drag[0],*point)));drag.clear()
        cv2.setMouseCallback(WINDOW,mouse)
        print(f'{WINDOW}: A hold | click object | P prepare | G follow + grip | Space stop | X release\n{folder}',flush=True)
        while True:
            try:frame_time,raw=frames.get(timeout=.15)
            except queue.Empty:
                if phase in ('FOLLOW','PREFIX','GRIP','PAUSING','GRIP_PLANNING','TARGET_HIDDEN','FIXED'):halt('Camera stream lost')
                key=cv2.waitKey(1)&255
                if key in (ord('x'),ord('X')):voice_muted=True;send('release')
                if key==32:halt('Stopped by user')
                if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
                continue
            if raw is None:raise RuntimeError('Camera disconnected')
            now=time.monotonic()
            try:state=states.get_nowait()
            except queue.Empty:pass
            if proc.poll() is not None or now-state['time']>1:raise RuntimeError('Motor feedback lost')
            if state['state']!='ACTIVE' and previous_state=='ACTIVE':
                if phase!='DONE':phase='STOPPED'
                generation+=1;plans=None;auto_execute=False;auto_go=False;auto_plan=False;arm_requested_at=None;notice=state['reason']
            if state['state']=='MONITOR' and not state['reason'].startswith('Ready:'):notice=state['reason']
            if phase=='ARMING':
                if state['state']=='ACTIVE':
                    phase='IDLE';auto_plan=True;arm_requested_at=None;notice='Holding enabled; preparing the complete sequence.'
                elif arm_requested_at is not None and now-arm_requested_at>2:
                    phase='IDLE';auto_execute=False;arm_requested_at=None;notice='Could not enable holding: '+state['reason']
            previous_state=state['state']
            if now-frame_time>.2:
                if phase=='FOLLOW':halt('Camera processing delayed')
                continue
            dt=.1 if last_frame_time is None else frame_time-last_frame_time;last_frame_time=frame_time
            real=cv2.remap(raw,*preview.maps,cv2.INTER_LINEAR)
            projected=visibility.projected_mask(np.asarray(state['q']))
            excluded=cv2.dilate(projected,np.ones((7,7),np.uint8))
            contours,_=detector.update(real,dt,excluded)
            contours=refine_packet_contours(real,refine_current_contours(real,contours),base_profile)
            regions=[plane.footprint(c) for c in contours];observations=observe_regions(real,contours,regions)
            names.update(observations,frame_time)
            if recognition_pending is not None:
                for candidate in recognition_pending['watches']:candidate.update(observations,frame_time)
            try:
                recognition_job,value=recognition_answers.get_nowait();recognition_busy=False
                pending=recognition_pending;recognition_pending=None
                if pending is None or recognition_job!=generation or phase!='RECOGNIZING':pass
                elif isinstance(value,Exception):phase='IDLE';auto_execute=False;notice=str(value)
                elif now-pending['started']>30:phase='IDLE';auto_execute=False;notice='Recognition expired. Submit a fresh request.'
                else:
                    candidate_id,description,marked=value
                    candidate=pending['watches'][candidate_id]
                    if not candidate.stable(frame_time):
                        phase='IDLE';auto_execute=False;notice='Selected object moved, disappeared or became ambiguous; submit again.'
                    else:
                        watch=candidate;active_name=pending['prompt'];plans=None;phase='IDLE'
                        cv2.imwrite(str(folder/'gpt_selection.jpg'),marked)
                        (folder/'gpt_selection.json').write_text(json.dumps({'prompt':active_name,'candidate_id':candidate_id+1,'description':description,'region':[v.tolist() for v in watch.current.region]},indent=2))
                        if state['state']=='ACTIVE':auto_plan=True
                        elif auto_execute:
                            send('arm');phase='ARMING';arm_requested_at=now;notice='Object identified; enabling holding and planning.'
                        else:notice='GPT selected '+description+'. A holds, then P plans.'
            except queue.Empty:pass
            if phase in ('IDLE','PREVIEW','STOPPED','REACHED'):
                if click:
                    point=click[-1];click.clear()
                    matches=[i for i,c in enumerate(contours) if cv2.pointPolygonTest(np.asarray(c,np.float32),point,False)>=0]
                    if len(matches)==1:
                        active_name=None;auto_execute=False;auto_go=False
                        selected=matches[0];watch=TargetWatch(observations[selected],frame_time);phase='IDLE';plans=None;notice='Object selected. P prepares its reach.'
                elif watch is None and not args.gpt_reach:
                    choices=[i for i,o in enumerate(observations) if .025<=o.size[0]<=.08 and .04<=o.size[1]<=.18]
                    if len(choices)==1:selected=choices[0];watch=TargetWatch(observations[selected],frame_time)
            if watch is not None:watch.update(observations,frame_time)
            try:
                job,kind,obs,captured,value=answers.get_nowait();busy=False
                if job!=generation:pass
                elif isinstance(value,Exception):halt('Planning refused: '+str(value));notice=str(value)
                elif state['state']!='ACTIVE' or max(abs(np.asarray(state['command'])-captured['command']))>1e-6:
                    halt('Arm changed while planning');notice='Press A and P again.'
                elif kind=='prepare':
                    plans=value;planned_obs=obs
                    warm=WarmGraspIK(preview.model,mapping,plane,profile)
                    approach=next(p for p in plans if p['stage']=='approach')
                    warm.reset(approach['plan']['goal'],obs.region)
                    phase='PREVIEW';notice=('Ready: '+str(active_name or 'selected object')+('. G grasps, places in plate, then sits.' if args.gpt_place else '. G reaches, descends and grips.' if args.gpt_grip else '. G reaches and holds; no grasp.') if args.named_reach else 'Ready. G reaches and holds for wrist setup.' if args.wrist_setup else 'Ready. G follows this object, then grips when aligned and stable.')
                    (folder/'initial_plan.json').write_text(json.dumps(plans,indent=2))
                    if auto_execute:auto_go=True
                elif not (watch.accepts_plan(obs,frame_time) or (args.gpt_grip and committed_grip_valid(watch,frame_time,grip_deadline) and pose_matches(watch.current,obs))):
                    phase='FOLLOW';notice='Object changed before descent; continuing to follow.'
                else:
                    plans=value;index=0;tail_started=None;planned_obs=obs;watch.commit(obs);phase='GRIP'
                    if args.gpt_grip:grip_deadline=now+sum(p['plan'].get('playback_duration',p['plan']['duration'])+6 for p in plans)
                    send('direct_segment',plan=plans[0]['plan'],segment_id=100)
                    notice=('Descending and closing at the confirmed target; temporary object occlusion is allowed.' if args.gpt_grip else 'Descending, then closing. Object must remain visible and stationary.')
                    (folder/'grip_plan.json').write_text(json.dumps(plans,indent=2))
            except queue.Empty:pass
            if phase=='PAUSING':
                if now-pause_time>2:halt('Worker pause acknowledgement timed out')
                elif pause_acknowledged(state,pause_token):
                    if pause_after=='wrist':
                        phase='WRIST_SETUP';notice='Holding at hover. Select the wrist object and mark the two fingertips. S saves; X releases.'
                    elif pause_after=='named':
                        phase='REACHED';notice='Reached '+str(active_name or 'object')+'. Holding above it. Enter another command, or X releases.'
                    elif pause_after=='grip':
                        if watch.can_descend(frame_time) or (args.gpt_grip and committed_grip_valid(watch,frame_time,grip_deadline)):launch('grip',watch.current,state)
                        else:phase='FOLLOW'
                    else:phase=('FOLLOW' if watch is not None and watch.fresh(frame_time) else 'TARGET_HIDDEN');tracking_since=None
            if wrist is not None:
                wrist_sample=wrist.latest()
                if wrist_sample is not None and wrist_sample[1] is not None:
                    wt,wf=wrist_sample
                    for event,value in wrist_events:
                        if event=='tip':
                            if len(tips)==2:tips.clear()
                            tips.append(value)
                        elif time.monotonic()-wt<=.25:
                            wrist_tracker.select(wf,value,wt)
                            wrist_continuation=WristContinuation()
                    wrist_events.clear()
                    wrist_features=wrist_tracker.update(wf,wt,time.monotonic())
                else:wrist_features=None
            wrist_active=wrist_continuation.observation is not None
            wrist_ok=wrist_active and wrist_sample is not None and wrist_continuation.valid(wrist_features,wrist_sample[0],time.monotonic())
            if wrist_active and phase in ('FOLLOW','PAUSING','GRIP_PLANNING','GRIP'):
                in_contact=phase=='GRIP' and plans[index]['stage'] not in ('lower',)
                if not in_contact and not wrist_ok:
                    halt('Wrist continuation timed out before settling; holding' if time.monotonic()>wrist_continuation.until else 'Wrist patch lost or frame stale; holding')
                elif watch.fresh(frame_time) and not pose_matches(watch.current,wrist_continuation.observation):
                    if not in_contact:halt('Overhead sees target displacement; holding')
            if phase=='TARGET_HIDDEN' and watch is not None and watch.fresh(frame_time):phase='FOLLOW'
            if phase in ('PREFIX','FOLLOW') and args.gpt_grip:approach_memory.update(watch,frame_time)
            if phase=='FOLLOW':
                remaining=float('inf')
                if args.gpt_grip and watch is not None and warm is not None:
                    remaining=float(np.linalg.norm(warm.solver.position(np.asarray(state['q']))-warm.target(watch.current.region)))
                occluded_finish=args.gpt_grip and (approach_memory.permits(watch,frame_time,remaining) or wrist_ok)
                if (watch is None or not watch.fresh(frame_time)) and not occluded_finish:
                    pause('follow');notice='Target obscured or ambiguous; holding until it is visible.'
                elif now-last_solve>=.075:
                    begin=time.monotonic()
                    try:
                        goal,info=warm.solve((wrist_continuation.observation if wrist_ok else watch.current).region);last_ik_ms=1000*(time.monotonic()-begin)
                        target_position=np.asarray(info['target'])
                        if time.monotonic()-frame_time>.2:raise ValueError('Vision/IK update too old')
                        send('servo_target',q=goal.tolist(),frame_time=frame_time);last_solve=now
                        error=np.linalg.norm(warm.solver.position(np.asarray(state['q']))-target_position)
                        aligned=hover_settled(error,state['q'],goal,.008 if args.gpt_grip else .004)
                        if aligned and (watch.stable(frame_time) or occluded_finish):
                            if tracking_since is None:tracking_since=now
                            if now-tracking_since>.8:
                                if wrist_ok:watch.current=wrist_continuation.observation
                                watch.commit(watch.current)
                                if args.named_reach and not args.gpt_grip:
                                    pause('named');notice='Reach complete; holding above the selected object.'
                                elif args.wrist_setup:
                                    pause('wrist');notice='Reaching complete; holding for wrist alignment setup.'
                                else:
                                    if args.gpt_grip:grip_deadline=now+15.
                                    pause('grip');notice='Aligned and stationary; preparing final descent and grip.'
                        else:tracking_since=None
                        if phase=='FOLLOW':notice=(f'Target hidden: finishing approach to last stable position | {error*1000:.0f} mm remaining' if occluded_finish else f'Following real object | tip error {error*1000:.0f} mm | IK {last_ik_ms:.1f} ms')
                    except ValueError as e:halt(str(e));notice=str(e)
            if phase in ('PREFIX','GRIP','FIXED'):
                stage=plans[index]['stage'];sid=index if phase=='PREFIX' else 100+index
                grip_valid=committed_grip_valid(watch,frame_time,grip_deadline) if args.gpt_grip else watch.can_descend(frame_time)
                if phase!='FIXED' and stage=='lower' and not grip_valid:
                    halt('Object moved or disappeared during descent');notice='Stopped before gripping. A then P starts again.'
                elif state.get('segment_id')==sid and float(state.get('segment_elapsed') or 0)>=plans[index]['plan']['duration']:
                    if tail_started is None:tail_started=now
                    error=abs(np.asarray(state['q'])-np.asarray(plans[index]['plan']['goal']))
                    if endpoint_settled(stage,error):
                        if phase=='PREFIX' and index==len(plans)-1:pause('follow')
                        elif phase in ('GRIP','FIXED') and index==len(plans)-1:
                            send('hold',reason='Placement complete; holding sitting pose' if args.gpt_place else 'Grip complete: holding object; X releases torque');phase='DONE';auto_execute=False;notice=('Placement sequence complete; holding sitting pose. A to start another selection; X releases.' if args.gpt_place else 'Grip complete. Holding here; no basket or return.')
                        elif phase=='GRIP' and stage=='lower' and not grip_valid:
                            halt('Target no longer stable before closure')
                        else:
                            index+=1;tail_started=None
                            if args.gpt_place:notice='Executing '+plans[index]['stage'].replace('_',' ')+'. Space stops; X releases.'
                            send('direct_segment',plan=plans[index]['plan'],segment_id=index if phase=='PREFIX' else 100+index)
                    elif now-tail_started>5:halt('Motion endpoint did not settle')
            if now-last_send>.06:send('heartbeat');last_send=now
            # Full GL rendering costs ~300 ms on this desktop. Use cheap FK
            # projection so the visual target stream is not blocked by display.
            overlay=real.copy();pixels=projected.astype(bool)
            overlay[pixels]=(overlay[pixels]*.65+np.array([220,160,20])*.35).astype(np.uint8)
            if target_position is not None:
                camera=preview.T[:3,:3]@target_position+preview.T[:3,3]
                if camera[2]>.005:
                    uv=preview.K@camera;uv=np.rint(uv[:2]/uv[2]).astype(int)
                    cv2.circle(overlay,tuple(uv),6,(0,255,0),2)
            if wrist is not None:
                wrist_sample=wrist.latest()
                if wrist_sample is not None and wrist_sample[1] is not None:
                    wt,wf=wrist_sample
                    overlay=wrist_tracker.draw(wf,tips)
                    if args.wrist_view:
                        cv2.rectangle(overlay,(0,65),(640,143),(20,20,20),-1)
                        cv2.putText(overlay,'WRIST: drag over object lettering; V continues stored reach',(8,91),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,255),1)
                        cv2.putText(overlay,'Patch visibility check only; no wrist position correction',(8,117),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                    if time.monotonic()-wt>.25:
                        cv2.putText(overlay,'WRIST STALE',(150,230),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                else:
                    wrist_features=None
                    overlay=np.zeros_like(real)
                    cv2.putText(overlay,'Wrist unavailable',(100,220),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            result=np.hstack([real,overlay])
            cv2.rectangle(result,(0,0),(1920,65),(20,20,20),-1)
            if args.named_reach:
                cv2.drawContours(result,contours,-1,(140,140,140),1)
                for name,nwatch in names.targets.items():
                    if nwatch.fresh(frame_time):
                        c=np.rint(plane.projected_footprint(nwatch.current.region)).astype(np.int32)
                        cv2.polylines(result,[c],True,(255,180,0),2)
                        xy=tuple(np.mean(c,axis=0).astype(int))
                        cv2.putText(result,name,xy,cv2.FONT_HERSHEY_SIMPLEX,.6,(255,180,0),2)
                prompt=('NAME: ' if entry=='label' else 'COMMAND: ')+entry_text+'_' if entry else 'Click object, N to name | Enter to type: reach for the highlighter | G executes reach'
                if args.gpt_reach and entry is None:prompt='GPT: Enter, type matchbox, Enter to identify | A hold | P plan | '+('G pick + place' if args.gpt_place else 'G grasp' if args.gpt_grip else 'G reach')
                cv2.rectangle(result,(0,440),(1280,480),(20,20,20),-1)
                cv2.putText(result,prompt,(8,466),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            if args.gpt_fixed_place and entry is None:
                cv2.rectangle(result,(0,440),(1280,480),(20,20,20),-1)
                cv2.putText(result,'Enter: pick and place the highlighter -> identify, plan and execute automatically',(8,466),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            action='G PICK + PLACE' if args.gpt_place else 'G REACH + GRIP' if args.gpt_grip else 'G REACH ONLY' if args.named_reach else 'G REACH TO WRIST SETUP' if args.wrist_setup else 'G FOLLOW + GRIP'
            cv2.putText(result,f'{state["state"]} / {phase} | A hold | click target | P prepare | {action} | Space stop | X release',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,255,255),1)
            cv2.putText(result,notice,(8,50),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
            if watch is not None and watch.fresh(frame_time):
                c=np.rint(plane.projected_footprint(watch.current.region)).astype(np.int32)
                cv2.polylines(result,[c],True,(0,255,0),2)
            if voice is not None:
                idle=phase in ('IDLE','PREVIEW','STOPPED','REACHED','DONE') and not busy and not recognition_busy and entry is None
                voice.listen(idle and not voice_muted)
                cv2.rectangle(result,(0,410),(1280,440),(20,20,20),-1)
                cv2.putText(result,voice.status+' | M mutes microphone',(8,432),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow(WINDOW,result);key=cv2.waitKey(1)&255
            if voice is not None and entry is None and key in (ord('m'),ord('M')):
                voice_muted=not voice_muted;voice.listen(False);key=255
            if voice is not None and key==255:
                spoken=voice.poll()
                if spoken is not None:
                    entry='command';entry_text=spoken;key=13
                    voice.listen(False)
                    with (folder/'voice_commands.jsonl').open('a') as voice_log:voice_log.write(json.dumps({'voice_command':spoken,'recorded_at_unix':time.time()})+'\n')
            if auto_go:
                if key==255:key=ord('g')
                auto_go=False
            if auto_plan:
                if key==255:key=ord('p')
                auto_plan=False
            if args.named_reach and entry is not None:
                typed=key;key=255
                if typed==27:entry=None;entry_text=''
                elif typed in (8,127):entry_text=entry_text[:-1]
                elif typed in (10,13):
                    try:
                        if entry=='label':
                            if watch is None or not watch.stable(frame_time):raise ValueError('Click a visible, stationary object first')
                            active_name=names.label(entry_text,watch.current,frame_time)
                            notice='Named '+active_name+'. Label another object, or Enter to request a reach.'
                        elif args.gpt_reach:
                            if recognition_busy:raise ValueError('Previous recognition is still finishing; try again shortly')
                            if not observations:raise ValueError('No detected objects in the overhead view')
                            from gpt_targets import recognize
                            from pick_place_command import parse_pick_place_command
                            execute_intent,recognition_text=parse_pick_place_command(entry_text)
                            if execute_intent and not args.gpt_fixed_place:raise ValueError('Automatic pick and place requires fixed pick-and-place mode')
                            auto_execute=execute_intent;auto_go=False;auto_plan=False
                            generation+=1;recognition_job=generation;plans=None;watch=None;active_name=None;phase='RECOGNIZING';recognition_busy=True
                            recognition_pending={'watches':[TargetWatch(o,frame_time) for o in observations],'prompt':entry_text,'started':now}
                            def identify(job,frame,cs,text):
                                try:value=recognize(frame,cs,text)
                                except Exception as exc:value=exc
                                recognition_answers.put((job,value))
                            threading.Thread(target=identify,args=(recognition_job,real.copy(),[c.copy() for c in contours],recognition_text),daemon=True).start()
                            notice=('GPT is identifying the object; the full pick-and-place will then run automatically.' if auto_execute else 'GPT is identifying the requested object; G starts the prepared plan.')
                        else:
                            active_name,watch=names.resolve(entry_text,frame_time)
                            phase='IDLE';plans=None
                            if state['state']!='ACTIVE':notice='Selected '+active_name+'. Press A to hold, then P to plan.'
                            else:key=ord('p')
                        entry=None;entry_text=''
                    except ValueError as exc:notice=str(exc)
                elif 32<=typed<=126 and len(entry_text)<100:entry_text+=chr(typed)
            elif args.named_reach and key in (ord('n'),ord('N'),10,13):
                if phase not in ('IDLE','PREVIEW','STOPPED','REACHED','DONE') or busy:
                    notice='Finish or stop the current reach before selecting another object.'
                else:
                    entry='label' if key in (ord('n'),ord('N')) else 'command';entry_text=''
                key=255
            if key in (ord('a'),ord('A')) and state['state'] in ('MONITOR','HOLD') and phase!='RECOGNIZING':
                send('arm');phase='IDLE';generation+=1;plans=None;notice='Holding requested. Select object and press P.'
            if key in (ord('p'),ord('P')):
                if state['state']!='ACTIVE':notice='P needs A to hold first: '+state['reason']
                elif phase not in ('IDLE','PREVIEW','STOPPED','REACHED','DONE') or busy:notice='Wait for the current operation, or Space to stop.'
                elif watch is None or not watch.stable(frame_time):notice='Click one clearly visible object and let it settle briefly.'
                else:
                    width,length=watch.current.size
                    profile=dict(base_profile,after_grasp='hold')
                    if args.named_reach:profile['position_only_reach']=not args.gpt_grip
                    if args.gpt_grip:
                        profile.update(preflight_grip=True,closing_gap_m=max(.005,float(width)-.004),name=active_name or 'GPT selected object')
                    if length<.08:profile.update(name='small box',closing_gap_m=max(.005,float(width)-.004))
                    if args.gpt_fixed_place:profile['closing_gap_m']=max(.005,profile['closing_gap_m']-.010)
                    if args.gpt_fixed_place and re.search(r'\b(?:biscuits?|crackers?)\b',active_name or '',re.I):
                        profile['closing_gap_m']=max(.005,profile['closing_gap_m']-.005)
                    approach_memory=FinalApproachMemory();wrist_continuation=WristContinuation()
                    launch('prepare',watch.current,state);notice=('Preplanning reach, grasp, plate placement and sitting return.' if args.gpt_place else 'Preparing the first reach; no basket planning.')
            if key in (ord('g'),ord('G')):
                if phase!='PREVIEW' or state['state']!='ACTIVE':notice='Press P first.'
                elif not watch.fresh(frame_time) or not pose_matches(watch.current,planned_obs):notice='Target changed before start; press P again.'
                elif np.max(abs(np.asarray(plans[0]['plan']['start'])-state['command']))>1e-6:notice='Arm changed; press P again.'
                else:
                    if args.gpt_fixed_place:
                        index=0;tail_started=None;phase='FIXED'
                        send('direct_segment',plan=plans[0]['plan'],segment_id=100);notice='Executing complete fixed plan; object occlusion is ignored.'
                    else:
                        plans=[p for p in plans if p['stage']!='approach'];index=0;tail_started=None;phase='PREFIX'
                        send('direct_segment',plan=plans[0]['plan'],segment_id=0);notice='Opening and preparing; then live following.'
            if key in (ord('v'),ord('V')) and wrist is not None:
                if phase!='TARGET_HIDDEN' or state['state']!='ACTIVE' or planned_obs is None:
                    notice='V is available when paused for a hidden target. Select the object patch in the wrist view.'
                else:
                    try:
                        remaining=float(np.linalg.norm(warm.solver.position(np.asarray(state['q']))-warm.target(planned_obs.region)))
                        wrist_continuation.start(planned_obs,wrist_features,wrist_sample[0] if wrist_sample else 0,time.monotonic(),remaining)
                        watch.current=planned_obs;watch.commit(planned_obs)
                        phase='FOLLOW';tracking_since=None;notice='Wrist patch selected: continuing stored target, not wrist-guided steering.'
                    except ValueError as exc:notice=str(exc)
            if key in (ord('s'),ord('S')) and wrist is not None:
                if phase!='WRIST_SETUP' or wrist_features is None or len(tips)!=2 or time.monotonic()-wrist_sample[0]>.25:
                    notice='Save at wrist hover: select a tracked object and right-click both fingertips first.'
                else:
                    cv2.imwrite(str(folder/'wrist_reference.png'),wrist_sample[1])
                    (folder/'wrist_reference.json').write_text(json.dumps({'recorded_at_unix':time.time(),'q':state['q'],'command':state['command'],'features':wrist_features,'tips_px':tips,'patch_box_px':wrist_tracker.box.tolist(),'camera_time':wrist_sample[0],'validated_for_control':False,'note':'Hover observation only; not a calibrated grasp target or motion map'},indent=2))
                    notice='Wrist observation saved. This is not yet a calibrated grasp target; X releases.'
            if key==32:voice_muted=True;halt('Stopped by user');notice='Stopped; no automatic resume. A and P start again.'
            if key in (ord('x'),ord('X')):voice_muted=True;auto_execute=False;auto_go=False;auto_plan=False;arm_requested_at=None;send('release');wrist_continuation=WristContinuation();generation+=1;phase='IDLE';plans=None;notice='Torque released.'
            if now-last_save>.25:
                sample={**state,'recorded_at_unix':time.time(),'phase':phase,'notice':notice,'ik_ms':last_ik_ms,'frame_age_s':time.monotonic()-frame_time,'wrist_setup':args.wrist_setup,'wrist_features':wrist_features,'named_reach':args.named_reach,'gpt_reach':args.gpt_reach,'gpt_grip':args.gpt_grip,'gpt_place':args.gpt_place,'fixed_plan':args.gpt_fixed_place,'automatic_command':auto_execute,'voice_status':voice.status if voice is not None else None,'active_name':active_name,'names':list(names.targets)}
                log.write(json.dumps(sample)+'\n');path=ROOT/'calibration/live_grip_live.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(sample,indent=2));temp.replace(path)
                cv2.imwrite(str(folder/'latest.jpg'),result)
                if args.wrist_view and wrist_sample is not None and wrist_sample[1] is not None and time.monotonic()-wrist_sample[0]<.25:
                    cv2.imwrite(str(folder/'wrist_latest.jpg'),wrist_sample[1])
                    if phase in ('PAUSING','TARGET_HIDDEN') and watch is not None and not watch.fresh(frame_time):
                        cv2.imwrite(str(folder/'wrist_at_occlusion.jpg'),wrist_sample[1])
                        (folder/'wrist_at_occlusion.json').write_text(json.dumps(sample,indent=2))
                last_save=now
            if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
    finally:
        if voice is not None:voice.close()
        stop.set()
        if camera_thread is not None:camera_thread.join(timeout=1)
        if cap is not None:cap.release()
        if wrist is not None:wrist.close()
        if proc is not None:
            if proc.poll() is None:
                proc.stdin.close()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.send_signal(signal.SIGINT);proc.wait(timeout=3)
            if proc.stdout:proc.stdout.close()
        if log:log.close()
        preview.close();cv2.destroyAllWindows()

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:pass
