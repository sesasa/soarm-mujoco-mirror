"""Preview and replay a simulated grasp through the direct LeRobot controller."""
import argparse,json,queue,signal,subprocess,threading,time
import cv2,numpy as np
from command_mapping import Mapping,ROOT
from mirror_robot import PYTHON
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from grasp_target import select_target,refine_packet_contours
from simulated_reach import trajectory
from replay_settling import endpoint_settled
from mirror_grasp_plan import build_sequence
from adaptive_reach import TargetWatch,observe_regions,planning_result_current,pause_acknowledged
WINDOW='SIM TO REAL GRASP'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reach-only',action='store_true')
    parser.add_argument('--adaptive',action='store_true',help='Pause/replan a moving target during approach; stop if it moves during descent')
    args=parser.parse_args()
    preview=Preview();mapping=Mapping();cap=None;proc=None;log=None
    folder=ROOT/'captures/mirror_grasp'/str(time.time_ns());folder.mkdir(parents=True)
    messages=queue.Queue(maxsize=1);answers=queue.Queue();plans=None;preview_plans=[];planning=False
    running=False;index=0;elapsed=0.;preview_at=0.;prepared_at=0.;closed=False;used=False;settling_since=None
    batch_mode=(ROOT/'config/batch_objects.json').exists() and json.loads((ROOT/'config/batch_objects.json').read_text()).get('enabled',False) and not args.reach_only
    completed=[];active_target=None;batch_running=False;refresh_at=None;job_id=0;automatic_retries=0
    watch=None;adaptive_phase=None;pause_token=0;pause_at=0.;pending_auto=False
    notice='A holds this pose. P previews the first of two objects.' if batch_mode else 'A holds this pose. P previews the complete reach and grasp.'
    try:
        plane=Plane(preview.K,preview.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
        roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),preview.K,preview.dist,P=preview.K).reshape(-1,2).round().astype(np.int32)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
        detector=Changes(mask,stable_seconds=.12 if args.adaptive else .5);baseline=cv2.imread(str(ROOT/'captures/scene_changes/baseline.png'));detector.reset(baseline)
        profile=json.loads((ROOT/'config/grasp_target.json').read_text()) if (ROOT/'config/grasp_target.json').exists() else json.loads((ROOT/'config/measured_objects.json').read_text())['pink_highlighter']
        spec=preview.meta['camera'];cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Overhead camera unavailable')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,640);cap.set(4,480)
        proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'src/control_worker.py'),'--direct'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
        def reader():
            for line in proc.stdout:
                try:
                    value=json.loads(line)
                    if messages.full():messages.get_nowait()
                    messages.put_nowait(value)
                except (ValueError,queue.Empty,queue.Full):pass
        threading.Thread(target=reader,daemon=True).start()
        def send(op,**kw):
            proc.stdin.write(json.dumps({'op':op,'time':time.monotonic(),**kw})+'\n');proc.stdin.flush()
        state=messages.get(timeout=15);target=np.asarray(state['q']);last=time.monotonic();last_send=last_save=0.;previous=None
        log=(folder/'tracking.jsonl').open('a',buffering=1)
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1440,360)
        print('A hold | P preview full grasp | G execute complete preview | Space stop | X release',flush=True)
        def launch_planning(scene_image,cs,rs,st,selected_index,done,automatic,job):
            captured=json.loads(json.dumps(st));scene_image=scene_image.copy()
            cs=[np.asarray(c).copy() for c in cs];rs=[tuple(np.asarray(v).copy() for v in r) for r in rs]
            def calculate():
                try:
                    job_folder=folder/f'planning_{job}';job_folder.mkdir(exist_ok=True)
                    cv2.imwrite(str(job_folder/'planning_scene.jpg'),scene_image)
                    (job_folder/'captured_state.json').write_text(json.dumps({'state':captured,'completed':done},indent=2))
                    import importlib,simulated_reach,grasp_motion,grasp_planning,grasp_tail,grasp_return,basket_place,reach_retarget,mirror_grasp_plan,basket_tracking,batch_grasp
                    for module in (simulated_reach,grasp_motion,grasp_planning,grasp_tail,grasp_return,basket_place,reach_retarget,mirror_grasp_plan,basket_tracking,batch_grasp):importlib.reload(module)
                    if batch_mode:
                        answer=batch_grasp.prepare_batch(preview.model,mapping,plane,captured,scene_image,cs,rs,preview.T,list(done),preview_remaining=automatic!='adaptive')
                        basket=answer['basket'];region=answer['region']
                    else:
                        basket=basket_tracking.measure_basket(scene_image,json.loads((ROOT/'config/basket_reference.json').read_text()),plane,preview.T)
                        region=[v.tolist() for v in rs[selected_index]]
                        answer={'plans':mirror_grasp_plan.build_sequence(preview.model,mapping,plane,captured,rs[selected_index],reach_only=args.reach_only,basket_config=basket),'target':None,'basket':basket}
                    if args.adaptive:
                        observations=observe_regions(scene_image,cs,rs)
                        answer['observed_target']=min(observations,key=lambda o:np.linalg.norm(o.center-np.asarray(region[0])))
                    job_folder=folder/f'planning_{job}';job_folder.mkdir(exist_ok=True)
                    for out in (folder,job_folder):
                        cv2.imwrite(str(out/'planning_scene.jpg'),scene_image)
                        (out/'planning_input.json').write_text(json.dumps({'state':captured,'region':region,'completed':done},indent=2))
                        (out/'planning_basket.json').write_text(json.dumps(basket,indent=2))
                    answers.put((job,answer,automatic))
                except Exception as exc:
                    (job_folder/'failure.txt').write_text(str(exc))
                    answers.put((job,exc,automatic))
            threading.Thread(target=calculate,daemon=True).start()
        def adaptive_pause(reason,now):
            nonlocal running,plans,preview_plans,adaptive_phase,pause_token,pause_at,job_id,pending_auto,notice
            pause_token+=1;pause_at=now;adaptive_phase='pause';job_id+=1
            send('direct_pause',pause_id=pause_token)
            running=False;plans=None;preview_plans=[];pending_auto=False
            notice='Adaptive pause: '+reason+'; waiting for a clear stationary target.'
        while True:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Overhead camera lost')
            now=time.monotonic();dt=now-last;last=now
            try:state=messages.get_nowait()
            except queue.Empty:pass
            if proc.poll() is not None or now-state['time']>1:raise RuntimeError('Robot feedback lost')
            status=(state['state'],state['reason'])
            if status!=previous:
                target=np.array(state['command']);previous=status
                if state['state']!='ACTIVE':
                    running=False;closed=False;batch_running=False;refresh_at=None;job_id+=1
                    adaptive_phase=None;pending_auto=False
                    if state['state']=='HOLD':notice='STOPPED: '+state['reason']
            real=cv2.remap(frame,*preview.maps,cv2.INTER_LINEAR)
            excluded=cv2.dilate(getattr(preview,'robot_mask',np.zeros_like(mask)),np.ones((25,25),np.uint8))
            contours,_=detector.update(real,dt,excluded);contours=refine_packet_contours(real,refine_current_contours(real,contours),profile)
            regions=[plane.footprint(c) for c in contours];display,selected,_=select_target(real,contours,regions,profile)
            observations=observe_regions(real,contours,regions) if args.adaptive else []
            if watch is not None and args.adaptive:watch.update(observations,now)
            try:
                # Only one planning thread is allowed in flight, including after cancellation.
                answer_job,answer,automatic=answers.get_nowait();planning=False
                if answer_job!=job_id:notice='Cancelled planning result discarded';answer=None
                if answer is None:pass
                elif isinstance(answer,Exception):
                    detection_failure=any(message in str(answer) for message in ('not clearly visible','location ambiguous','Need one clearly separated','extends out of view'))
                    if automatic=='adaptive':
                        adaptive_phase=None;batch_running=False
                        send('hold',reason='Adaptive replan refused: '+str(answer))
                        notice='Adaptive replan stopped: '+str(answer)
                    elif automatic and batch_running and detection_failure and automatic_retries<3:
                        automatic_retries+=1;refresh_at=now+1.
                        notice=f'Refreshing plate/object view again ({automatic_retries}/3); arm holds still.'
                    else:
                        notice=('Second pickup stopped: ' if automatic else 'Plan refused: ')+str(answer);batch_running=False
                elif state['state']!='ACTIVE':notice='Plan discarded: enable holding and press P again'
                elif automatic=='adaptive' and not planning_result_current(state,answer['plans'][0]['plan']['start'],watch,answer['observed_target'],now):
                    adaptive_phase='pause';notice='Target changed during planning; waiting to refresh again.'
                else:
                    plans=answer['plans'];preview_plans=answer.get('preview_plans',plans);active_target=answer['target'];preview_at=now;prepared_at=time.time();used=False;closed=False
                    if args.adaptive:
                        if automatic=='adaptive':watch.commit(answer['observed_target'])
                        else:
                            watch=TargetWatch(answer['observed_target'],now);watch.update(observations,now)
                        adaptive_phase=None
                    path=ROOT/'config/basket.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(answer['basket'],indent=2));tmp.replace(path)
                    (folder/'plan.json').write_text(json.dumps(plans,indent=2))
                    (folder/'preview_plan.json').write_text(json.dumps(preview_plans,indent=2))
                    roll=next((p.get('retarget',{}).get('roll_change_deg') for p in plans if p['stage']=='approach'),None)
                    notice=f"PREVIEW: current-target approach only ({sum(p['plan'].get('playback_duration',p['plan']['duration']) for p in plans):.0f}s). G executes, then holds." if args.reach_only else f"PREVIEW: open, reach, grasp, basket, sit ({sum(p['plan'].get('playback_duration',p['plan']['duration']) for p in plans):.0f}s). G executes."
                    if roll is not None:notice+=f' Wrist roll {roll:+.0f} deg.'
                    if batch_mode:notice='PREVIEW: packet, then box, then sitting. G runs both; object 2 is re-detected after placement 1.' if not completed else 'Second object freshly planned from the camera.'
                    if automatic:
                        if (automatic is True and not batch_running) or max(abs(np.asarray(plans[0]['plan']['start'])-state['command']))>1e-6:
                            plans=None;batch_running=False;notice='Automatic continuation cancelled: arm command changed'
                        elif args.adaptive:
                            pending_auto=True;notice='Fresh plan ready; verifying target stability before resuming.'
                        else:
                            index=0;elapsed=0.;settling_since=None;used=True;running=True
                            send('direct_segment',plan=plans[0]['plan'],segment_id=0)
                            notice=f'Object {len(completed)+1}/2: {active_target}; Space stops the entire sequence.'
            except queue.Empty:pass
            if pending_auto and plans is not None and args.adaptive:
                if state['state']!='ACTIVE':pending_auto=False
                elif watch.moved():
                    adaptive_pause('target moved before continuation',now)
                elif now-preview_at>10:
                    pending_auto=False;plans=None;batch_running=False
                    send('hold',reason='Adaptive continuation needs a clear stationary target')
                elif watch.can_descend(now):
                    pending_auto=False
                    if not planning_result_current(state,plans[0]['plan']['start'],watch,watch.reference,now):
                        plans=None;batch_running=False
                        send('hold',reason='Arm command changed before adaptive continuation')
                    else:
                        used=True;running=True;index=0;elapsed=0.;settling_since=None
                        send('direct_segment',plan=plans[0]['plan'],segment_id=0)
                        notice='Adaptive sequence resumed from the held pose.'
            if running and args.adaptive and watch is not None:
                action=watch.action(plans[index]['stage'],now)
                if action=='pause':adaptive_pause(watch.problem or 'object moved',now)
                elif action=='stop':
                    send('hold',reason='Target moved or became unobservable during descent')
                    running=False;batch_running=False;refresh_at=None;plans=None
                    notice='Stopped during descent: inspect target, then A and P for a new attempt.'
            if running and state['state']=='ACTIVE':
                if dt>.5:
                    send('hold',reason='Replay camera/control cycle delayed');running=False;batch_running=False;refresh_at=None;notice='Stopped: control cycle delayed'
                else:
                    elapsed=float(state.get('segment_elapsed') or 0.) if state.get('segment_id')==index else 0.;plan=plans[index]['plan'];target=np.asarray(state['command'])
                    if elapsed>=plan['duration']:
                        if settling_since is None:settling_since=now
                        error=np.abs(np.asarray(state['q'])-np.asarray(plan['goal']))
                        if endpoint_settled(plans[index]['stage'],error):
                            if plans[index]['stage']=='approach':
                                reference=json.loads(json.dumps(plans[index]));reference['surface_center_base_m']=reference['retarget']['current_surface_center_base_m']
                                reference['joint_mapping']=mapping.mapping;reference['source_session']=str(folder);reference['completion_evidence']='Worker feedback reached approach endpoint within replay settling tolerances'
                                path=ROOT/'calibration/reach_reference.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(reference,indent=2));tmp.replace(path)
                            if index==len(plans)-1:
                                running=False;closed=plans[index]['stage']=='close'
                                notice='Approach finished; holding near the object.' if args.reach_only else 'Object released; returned to sitting pose. X releases torque.'
                                if batch_mode:
                                    completed.append(active_target)
                                    if len(completed)<2:
                                        automatic_retries=0;refresh_at=now+1.;notice='First release finished; refreshing scene for object 2.'
                                    else:
                                        batch_running=False;notice='Both placement sequences finished; sitting. X releases torque.'
                            else:
                                if args.adaptive and plans[index+1]['stage']=='lower' and not watch.can_descend(now):
                                    adaptive_pause('final descent needs a fresh stable observation',now)
                                else:
                                    index+=1;elapsed=0.;settling_since=None;send('direct_segment',plan=plans[index]['plan'],segment_id=index)
                        elif now-settling_since>5:
                            send('hold',reason='Replay endpoint did not settle');running=False;batch_running=False;refresh_at=None;notice='Stopped: endpoint did not settle'
            if adaptive_phase=='planning' and now-pause_at>30:
                adaptive_phase=None;batch_running=False;job_id+=1
                send('hold',reason='Adaptive planning timed out')
                notice='Adaptive planning timed out; late results will be discarded.'
            if adaptive_phase=='pause' and state['state']=='ACTIVE':
                if now-pause_at>30 or (now-pause_at>2 and not pause_acknowledged(state,pause_token)):
                    adaptive_phase=None;batch_running=False;plans=None
                    send('hold',reason='Adaptive pause timed out waiting for acknowledgement or stable view')
                    notice='Adaptive wait stopped; clear the view, then A and P.'
                elif (pause_acknowledged(state,pause_token) and not planning and watch.stable(now)
                      and np.max(abs(np.asarray(state['q'])[:5]-np.asarray(state['command'])[:5]))<=np.deg2rad(3)):
                    # Choose the tracked identity, not whichever contour happens to be first.
                    chosen=min(range(len(observations)),key=lambda j:np.linalg.norm(observations[j].center-watch.current.center))
                    planning=True;adaptive_phase='planning';job_id+=1
                    notice='Adaptive replan: current object, plate and held arm pose.'
                    launch_planning(real,contours,regions,state,chosen,list(completed),'adaptive',job_id)
            if refresh_at is not None and now>=refresh_at and batch_running and state['state']=='ACTIVE' and not planning and adaptive_phase is None:
                refresh_at=None;planning=True;plans=None;job_id+=1
                notice='Planning second object from the current camera view...'
                launch_planning(real,contours,regions,state,selected,list(completed),True,job_id)
            if now-last_send>=1/15:
                send('heartbeat')
                last_send=now
            virtual=plans is not None and not used and state['state']=='ACTIVE'
            q=np.asarray(state['q'])
            if virtual:
                count=len(preview_plans)
                t=min(3*count-.001,now-preview_at);i=min(count-1,int(t/3));p=preview_plans[i]['plan']
                q=trajectory({**p,'start':np.array(p['start']),'goal':np.array(p['goal']),'duration':3.},t-i*3)
            elif running:q=target
            from basket_place import basket_visuals
            basket_path=ROOT/'config/basket.json'
            basket_regions=basket_visuals(json.loads(basket_path.read_text())) if basket_path.exists() else []
            result=preview.draw(frame,{},regions=list(display)+basket_regions,show_marker=False,q_override=q)
            if virtual or running:preview.draw(frame,{},show_marker=False,q_override=np.asarray(state['q']))
            cv2.rectangle(result,(0,0),(1920,65),(20,20,20),-1)
            cv2.putText(result,f"{state['state']} | A hold | P preview | G EXECUTE PREVIEW | Space stop | X release",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
            cv2.putText(result,('PREVIEW: '+preview_plans[i].get('preview_target','object')+' / '+preview_plans[i]['stage']) if virtual else 'SIMULATION COMMAND' if running else state['reason'],(8,49),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.rectangle(result,(0,420),(1920,480),(20,20,20),-1)
            cv2.putText(result,notice,(8,445),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.putText(result,'ADAPTIVE: approach pauses/replans for motion; descent needs visibility. Space cancels.' if args.adaptive else 'G runs both objects; scene refreshes between pickups. Space cancels both.' if batch_mode else 'Keep the selected object stationary. Operator watches clearance; Space stops and holds.',(8,468),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.imshow(WINDOW,result);key=cv2.waitKey(1)&255
            if key in (ord('a'),ord('A')) and state['state'] in ('MONITOR','HOLD'):
                adaptive_phase=None;pending_auto=False;watch=None
                send('arm');plans=None;running=False;closed=False;batch_running=False;refresh_at=None;job_id+=1;notice='Holding requested; P previews reach and grasp'
            if key in (ord('p'),ord('P')):
                if state['state']!='ACTIVE' or running or planning or adaptive_phase is not None or pending_auto:notice='Press A to hold first; wait for current motion/planning'
                elif selected is None and not batch_mode:notice='Place the selected object clearly in view for target selection'
                else:
                    plans=None;planning=True;batch_running=False;refresh_at=None;completed=[];job_id+=1
                    notice='Planning from current object and plate positions; arm stays held'
                    launch_planning(real,contours,regions,state,selected,[],False,job_id)
            if key in (ord('g'),ord('G')):
                if state['state']!='ACTIVE' or plans is None or used or planning:notice='Press P to prepare a new preview first'
                elif now-preview_at<3*len(preview_plans):notice='Wait for the full sequence preview'
                elif time.time()-prepared_at>120:notice='Preview expired; press P again'
                elif max(abs(np.asarray(plans[0]['plan']['start'])-state['command']))>1e-6:notice='Command changed; press P again'
                elif args.adaptive and not watch.can_descend(now):notice='Target moved or is not stable/visible; wait or press P for a fresh preview'
                else:
                    send('direct_segment',plan=plans[0]['plan'],segment_id=0)
                    batch_running=batch_mode;running=True;used=True;index=0;elapsed=0.;settling_since=None;notice='Executing current-target approach only; will hold before descent.' if args.reach_only else 'Executing grasp, basket placement and sitting return. Space stops.'
            if key==32:
                send('hold');running=False;closed=False;plans=None;batch_running=False;refresh_at=None;job_id+=1;notice='Stop requested; entire sequence cancelled'
                adaptive_phase=None;pending_auto=False
            if key in (ord('x'),ord('X')):
                send('release');running=False;closed=False;plans=None;batch_running=False;refresh_at=None;job_id+=1;notice='Torque release requested'
                adaptive_phase=None;pending_auto=False
            if now-last_save>.25:
                sample={**state,'recorded_at_unix':time.time(),'notice':notice,'replay_running':running,'replay_stage':plans[index]['stage'] if plans and running else None,'replay_elapsed':elapsed,'batch_completed':completed,'batch_running':batch_running,'batch_target':active_target}
                sample.update(adaptive_enabled=args.adaptive,adaptive_phase=adaptive_phase,adaptive_problem=watch.problem if watch else None)
                log.write(json.dumps(sample)+'\n');path=ROOT/'calibration/mirror_grasp_live.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(sample,indent=2));tmp.replace(path)
                cv2.imwrite(str(folder/'latest.jpg'),result);last_save=now
            if key==27:break
            try:
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
            except cv2.error:break
    finally:
        if cap is not None:cap.release()
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
