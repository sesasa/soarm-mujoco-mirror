"""Operator-triggered approach, lower, close and lift for the real SO101."""
import json,queue,signal,subprocess,threading,time
import cv2,numpy as np
from command_mapping import Mapping,ROOT
from mirror_robot import PYTHON
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from measured_objects import highlighter_proxy
from real_hover import selected_target_visible,scene_matches
from simulated_reach import trajectory
from grasp_motion import GraspMotion
from grasp_planning import make_grasp_report

WINDOW='REAL GRASP - each stage needs G'

def main():
    preview=Preview();mapping=Mapping();cap=wrist=proc=None;log=None
    folder=ROOT/'captures/real_grasp'/str(time.time_ns());folder.mkdir(parents=True)
    messages=queue.Queue(maxsize=1);plans=queue.Queue();planning=False;report=None
    context=None;opening=None;preview_since=0.;notice='A holds arm; hands clear, then P previews approach'
    previous_target=None;stable=0.;last_tick=time.monotonic();last_send=last_save=0.;stop_sent=False
    try:
        plane=Plane(preview.K,preview.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
        roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),preview.K,preview.dist,P=preview.K).reshape(-1,2).round().astype(np.int32)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
        detector=Changes(mask);baseline=cv2.imread(str(ROOT/'captures/scene_changes/baseline.png'))
        if baseline is None:raise RuntimeError('Table baseline unavailable')
        detector.reset(baseline);profile=json.loads((ROOT/'config/measured_objects.json').read_text())['pink_highlighter']
        spec=preview.meta['camera'];cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Overhead camera unavailable')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,640);cap.set(4,480)
        ws=next(c for c in json.loads((ROOT/'config/cameras.json').read_text())['cameras'] if c['role']=='wrist')
        wrist=cv2.VideoCapture(ws['path'],cv2.CAP_V4L2)
        if wrist.isOpened():
            wrist.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*ws['fourcc']));wrist.set(3,640);wrist.set(4,480)
        proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'src/control_worker.py'),'--allow-grasp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
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
        state=messages.get(timeout=15);log=(folder/'tracking.jsonl').open('a',buffering=1)
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1440,360)
        print('S seated start | A hold | P approach | D lower | C close | L lift | O open | G execute preview | Space hold | X release',flush=True)
        print('Each stage stops and holds. Support arm before X or closing.',flush=True)
        last_stage=None;executed_key=None;last_state_reason=None
        while True:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Overhead camera lost')
            wrist_frame=None
            if wrist.isOpened():
                wok,wrist_frame=wrist.read()
                if not wok:wrist_frame=None
            now=time.monotonic();dt=now-last_tick;last_tick=now
            try:state=messages.get_nowait()
            except queue.Empty:pass
            if proc.poll() is not None or now-state['time']>1:raise RuntimeError('Robot feedback lost')
            if now-last_send>.10:send('heartbeat');last_send=now
            real=cv2.remap(frame,*preview.maps,cv2.INTER_LINEAR)
            excluded=cv2.dilate(getattr(preview,'robot_mask',np.zeros_like(mask)),np.ones((25,25),np.uint8))
            contours,status=detector.update(real,dt,excluded);contours=refine_current_contours(real,contours)
            regions=[plane.footprint(c) for c in contours]
            display,index,_=highlighter_proxy(real,contours,regions,profile)
            anchor={'surface_center_base_m':context[0].tolist()} if context is not None else None
            visible=selected_target_visible(anchor,regions,contours,real) if anchor else index is not None
            target=regions[index] if context is None and index is not None else context
            stable=stable+min(dt,.2) if visible and target is not None and previous_target is not None and scene_matches([previous_target],[target]) else 0.
            previous_target=target if visible else None
            covered=False
            if context is not None:
                poly=plane.projected_footprint(context).round().astype(np.int32)
                target_mask=np.zeros_like(mask);cv2.fillPoly(target_mask,[poly],255)
                covered=np.count_nonzero((target_mask>0)&(excluded>0))>max(20,.2*np.count_nonzero(target_mask))
            if state['state']=='GRASP' and not stop_sent:
                stage=state['grasp_running'];reason=None
                if dt>.5:reason='Camera cycle delayed'
                elif stage in ('approach','lower') and not visible and not covered:reason='Target lost outside expected robot occlusion'
                if reason:
                    send('hold',reason=reason);stop_sent=True;notice='Stopped: '+reason
                    cv2.imwrite(str(folder/'stop_frame.png'),frame)
            state_reason=(state['state'],state['reason'])
            if state_reason!=last_state_reason and state['state']=='HOLD' and state['reason']=='Seated startup complete; P previews approach':
                notice=state['reason']
            elif state_reason!=last_state_reason and state['state']=='HOLD' and not state['reason'].startswith('Stage complete:'):
                notice='STOPPED: '+state['reason']+'; do not advance to the next stage'
            last_state_reason=state_reason
            if state.get('grasp_stage')!=last_stage:
                last_stage=state.get('grasp_stage')
                if last_stage:
                    notice={'approach':'Approach done. Check alignment, then D previews lowering.',
                            'lower':'Lowering done. Check jaws straddle object, then C previews closing.',
                            'close':'Closing done. Inspect actual grip; L previews 2 cm lift. C tightens 2 degrees.',
                            'lift':'Lift done. Holding object if grasp succeeded. No further automatic motion.',
                            'open':'Jaw opened; arm stays held.'}[last_stage]
            try:
                answer=plans.get_nowait();planning=False
                if isinstance(answer,Exception):notice='Stage refused: '+str(answer)
                else:
                    report=answer;preview_since=now;notice=f"PREVIEW {report['stage']}: {report['plan']['duration']:.0f}s motion. G executes after 5s preview."
                    (folder/f"plan_{report['stage']}_{time.time_ns()}.json").write_text(json.dumps(report,indent=2))
                    cv2.imwrite(str(folder/'before.jpg'),frame)
            except queue.Empty:pass
            q=np.asarray(state['q']);virtual=report is not None and state['state'] in ('ACTIVE','HOLD') and report['prepared_at_unix']!=executed_key
            if virtual:
                path={**report['plan'],'start':np.array(report['plan']['start']),'goal':np.array(report['plan']['goal']),'duration':5.}
                q=trajectory(path,now-preview_since)
            result=preview.draw(frame,{},regions=display,show_marker=False,q_override=q)
            if virtual:preview.draw(frame,{},show_marker=False,q_override=np.asarray(state['q']))
            if wrist_frame is not None:result[:,640:1280]=cv2.resize(wrist_frame,(640,480))
            cv2.rectangle(result,(0,0),(1920,65),(20,20,20),-1)
            cv2.putText(result,f"{state['state']} | S seated start | A hold | P approach | D lower | C close | L lift | O open | G MOVE | Space stop | X release",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
            cv2.putText(result,'OVERHEAD                    '+state['reason'],(8,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
            cv2.putText(result,'WRIST CAMERA',(650,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
            cv2.putText(result,'SIMULATION PREVIEW' if virtual else 'MEASURED ROBOT OVERLAY',(1290,47),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
            cv2.rectangle(result,(0,418),(1920,480),(20,20,20),-1)
            cv2.putText(result,notice,(8,442),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,255),1)
            cv2.putText(result,f"Target {'visible' if visible else 'robot occluded' if covered else 'uncertain'} | Last completed: {last_stage or 'none'} | Each stage requires your G key",(8,467),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,0),1)
            cv2.imshow(WINDOW,result);key=cv2.waitKey(1)&255
            if key in (ord('s'),ord('S')) and state['state']=='MONITOR':
                send('seated_start');report=None;context=None;opening=None;notice='Seated startup requested; keep hands clear'
            if key in (ord('a'),ord('A')) and state['state']=='MONITOR':
                send('arm');report=None;context=None;opening=None;notice='Holding requested; hands clear, then P'
            stage={ord('p'):'approach',ord('P'):'approach',ord('d'):'lower',ord('D'):'lower',ord('c'):'close',ord('C'):'close',ord('l'):'lift',ord('L'):'lift',ord('o'):'open',ord('O'):'open'}.get(key)
            if stage:
                if state['state'] not in ('ACTIVE','HOLD') or planning:notice='Enable holding with A and finish current stage first'
                elif stage=='approach' and (index is None or stable<.7):notice='Keep the highlighter visible and stationary before P'
                elif stage!='approach' and context is None:notice='Prepare approach with P first'
                elif stage=='lower' and state.get('grasp_stage')!='approach':notice='Approach did not complete; lowering is blocked'
                elif stage=='close' and state.get('grasp_stage') not in ('lower','close'):notice='Lowering must complete before closing'
                elif stage=='lift' and state.get('grasp_stage')!='close':notice='Close and inspect the grip before lifting'
                else:
                    if stage=='approach':context=tuple(np.asarray(v).copy() for v in regions[index]);opening=state['command'][5]
                    report=None;planning=True;notice='Preparing '+stage+'; real arm keeps holding'
                    captured=json.loads(json.dumps(state));region=tuple(v.copy() for v in context)
                    def calculate(st=captured,reg=region,action=stage,op=opening):
                        try:plans.put(make_grasp_report(preview.model,mapping,plane,st,reg,action,op))
                        except Exception as error:plans.put(error)
                    threading.Thread(target=calculate,daemon=True).start()
            if key in (ord('g'),ord('G')):
                try:
                    if report is None:raise ValueError('Prepare a stage first: P, D, C, L or O')
                    if now-preview_since<5:raise ValueError('Wait for the five-second preview')
                    if report['stage']=='approach' and (not visible or stable<.7):raise ValueError('Approach target is not stable and visible')
                    if report['stage']=='lower' and not visible and not covered:raise ValueError('Target is lost; inspect before lowering')
                    GraspMotion(report,mapping,state['command'],state['q'],state.get('grasp_stage'),time.time())
                    send('grasp',report=report);executed_key=report['prepared_at_unix'];stop_sent=False;notice='Requested '+report['stage']+'; Space stops'
                except ValueError as error:notice=str(error)
            if key==32:send('hold');report=None;stop_sent=True;notice='Stop requested; holding'
            if key in (ord('x'),ord('X')):send('release');report=None;context=None;stop_sent=True;notice='Torque released; support arm'
            if now-last_save>.25:
                sample={**state,'recorded_at_unix':time.time(),'notice':notice,'target_visible':visible,'target_occluded':bool(covered),'target_stable_s':stable}
                log.write(json.dumps(sample)+'\n');path=ROOT/'calibration/grasp_live.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(sample,indent=2));tmp.replace(path)
                cv2.imwrite(str(folder/'latest.jpg'),result);last_save=now
            if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
    finally:
        for camera in (cap,wrist):
            if camera is not None:camera.release()
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
