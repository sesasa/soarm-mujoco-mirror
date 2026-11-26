"""Supervised single physical high hover, with live camera and joint feedback."""
import json,queue,signal,subprocess,threading,time
from pathlib import Path
import cv2,numpy as np
from command_mapping import Mapping,ROOT
from mirror_robot import PYTHON
from preview_camera_alignment import Preview
from scene_changes import Plane,Changes,RAW_ROI,refine_current_contours
from measured_objects import highlighter_proxy
from simulated_reach import ReachSolver,hover_point,trajectory
from table_clearance import TableClearance
from hover_trajectory import HoverTrajectory

WINDOW='REAL ARM - supervised high hover'

def make_report(model,mapping,plane,state,region):
    start=np.asarray(state['command'],float);measured=np.asarray(state['q'],float)
    mapping.check(start);mapping.check(measured)
    solver=ReachSolver(model);solver.bounds=np.asarray(mapping.limits)
    up=plane.base_from_board[:3,2].copy()
    if up[2]<0:up=-up
    target=hover_point(region[0],up,.12)
    plan=solver.solve(start,target)
    if plan['start_adjustment_deg']>1e-5:raise ValueError('Planner tried to alter the starting command')
    plan['goal'][5]=start[5]
    plan['duration']=max(5.,1.875*np.max(np.abs(plan['goal']-start))/np.deg2rad(3))
    checker=TableClearance(model,plane.base_from_board[:3,3],up)
    clearance=checker.path(plan,samples=161)
    initial=checker.at(measured)
    if initial['minimum_m']<.005 or initial['gripper_holder_minimum_m']<.08:
        raise ValueError('Measured starting pose has insufficient modeled table clearance')
    result={'mode':'position_only_high_hover','height_above_table_m':.12,
            'prepared_at_unix':time.time(),'joint_mapping':mapping.mapping,
            'plan':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()},
            'clearance':clearance,'initial_clearance':initial,
            'surface_center_base_m':np.asarray(region[0]).tolist(),
            'hardware_commands':False,'grasp_enabled':False}
    HoverTrajectory(result,mapping,state['command'],state['q'],time.time())
    return result

def scene_matches(reference,current):
    """Allow a known footprint to split into contained image fragments."""
    groups=[[] for _ in reference]
    for center,size,rotation in current:
        corners=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*size
        world=corners@np.asarray(rotation).T+center
        owners=[]
        for i,(rc,rs,rr) in enumerate(reference):
            local=(world-rc)@np.asarray(rr)
            if np.all(np.abs(local[:,:2])<=np.asarray(rs)[:2]+.006):owners.append((i,local))
        if len(owners)!=1:return False
        i,local=owners[0];groups[i].append(local)
    for (center,size,rotation),parts in zip(reference,groups):
        if not parts:return False
        points=np.concatenate(parts)
        if np.max(np.abs(points[:,:2].min(axis=0)+np.asarray(size)[:2]))>.008:return False
        if np.max(np.abs(points[:,:2].max(axis=0)-np.asarray(size)[:2]))>.008:return False
        if len(parts)>1:
            # Reject overlapping duplicate detections and mostly missing objects.
            areas=[np.prod(x[:,:2].max(axis=0)-x[:,:2].min(axis=0)) for x in parts]
            original=4*np.prod(np.asarray(size)[:2])
            if not .6*original<=sum(areas)<=1.15*original:return False
    return True

def selected_target_visible(report,regions,contours,real):
    """Track the selected spatial target, not every red patch in the scene."""
    surface=np.asarray(report['surface_center_base_m'])
    candidates=[i for i,r in enumerate(regions) if np.linalg.norm(r[0]-surface)<.008]
    if len(candidates)!=1:return False
    i=candidates[0];mask=np.zeros(real.shape[:2],np.uint8)
    cv2.drawContours(mask,[np.asarray(contours[i],np.int32)],-1,255,-1)
    hsv=cv2.cvtColor(real,cv2.COLOR_BGR2HSV)
    pink=((hsv[:,:,0]<=6)|(hsv[:,:,0]>=170))&(hsv[:,:,1]>100)&(hsv[:,:,2]>100)
    count=np.count_nonzero(pink&(mask>0))
    return bool(count>=120 and count/max(np.count_nonzero(mask),1)>.18)

def main():
    preview=Preview();mapping=Mapping();cap=proc=None;log=None
    folder=ROOT/'captures/real_hover'/str(time.time_ns());folder.mkdir(parents=True)
    messages=queue.Queue(maxsize=1);plans=queue.Queue();planning=False
    report=None;reference=None;preview_since=0.;notice='Support arm into allowed range; A holds at measured pose'
    last_message=0.;last_save=0.;stable=0.;previous_regions=[];last_tick=time.monotonic();stop_sent=False
    tracked_surface=None;previous_target=None
    try:
        plane=Plane(preview.K,preview.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
        roi=cv2.undistortPoints(RAW_ROI.reshape(-1,1,2),preview.K,preview.dist,P=preview.K).reshape(-1,2).round().astype(np.int32)
        mask=np.zeros((480,640),np.uint8);cv2.fillPoly(mask,[roi],255)
        detector=Changes(mask);baseline=cv2.imread(str(ROOT/'captures/scene_changes/baseline.png'))
        if baseline is None:raise RuntimeError('Saved table baseline unavailable')
        detector.reset(baseline)
        profile=json.loads((ROOT/'config/measured_objects.json').read_text())['pink_highlighter']
        spec=preview.meta['camera'];cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Overhead camera unavailable')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,640);cap.set(4,480)
        proc=subprocess.Popen([PYTHON,'-u',str(ROOT/'src/control_worker.py'),'--allow-hover'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
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
        state=messages.get(timeout=15)
        log=(folder/'tracking.jsonl').open('a',buffering=1)
        cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1440,360)
        print('A: hold current pose | P: preview high hover | G: run once | Space: stop/hold | X: torque off',flush=True)
        print('Support the arm before X or closing the window.',flush=True)
        while True:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera frame lost')
            now=time.monotonic();dt=now-last_tick;last_tick=now
            try:state=messages.get_nowait()
            except queue.Empty:pass
            if proc.poll() is not None or now-state['time']>1:raise RuntimeError('Robot feedback lost')
            if now-last_message>.10:send('heartbeat');last_message=now
            real=cv2.remap(frame,*preview.maps,cv2.INTER_LINEAR)
            excluded=cv2.dilate(getattr(preview,'robot_mask',np.zeros_like(mask)),np.ones((25,25),np.uint8))
            contours,status=detector.update(real,dt,excluded)
            contours=refine_current_contours(real,contours)
            regions=[plane.footprint(c) for c in contours]
            display,index,measured=highlighter_proxy(real,contours,regions,profile)
            # Keep the user's selected target even if another object's label is pink.
            anchor=report if report is not None else ({'surface_center_base_m':tracked_surface} if tracked_surface is not None else None)
            if anchor is not None and selected_target_visible(anchor,regions,contours,real):
                index=min(range(len(regions)),key=lambda i:np.linalg.norm(regions[i][0]-np.asarray(anchor['surface_center_base_m'])))
            elif report is not None:
                index=None
            valid=index is not None and not status.startswith(('Large change','Work area obscured'))
            current_target=regions[index] if valid else None
            consistent=valid and previous_target is not None and scene_matches([previous_target],[current_target])
            stable=stable+min(dt,.2) if consistent else 0.
            previous_target=current_target
            if valid and report is None:tracked_surface=regions[index][0].tolist()
            previous_regions=regions
            if state['state']=='HOVER' and not stop_sent:
                target_ok=report is not None and selected_target_visible(report,regions,contours,real)
                scene_ok=reference is not None and scene_matches(reference,regions)
                if not target_ok or not scene_ok or dt>.5:
                    reason='selected target lost' if not target_ok else 'scene footprints changed' if not scene_ok else 'camera cycle delayed'
                    send('hold',reason='Camera guard: '+reason);stop_sent=True;notice='Stopped: '+reason
                    cv2.imwrite(str(folder/'stop_frame.png'),frame)
                    (folder/'stop_diagnostic.json').write_text(json.dumps({'reason':reason,'cycle_s':dt,
                        'state':state,'target_visible':target_ok,'scene_matches':scene_ok,
                        'contours':[c.tolist() for c in contours],'regions':[[v.tolist() for v in r] for r in regions]},indent=2))
            try:
                outcome=plans.get_nowait();planning=False
                if isinstance(outcome,Exception):notice='Preview refused: '+str(outcome)
                else:
                    report,reference=outcome
                    preview_since=now;notice=f"PREVIEW ONLY: {report['plan']['duration']:.0f}s real path; G runs after preview"
                    (folder/'plan.json').write_text(json.dumps(report,indent=2))
                    cv2.imwrite(str(folder/'before.jpg'),frame)
            except queue.Empty:pass
            q=np.asarray(state['q']);target=None
            if report:
                target=np.asarray(report['plan']['target'])
                if state['state']=='ACTIVE':
                    plan={**report['plan'],'start':np.array(report['plan']['start']),'goal':np.array(report['plan']['goal']),'duration':5.}
                    q=trajectory(plan,now-preview_since)
            result=preview.draw(frame,{},regions=display,show_marker=False,target=target,q_override=q)
            # Detection exclusion always follows actual feedback, not previewed motion.
            if report and state['state']=='ACTIVE':
                preview.draw(frame,{},show_marker=False,q_override=np.asarray(state['q']))
            cv2.rectangle(result,(0,0),(1920,64),(20,20,20),-1)
            cv2.putText(result,f"{state['state']} | A: hold | P: preview | G: MOVE REAL ARM | Space: stop/hold | X: release",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
            cv2.putText(result,state['reason'],(8,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.rectangle(result,(0,418),(1920,480),(20,20,20),-1)
            cv2.putText(result,notice,(8,442),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.putText(result,f"12 cm reference hover; jaws unchanged | target {'visible' if valid else 'uncertain'} | stable {stable:.1f}s | no grasp",(8,467),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imshow(WINDOW,result);key=cv2.waitKey(1)&255
            if key in (ord('a'),ord('A')) and state['state'] in ('MONITOR','HOLD'):
                report=None;send('arm');notice='Holding; move hands clear, then P previews'
            if key in (ord('p'),ord('P')):
                if state['state']!='ACTIVE' or planning or stable<.7 or state.get('hover_used'):
                    notice='Preview needs ACTIVE holding, stable visible target, and unused hover'
                else:
                    report=None;planning=True;notice='Planning; real arm stays holding'
                    captured_state=json.loads(json.dumps(state));captured_regions=[(c.copy(),s.copy(),r.copy()) for c,s,r in regions];captured_region=captured_regions[index]
                    def calculate(st=captured_state,rs=captured_regions,reg=captured_region):
                        try:plans.put((make_report(preview.model,mapping,plane,st,reg),rs))
                        except Exception as error:plans.put(error)
                    threading.Thread(target=calculate,daemon=True).start()
            if key in (ord('g'),ord('G')):
                try:
                    if report is None:raise ValueError('No prepared path: press P to preview')
                    if state['state']!='ACTIVE':raise ValueError('Arm is not holding in ACTIVE mode: press A first')
                    if now-preview_since<5:raise ValueError('Preview still playing; wait five seconds before G')
                    if not valid:raise ValueError('Selected highlighter is not visible')
                    if stable<.7:raise ValueError(f'Highlighter settling: {stable:.1f}/0.7 seconds; try G when stable')
                    if not scene_matches(reference,regions):raise ValueError('Scene changed since preview; press P to refresh')
                    HoverTrajectory(report,mapping,state['command'],state['q'],time.time())
                    send('hover',report=report);notice='Real hover requested; Space stops';stop_sent=False
                except ValueError as error:notice=str(error)
            if key==32:
                send('hold');report=None;notice='Stop requested; holding';stop_sent=True
            if key in (ord('x'),ord('X')):
                send('release');report=None;notice='Torque released; support arm';stop_sent=True
            if now-last_save>.2:
                sample={**state,'recorded_at_unix':time.time(),'notice':notice,'target_visible':valid,'target_stable_s':stable}
                log.write(json.dumps(sample)+'\n')
                path=ROOT/'calibration/hover_live.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(sample,indent=2));tmp.replace(path)
                cv2.imwrite(str(folder/'latest.jpg'),result);last_save=now
                if state['state']=='HOLD' and report:
                    (folder/'result.json').write_text(json.dumps({'state':state,'goal_error_deg':np.rad2deg(np.array(state['q'])-report['plan']['goal']).tolist(),'grasp_executed':False},indent=2))
            if key==27 or cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:break
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
