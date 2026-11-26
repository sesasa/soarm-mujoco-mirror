"""Record stationary overhead marker images with read-only joint feedback."""
import os
import json
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('LEROBOT_PYTHON','python')

def main():
    cfg=json.loads((ROOT/'config/cameras.json').read_text())
    spec=next(c for c in cfg['cameras'] if c['role']=='overhead')
    marker=json.loads((ROOT/'config/gripper_registration_marker.json').read_text())
    if not marker['printed_scale_verified']:raise ValueError('Marker scale unverified')
    folder=ROOT/'calibration/marker_views'/str(time.time_ns());folder.mkdir(parents=True)
    meta={'status':'capture_in_progress','marker':marker,'camera':spec,'size':[cfg['width'],cfg['height']],
          'intrinsics':json.loads((ROOT/'calibration/cameras/overhead_640x480.json').read_text()),
          'joint_mapping':json.loads((ROOT/'config/joint_mapping.json').read_text()),
          'robot_model':'assets/so101/so101.xml','views':[],
          'notes':'Read-only motor stream. Marker must stay rigid on fixed gripper housing. Software timestamps and stationary gating, not hardware synchronization.'}
    def save(): (folder/'session.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    proc=subprocess.Popen([PYTHON,str(ROOT/'src/robot_setup.py'),'stream'],stdout=subprocess.PIPE,text=True,bufsize=1)
    samples=queue.Queue(maxsize=100)
    def read():
        for line in proc.stdout:
            try:
                item=(time.monotonic(),json.loads(line))
                samples.put_nowait(item)
            except (ValueError,queue.Full):pass
    threading.Thread(target=read,daemon=True).start()
    cap=None
    try:
        cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Camera unavailable')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,cfg['width']);cap.set(4,cfg['height'])
        params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,marker['dictionary'])),params)
        hist=deque();frames=deque();notice='Hold still ~2 seconds; SPACE saves, ESC finishes';start=time.monotonic()
        names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        print(f'Session: {folder}\nRead-only capture. SPACE saves a stationary pose; ESC finishes.',flush=True)
        while True:
            now=time.monotonic()
            while True:
                try:hist.append(samples.get_nowait())
                except queue.Empty:break
            while hist and now-hist[0][0]>2:hist.popleft()
            if proc.poll() is not None:raise RuntimeError('Motor feedback process stopped')
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera read failed')
            if list(frame.shape[1::-1])!=meta['size']:raise RuntimeError('Wrong capture resolution')
            corners,ids,_=detector.detectMarkers(frame);uv=None
            if ids is not None:
                matches=np.where(ids.ravel()==marker['id'])[0]
                if len(matches)==1:uv=corners[int(matches[0])].reshape(4,2)
            if uv is None:frames.clear()
            else:frames.append((now,uv.copy()))
            while frames and now-frames[0][0]>2:frames.popleft()
            fresh=bool(hist and now-hist[-1][0]<.25)
            stable=False
            if fresh and len(hist)>=15 and hist[-1][0]-hist[0][0]>1.4 and len(frames)>=5 and now-frames[0][0]>1.4:
                positions=np.array([[h[1]['positions'][n] for n in names] for h in hist])
                pixels=np.array([f[1] for f in frames])
                stable=bool(np.isfinite(positions).all() and np.max(np.ptp(positions[:,:5],axis=0))<.8 and np.ptp(positions[:,5])<1 and np.max(np.linalg.norm(pixels-pixels.mean(0),axis=2))<1.5)
            display=frame.copy()
            if uv is not None:cv2.aruco.drawDetectedMarkers(display,[uv.reshape(1,4,2)],np.array([[marker['id']]]))
            cv2.rectangle(display,(0,0),(640,64),(25,25,25),-1)
            for i,line in enumerate([f"Saved {len(meta['views'])} | {'READY' if stable else 'HOLD STILL / SHOW MARKER'}",notice]):
                cv2.putText(display,line,(8,22+25*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            cv2.imshow('Robot-camera registration (read only)',display);key=cv2.waitKey(1)&255
            if key==27:break
            if key==32:
                if not stable:notice='Wait until READY; marker and arm must stay still'
                else:
                    record=hist[-1][1];values=np.array([record['positions'][n] for n in names])
                    if meta['views'] and abs(values[5]-meta['views'][0]['positions']['gripper'])>2:
                        notice='Gripper opening changed; restore it before saving';continue
                    if any(np.linalg.norm(values[:5]-np.array([v['positions'][n] for n in names[:5]]))<8 for v in meta['views']):
                        notice='Too similar: use a different arm position or wrist tilt';continue
                    filename=f"view_{len(meta['views']):03d}.png"
                    if not cv2.imwrite(str(folder/filename),frame):raise RuntimeError('Image save failed')
                    meta['views'].append({'file':filename,'corners_px':uv.tolist(),'positions':record['positions'],'motor_timestamp':record['time'],'saved_time':time.time(),'motor_receipt_age_s':now-hist[-1][0]})
                    save();notice='Saved. Move gently to a different pose, then hold still'
            if now-start>10 and not fresh:raise RuntimeError('Fresh joint feedback unavailable')
    finally:
        if cap is not None:cap.release()
        cv2.destroyAllWindows()
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
        if proc.stdout:proc.stdout.close()
        meta['status']='captured_requires_review';save()
        print(f'Saved {len(meta["views"])} views in {folder}',flush=True)
if __name__=='__main__':main()
