"""Manual paired-camera captures, with read-only encoder checks before/after."""
import os
import argparse,json,subprocess,time
from pathlib import Path
import cv2,numpy as np
from camera_calibration import detect
from estimate_wrist_extrinsics import main as estimate
ROOT=Path(__file__).resolve().parents[1]

def observe():
    p=subprocess.run([os.environ.get('LEROBOT_PYTHON','python'),str(ROOT/'src/robot_setup.py'),'observe'],capture_output=True,text=True,timeout=10,check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])

def main(folder=None):
    cfg=json.loads((ROOT/'config/cameras.json').read_text());caps=[];folder=folder or ROOT/'calibration/wrist_extrinsics';folder.mkdir(parents=True,exist_ok=True)
    notice='Keep board still. SPACE saves both views; ESC finishes'
    def pair():
        for _,c in caps:
            if not c.grab():raise RuntimeError('Camera grab failed')
        frames={}
        for s,c in caps:
            ok,f=c.retrieve()
            if not ok or f.shape[:2]!=(480,640):raise RuntimeError('Camera frame failed')
            frames[s['role']]=f
        return frames
    try:
        for s in cfg['cameras']:
            c=cv2.VideoCapture(s['path'],cv2.CAP_V4L2);caps.append((s,c))
            if not c.isOpened():raise RuntimeError('Camera unavailable')
            c.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*s['fourcc']));c.set(3,640);c.set(4,480)
        print('Paired capture open. SPACE: save a still pose. ESC: finish. No motion commands.',flush=True)
        while True:
            frames=pair();found={};corners={};panels=[]
            for role in ['overhead','wrist']:
                found[role],corners[role]=detect(frames[role]);im=frames[role].copy()
                if found[role]:cv2.drawChessboardCorners(im,(9,6),corners[role],True)
                cv2.rectangle(im,(0,0),(640,58),(20,20,20),-1)
                for i,t in enumerate([f'{role.upper()} | '+('BOARD DETECTED' if found[role] else 'Show whole small board'),notice]):cv2.putText(im,t,(8,20+24*i),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                panels.append(im)
            cv2.imshow('Wrist optical calibration - read only',np.hstack(panels));key=cv2.waitKey(1)&255
            if key==27:break
            if key==32:
                if not all(found.values()):notice='Both cameras must detect the entire board';continue
                # Drain queued camera frames around the slow one-shot encoder reads.
                a=observe()
                for _ in range(5):first=pair()
                b=observe()
                for _ in range(5):last=pair()
                names=list(a['positions']);diff=np.array([b['positions'][n]-a['positions'][n] for n in names])
                stable=np.isfinite(diff).all() and np.max(np.abs(diff))<.6
                for role in first:
                    ok1,c1=detect(first[role]);ok2,c2=detect(last[role])
                    stable=stable and ok1 and ok2
                    if ok1 and ok2:stable=stable and min(np.max(np.linalg.norm(c1.reshape(-1,2)-c2.reshape(-1,2),axis=1)),np.max(np.linalg.norm(c1.reshape(-1,2)-c2.reshape(-1,2)[::-1],axis=1)))<1.5
                if not stable:notice='Movement detected: hold still and press SPACE again';continue
                i=0
                while (folder/f'pose_{i:03d}').exists():i+=1
                p=folder/f'pose_{i:03d}';p.mkdir()
                for role,f in first.items():
                    if not cv2.imwrite(str(p/(role+'.png')),f):raise RuntimeError('Failed to save frame')
                (p/'joints.json').write_text(json.dumps(a,indent=2)+'\n');(p/'capture_check.json').write_text(json.dumps({'before':a,'after':b,'stationarity_check_passed':True,'saved_time':time.time(),'synchronization':'Sequential read-only encoder queries and paired camera grabs; not hardware synchronized.'},indent=2)+'\n')
                try:estimate(p);notice=f'Saved pose {i}. Change arm position / wrist angle gently'
                except ValueError as e:notice='Saved for review: '+str(e)
    finally:
        for _,c in caps:c.release()
        cv2.destroyAllWindows()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Separate capture folder for a changed camera mounting')
    args=parser.parse_args();main(args.output)
