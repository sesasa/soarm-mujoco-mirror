"""Capture both cameras concurrently with verified formats; never opens the robot bus."""
import argparse
import json
import time
import subprocess
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--preview',action='store_true')
    p.add_argument('--alignment',action='store_true',help='Show labeled previews with center guides for camera positioning')
    p.add_argument('--auto-exposure',action='store_true',help='Use automatic exposure instead of saved manual exposure settings')
    args=p.parse_args()
    cfg=json.loads((ROOT/'config/cameras.json').read_text())
    out=ROOT/'captures';out.mkdir(exist_ok=True)
    cameras=[];report=[]
    try:
        for spec in cfg['cameras']:
            cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
            cameras.append((spec,cap))
            if not cap.isOpened():raise RuntimeError(f"Cannot open {spec['path']}")
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg['height'])
            controls=dict(spec.get('controls',{}))
            if args.auto_exposure:
                controls.pop('exposure_time_absolute',None)
                controls['auto_exposure']=3
            for key,value in controls.items():
                subprocess.run(['v4l2-ctl','-d',spec['path'],'--set-ctrl',f'{key}={value}'],check=True)
            code=int(cap.get(cv2.CAP_PROP_FOURCC))
            actual=''.join(chr((code>>(8*i))&255) for i in range(4))
            if actual!=spec['fourcc']:raise RuntimeError(f"{spec['name']}: expected {spec['fourcc']}, got {actual}")
            report.append({**spec,'applied_controls':controls,'negotiated_fourcc':actual,'capture_fps':cap.get(cv2.CAP_PROP_FPS),'frames':0})
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            start=time.monotonic()
            for index,(spec,cap) in enumerate(cameras):
                ok,frame=cap.read()
                if not ok:raise RuntimeError(f"Read failed: {spec['name']}")
                if frame.shape[:2]!=(cfg['height'],cfg['width']):raise RuntimeError('Unexpected image dimensions')
                report[index]['frames']+=1
                cv2.imwrite(str(out/f"{spec['name']}.jpg"),frame)
                rotation=spec.get('rotation_deg',0)
                transforms={0:None,90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}
                if rotation not in transforms:raise ValueError('rotation_deg must be 0, 90, 180 or 270')
                display=frame.copy() if rotation==0 else cv2.rotate(frame,transforms[rotation])
                cv2.imwrite(str(out/f"{spec['name']}_view.jpg"),display)
                if args.alignment:
                    h,w=display.shape[:2]
                    cv2.line(display,(w//2-20,h//2),(w//2+20,h//2),(0,220,255),1)
                    cv2.line(display,(w//2,h//2-20),(w//2,h//2+20),(0,220,255),1)
                    cv2.rectangle(display,(w//5,h//5),(4*w//5,4*h//5),(0,220,255),1)
                    label=f"{spec['role'].upper()} | {spec['name']} | rotate {rotation} deg"
                    cv2.rectangle(display,(0,0),(w,30),(20,20,20),-1)
                    cv2.putText(display,label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                if args.preview or args.alignment:cv2.imshow(spec['role'],display)
            if (args.preview or args.alignment) and cv2.waitKey(1)==27:break
            time.sleep(max(0,1/cfg['loop_hz']-(time.monotonic()-start)))
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        for _,cap in cameras:cap.release()
        if args.preview or args.alignment:cv2.destroyAllWindows()

if __name__=='__main__':main()
