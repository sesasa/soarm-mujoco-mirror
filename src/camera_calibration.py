"""Collect raw checkerboard views and estimate per-camera intrinsics. No robot I/O."""
import argparse
import json
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PATTERN=(9,6)

def detect(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    return cv2.findChessboardCornersSB(gray,PATTERN,flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

def object_points(square):
    p=np.zeros((54,3),np.float32);p[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)*square
    return p

def fit_views(corners,size,square):
    objects=[object_points(square) for _ in corners]
    rms,K,dist,rvecs,tvecs=cv2.calibrateCamera(objects,corners,size,None,None)
    errors=[]
    for obj,uv,r,t in zip(objects,corners,rvecs,tvecs):
        projected,_=cv2.projectPoints(obj,r,t,K,dist)
        errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-uv.reshape(-1,2))**2,axis=1)))))
    if not np.isfinite(K).all() or not np.isfinite(dist).all() or min(K[0,0],K[1,1])<=0:
        raise ValueError('Non-physical calibration estimate')
    return rms,K,dist,errors

def capture(args):
    cfg=json.loads((ROOT/'config/cameras.json').read_text())
    spec=next(c for c in cfg['cameras'] if c['role']==args.camera)
    boards=json.loads((ROOT/'config/calibration_boards.json').read_text())
    board=args.board or ('large' if args.camera=='overhead' else 'small')
    nominal_square=boards['boards'][board]['square_m']
    measured_bar=getattr(args,'measured_bar_mm',None)
    if measured_bar is not None and (not np.isfinite(measured_bar) or measured_bar<=0):
        raise ValueError('Measured bar length must be positive and finite')
    print_scale=measured_bar/100 if measured_bar is not None else 1.0
    square=nominal_square*print_scale
    folder=(args.resume.resolve() if args.resume else ROOT/'calibration/camera_views'/f'{args.camera}_{time.time_ns()}')
    meta={'camera':spec,'board':board,'inner_corners':list(PATTERN),'square_m':square,'size':[cfg['width'],cfg['height']],
          'raw_sensor_orientation':True,'printed_scale_verified':args.scale_verified or measured_bar is not None,
          'nominal_square_m':nominal_square,'print_scale':print_scale,'measured_check_bar_mm':measured_bar,
          'scale_assumption':'Both orthogonal bars on this sheet have the stated length; uniform scaling','views':[]}
    if args.resume:
        meta=json.loads((folder/'session.json').read_text())
        if meta['camera']['path']!=spec['path'] or meta['camera']['role']!=args.camera:
            raise ValueError('Resume session belongs to a different camera')
        if meta['size']!=[cfg['width'],cfg['height']] or meta['camera']['fourcc']!=spec['fourcc']:
            raise ValueError('Resume session uses a different capture mode')
        if meta['inner_corners']!=list(PATTERN) or not meta.get('raw_sensor_orientation'):
            raise ValueError('Resume session uses a different pattern or image orientation')
        if args.board and args.board!=meta['board']:raise ValueError('Different board requested')
        if measured_bar is not None and not np.isclose(square,meta['square_m']):
            raise ValueError('Printed scale differs from saved session')
        square=meta['square_m'];board=meta['board']
        for view in meta['views']:
            if not (folder/view['file']).is_file():raise ValueError('Saved view file missing')
        print(f"Resuming {len(meta['views'])} saved views at {square*1000:g} mm per square",flush=True)
    else:
        folder.mkdir(parents=True)
    # Unverified prints may be previewed; saving remains blocked until confirmation.
    cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
    try:
        if not cap.isOpened():raise RuntimeError('Camera unavailable; close other camera previews')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg['width']);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg['height'])
        for key,value in spec.get('controls',{}).items():
            subprocess.run(['v4l2-ctl','-d',spec['path'],'--set-ctrl',f'{key}={value}'],check=True)
        code=int(cap.get(cv2.CAP_PROP_FOURCC));actual=''.join(chr((code>>(8*i))&255) for i in range(4))
        if actual!=spec['fourcc']:raise RuntimeError('Unexpected capture format')
        print(f'Session: {folder}\nSPACE saves a detected view; Escape finishes. Aim for 20–30 varied views.',flush=True)
        notice=('Lift one edge to tilt in 3D; show the full board' if meta['printed_scale_verified'] else 'Measure both 10 cm bars; press V only if both match')
        while True:
            start=time.monotonic();ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera read failed')
            if list(frame.shape[1::-1])!=meta['size']:raise RuntimeError('Unexpected resolution')
            found,corners=detect(frame);display=frame.copy()
            if found:cv2.drawChessboardCorners(display,PATTERN,corners,found)
            cv2.rectangle(display,(0,0),(frame.shape[1],62),(20,20,20),-1)
            lines=[f"{args.camera.upper()} | {square*1000:g}mm squares | saved {len(meta['views'])}/20+ | SPACE: save ESC: finish",notice]
            for i,line in enumerate(lines):cv2.putText(display,line,(8,20+25*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
            cv2.imshow('Camera calibration',display)
            key=cv2.waitKey(1)&255
            if key==27:break
            if key in (ord('v'),ord('V')) and not meta['printed_scale_verified']:
                meta['printed_scale_verified']=True
                (folder/'session.json').write_text(json.dumps(meta,indent=2)+'\n')
                notice='Print scale confirmed. SPACE saves a detected board'
            if key==32:
                if not meta['printed_scale_verified']:
                    notice='First measure both 10 cm bars, then V to confirm'
                elif not found:notice='No complete 9 x 6 corner grid; show the whole board'
                elif any(float(np.mean(np.linalg.norm(corners.reshape(-1,2)-np.array(v['corners']).reshape(-1,2),axis=1)))<8 for v in meta['views']):
                    notice='Too similar to an earlier view; tilt or reposition the board'
                else:
                    filename=f"view_{len(meta['views']):03d}.png"
                    if not cv2.imwrite(str(folder/filename),frame):raise RuntimeError('Failed to save image')
                    meta['views'].append({'file':filename,'corners':corners.tolist()})
                    (folder/'session.json').write_text(json.dumps(meta,indent=2)+'\n')
                    notice='Saved. Move to a different location / tilt / distance'
            time.sleep(max(0,1/cfg['loop_hz']-(time.monotonic()-start)))
    finally:
        cap.release();cv2.destroyAllWindows()
        (folder/'session.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f'Saved {len(meta["views"])} views. Solve with:\n.venv/bin/python src/camera_calibration.py solve {folder}')

def solve(args):
    folder=args.session.resolve();meta=json.loads((folder/'session.json').read_text())
    if not meta.get('printed_scale_verified'):raise ValueError('Printed scale was not verified')
    if len(meta['views'])<15:raise ValueError('Need at least 15 distinct views; prefer 20–30')
    all_corners=[]
    for view in meta['views']:
        frame=cv2.imread(str(folder/view['file']))
        if frame is None or list(frame.shape[1::-1])!=meta['size']:raise ValueError('Missing or wrong-size image')
        found,corners=detect(frame)
        if not found:raise ValueError(f"Board no longer detectable in {view['file']}")
        all_corners.append(corners)
    # First check independent held-out views; their extrinsics are fitted, intrinsics aren't.
    train=[c for i,c in enumerate(all_corners) if i%5!=0]
    held=[c for i,c in enumerate(all_corners) if i%5==0]
    _,K0,d0,_=fit_views(train,tuple(meta['size']),meta['square_m'])
    held_errors=[];obj=object_points(meta['square_m'])
    for uv in held:
        ok,r,t=cv2.solvePnP(obj,uv,K0,d0)
        if not ok:raise ValueError('Held-out pose estimation failed')
        projection,_=cv2.projectPoints(obj,r,t,K0,d0)
        held_errors.append(float(np.sqrt(np.mean(np.sum((projection.reshape(-1,2)-uv.reshape(-1,2))**2,axis=1)))))
    rms,K,dist,errors=fit_views(all_corners,tuple(meta['size']),meta['square_m'])
    result={'status':'candidate_requires_review','camera':meta['camera'],'size':meta['size'],'board':meta['board'],
            'square_m':meta['square_m'],'matrix':K.tolist(),'distortion':dist.ravel().tolist(),
            'rms_px':float(rms),'per_view_rms_px':errors,'held_out_rms_px':held_errors,
            'source_session':str(folder),'model':'OpenCV pinhole with 5 distortion coefficients','raw_sensor_orientation':True,
            'notes':'Intrinsics only. Does not establish robot/world extrinsics or wrist hand-eye calibration; not auto-applied to simulation.'}
    out=folder/'intrinsics_candidate.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Candidate: {out}\nTraining RMS: {rms:.3f} px; held-out errors: {held_errors}')
    image=cv2.imread(str(folder/meta['views'][0]['file']))
    cv2.imwrite(str(folder/'undistorted_preview.png'),cv2.undistort(image,K,dist,None,K))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('capture');c.add_argument('--camera',choices=['overhead','wrist'],required=True)
    c.add_argument('--board',choices=['large','small']);c.add_argument('--scale-verified',action='store_true')
    c.add_argument('--resume',type=Path,help='Continue an existing session, retaining its board scale and views')
    c.add_argument('--measured-bar-mm',type=float,help='Actual length of BOTH printed 100 mm bars on this sheet; applies uniform scale correction')
    s=sub.add_parser('solve');s.add_argument('session',type=Path)
    args=p.parse_args();capture(args) if args.action=='capture' else solve(args)
if __name__=='__main__':main()
