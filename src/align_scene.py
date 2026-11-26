"""Reproducible approximate visual alignment; NOT calibrated metric registration."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1]

def main():
    cfg=json.loads((ROOT/'config/scene.json').read_text())
    # Manually selected corners; left end is partly hidden in the phone image.
    phone=np.float32([[437,796],[848,751],[857,836],[451,892]])
    world=np.float32([[.20,.025],[.42,.025],[.42,-.025],[.20,-.025]])
    H=cv2.getPerspectiveTransform(phone,world)
    def xy(p):return cv2.perspectiveTransform(np.float32([[p]]),H)[0,0].astype(float)
    cfg['remote_position_m']=[.31,0]
    cfg['wire_holder_position_m']=[*xy([666,547]),0]
    cfg['stand_position_m']=list(xy([180,367]))
    cfg['tape_measure_position_m']=[*xy([655,957]),.023]
    cfg['candle_position_m']=[*xy([776,990]),.013]
    # Put the robot near the middle of the indented edge, preserving the traced outline.
    if 'pre_alignment_table_outline_m' not in cfg:
        cfg['pre_alignment_table_outline_m']=cfg['table_outline_m']
    cfg['table_outline_m']=[[x,y+.26] for x,y in cfg['pre_alignment_table_outline_m']]
    # Remote corner correspondences in the fixed overhead frame, ordered like world corners.
    image=np.float64([[433,269],[409,106],[443,108],[469,271]])
    obj=np.column_stack([world,np.full(4,.018)])
    # Additional elevated objects constrain the approximate view orientation and FOV.
    obj=np.vstack([obj,[*cfg['wire_holder_position_m'][:2],.055],
                   [*cfg['stand_position_m'],.024],[*cfg['tape_measure_position_m'][:2],.046],
                   [*cfg['candle_position_m'][:2],.026]])
    image=np.vstack([image,[333,147],[233,370],[493,191],[497,152]])
    def matrix(f):return np.array([[f,0,320],[0,f,240],[0,0,1.]])
    ok,r,t=cv2.solvePnP(obj,image,matrix(400),np.zeros(5),flags=cv2.SOLVEPNP_SQPNP)
    if not ok:raise RuntimeError('Initial camera fit failed')
    def residual(p):
        pred,_=cv2.projectPoints(obj,p[:3],p[3:6],matrix(p[6]),np.zeros(5))
        return (pred[:,0]-image).ravel()
    fit=least_squares(residual,np.r_[r.ravel(),t.ravel(),400],bounds=(np.r_[np.full(6,-np.inf),180],np.r_[np.full(6,np.inf),1000]),loss='soft_l1',f_scale=5)
    R,_=cv2.Rodrigues(fit.x[:3]);C=-R.T@fit.x[3:6]
    if C[2]<=0 or np.any((obj@R.T+fit.x[3:6])[:,2]<=0):raise RuntimeError('Invalid camera fit')
    cfg['overhead_position_m']=C.tolist()
    cfg['overhead_xyaxes']=np.r_[R[0],-R[1]].tolist() # OpenCV right/down/forward -> MuJoCo right/up/back
    cfg['overhead_fovy_deg']=float(np.degrees(2*np.arctan(240/fit.x[6])))
    cfg['overhead_lookat_m']=(C+R[2]).tolist()
    pred,_=cv2.projectPoints(obj,fit.x[:3],fit.x[3:6],matrix(fit.x[6]),np.zeros(5))
    errors=np.linalg.norm(pred[:,0]-image,axis=1)
    cfg['alignment']={'status':'approximate visual fit; not metric calibration','phone_source':'Photos-1-001/IMG_8798.HEIC',
      'phone_size_px':[1200,1600],'phone_remote_corners_px':phone.tolist(),'overhead_reference':'captures/alignment/overhead_reference.jpg',
      'overhead_landmarks_px':image.tolist(),'world_landmarks_m':obj.tolist(),'phone_to_world_homography':H.tolist(),
      'reprojection_error_px':errors.tolist(),'rms_px':float(np.sqrt(np.mean(errors**2))),
      'limitations':['Remote partly occluded; corners picked manually','Remote surface used as approximate table plane','No lens distortion calibration','Robot base registration and table translation estimated','No encoder-derived arm pose yet']}
    cfg['notes']='Latest arrangement visually fitted using IMG_8798 and overhead landmarks. Remote is 22x5cm. Pose and scale remain approximate; see alignment metadata. Not suitable for physical collision clearance decisions.'
    (ROOT/'config/scene_alignment_proposal.json').write_text(json.dumps(cfg,indent=2)+'\n')
    frame=cv2.imread(str(ROOT/'captures/alignment/overhead_reference.jpg'))
    for i,(actual,predicted) in enumerate(zip(image,pred[:,0])):
        a=tuple(np.rint(actual).astype(int));b=tuple(np.rint(predicted).astype(int))
        cv2.circle(frame,a,5,(0,255,0),1);cv2.drawMarker(frame,b,(0,0,255),cv2.MARKER_CROSS,9,1)
        cv2.line(frame,a,b,(0,180,255),1);cv2.putText(frame,str(i),a,cv2.FONT_HERSHEY_SIMPLEX,.4,(255,0,0),1)
    cv2.imwrite(str(ROOT/'captures/alignment/landmark_fit.jpg'),frame)
    print(json.dumps({'camera_position':C.tolist(),'fovy':cfg['overhead_fovy_deg'],'rms_px':cfg['alignment']['rms_px'],'objects':{k:v for k,v in cfg.items() if k.endswith('position_m')}},indent=2))
if __name__=='__main__':main()
