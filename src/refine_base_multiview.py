"""Offline diagnostic fit to manually picked visible joint-axis landmarks.
The shaft-depth coordinates are nuisance parameters shared across views.
No joint calibration or working scene is modified.
"""
import json
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.optimize import least_squares
from mirror_robot import convert
from simulate import JOINTS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'calibration/table_registration'

def main():
    initial=json.loads((OUT/'base_registration_candidate.json').read_text())
    intr=json.loads((ROOT/'calibration/cameras/overhead_640x480.json').read_text())
    plane=json.loads((OUT/'board_plane.json').read_text())
    K=np.array(intr['matrix']);dist=np.array(intr['distortion'])
    ct=np.array(plane['T_camera_from_board'])@np.diag([1.,-1.,-1.,1.])
    T0=np.array(initial['T_table_from_base']);p0=np.r_[T0[:2,3],np.arctan2(T0[1,0],T0[0,0]),[0.,0.,0.]]
    # Picks are approximate near-side shaft centres, partly obscured by cables.
    images=['registration_raw','pose2_raw','pose3_raw']
    observations=['joint_observation','pose2_joints','pose3_joints']
    uv=np.array([[[490,341],[466,216],[461,122]],[[486,356],[433,236],[343,149]],[[479,383],[409,294],[255,207]]],float)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/so101/so101.xml'));d=mujoco.MjData(m)
    anchors=[];axes=[]
    ids=[m.joint(n).id for n in ['shoulder_lift','elbow_flex','wrist_flex']]
    for name in observations:
        obs=json.loads((ROOT/f'captures/base/{name}.json').read_text())
        q=convert(obs['positions'],json.loads((ROOT/'config/joint_mapping.json').read_text()))
        for n,v in zip(JOINTS,q):d.qpos[m.joint(n).qposadr[0]]=v
        mujoco.mj_forward(m,d);anchors.append(d.xanchor[ids].copy());axes.append(d.xaxis[ids].copy())
    anchors=np.array(anchors);axes=np.array(axes)
    def transform(p):
        c,s=np.cos(p[2]),np.sin(p[2]);T=np.array([[c,-s,0,p[0]],[s,c,0,p[1]],[0,0,1,.0024],[0,0,0,1.]])
        return ct@T
    def project(p,idx):
        T=transform(p);xyz=anchors[idx]+axes[idx]*p[3:,None]
        return cv2.projectPoints(xyz,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,dist)[0].reshape(-1,2)
    plate=np.array(initial['model_points_base_m']);plate_uv=np.array(initial['raw_pixels'])
    def residual(p,indices):
        T=transform(p);pred=cv2.projectPoints(plate,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,dist)[0].reshape(-1,2)
        return np.r_[np.concatenate([(project(p,i)-uv[i]).ravel()/5 for i in indices]),(pred-plate_uv).ravel()/3]
    low=p0+[-.03,-.03,-.35,-.06,-.06,-.06];high=p0+[.03,.03,.35,.06,.06,.06]
    def fit(indices):return least_squares(lambda p:residual(p,indices),p0,bounds=(low,high),loss='soft_l1')
    f=fit(range(3));cv=[]
    for i in range(3):
        h=fit([j for j in range(3) if j!=i]);cv.append(float(np.sqrt(np.mean(np.sum((project(h.x,i)-uv[i])**2,axis=1)))))
    result={'status':'diagnostic_only_manual_landmarks_not_verified','model':'Shared planar base pose and 3 shaft-depth nuisance coordinates; all joint offsets fixed','images':images,'landmark_names':['shoulder_lift_near_side_axis','elbow_near_side_axis','wrist_flex_near_side_axis'],'manual_pixels':uv.tolist(),'fitted_parameters_xy_yaw_axisdepth':f.x.tolist(),'T_camera_from_base':transform(f.x).tolist(),'per_pose_rms_px':[float(np.sqrt(np.mean(np.sum((project(f.x,i)-uv[i])**2,axis=1)))) for i in range(3)],'leave_one_pose_out_rms_px':cv,'limitations':['Manual picks are approximate and obscured by cables.','Fitted shaft depth must be verified against CAD before accepting correspondence.','No changes applied to robot, joint mapping or working scene.']}
    (OUT/'multiview_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    for i,name in enumerate(images):
        frame=cv2.imread(str(ROOT/f'captures/base/{name}.png'))
        for j,(a,b) in enumerate(zip(uv[i],project(f.x,i))):
            aa=tuple(np.rint(a).astype(int));bb=tuple(np.rint(b).astype(int));cv2.circle(frame,aa,5,(0,255,0),1);cv2.drawMarker(frame,bb,(0,0,255),cv2.MARKER_CROSS,9,1);cv2.putText(frame,str(j),aa,cv2.FONT_HERSHEY_SIMPLEX,.4,(0,150,255),1)
        cv2.imwrite(str(OUT/f'multiview_landmarks_pose{i+1}.jpg'),frame)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
