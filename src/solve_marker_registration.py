"""Fit camera-from-base and gripper-from-marker transforms from stationary captures.
Optimizes raw corner reprojection with fixed intrinsics and fixed joint calibration.
No hardware access or changes to working scene configuration.
"""
import argparse
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mirror_robot import convert
from simulate import JOINTS
ROOT=Path(__file__).resolve().parents[1]

def transform(p):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix();T[:3,3]=p[3:6];return T

def parameters(T):return np.r_[Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3]]

def project(T,points,K,dist):
    return cv2.projectPoints(points,Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3],K,dist)[0].reshape(-1,2)

def fit_transforms(G,uv,points,K,dist,initial,indices,fk_builder=None):
    def residual(p):
        X,Y=transform(p[:6]),transform(p[6:12]);out=[]
        current_G=G if fk_builder is None else fk_builder(p[12:])
        for i in indices:
            T=X@current_G[i]@Y
            out.extend((project(T,points,K,dist)-uv[i]).ravel())
            depths=(points@T[:3,:3].T+T[:3,3])[:,2]
            out.extend(np.minimum(depths-.03,0)*1000)
        return np.array(out)
    bounds=(-np.inf,np.inf) if fk_builder is None else (np.r_[np.full(12,-np.inf),np.full(3,-np.deg2rad(10))],np.r_[np.full(12,np.inf),np.full(3,np.deg2rad(10))])
    return least_squares(residual,initial,bounds=bounds,loss='soft_l1',f_scale=2,max_nfev=600,xtol=1e-10,ftol=1e-10,gtol=1e-10)

def solve(folder,refine_joints=False):
    meta=json.loads((folder/'session.json').read_text())
    if len(meta['views'])<10:raise ValueError('Need at least 10 varied poses')
    if not meta['marker']['printed_scale_verified']:raise ValueError('Marker scale unverified')
    K=np.array(meta['intrinsics']['matrix']);dist=np.array(meta['intrinsics']['distortion']);s=meta['marker']['actual_black_square_m']/2
    points=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],float)
    m=mujoco.MjModel.from_xml_path(str(ROOT/meta['robot_model']));d=mujoco.MjData(m)
    G=[];uv=[];poses=[];joints=[]
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,meta['marker']['dictionary'])),params)
    for view in meta['views']:
        im=cv2.imread(str(folder/view['file']))
        if im is None or list(im.shape[1::-1])!=meta['size']:raise ValueError('Wrong image size')
        corners,ids,_=detector.detectMarkers(im)
        if ids is None or sum(ids.ravel()==meta['marker']['id'])!=1:raise ValueError('Marker detection failed')
        px=corners[int(np.where(ids.ravel()==meta['marker']['id'])[0][0])].reshape(4,2);uv.append(px)
        q=convert(view['positions'],meta['joint_mapping']);joints.append(q)
        for name,value in zip(JOINTS,q):d.qpos[m.joint(name).qposadr[0]]=value
        mujoco.mj_forward(m,d);b=m.body('gripper').id;T=np.eye(4);T[:3,:3]=d.xmat[b].reshape(3,3);T[:3,3]=d.xpos[b];G.append(T)
        ok,r,t=cv2.solvePnP(points,px,K,dist,flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:raise ValueError('Marker pose failed')
        poses.append(transform(np.r_[r.ravel(),t.ravel()]))
    G=np.array(G);uv=np.array(uv);joints=np.array(joints)
    def fk_builder(offsets):
        transforms=[]
        for q0 in joints:
            q=q0.copy();q[1:4]+=offsets
            for name,value in zip(JOINTS,q):d.qpos[m.joint(name).qposadr[0]]=value
            mujoco.mj_forward(m,d);b=m.body('gripper').id;T=np.eye(4);T[:3,:3]=d.xmat[b].reshape(3,3);T[:3,3]=d.xpos[b];transforms.append(T)
        return np.array(transforms)
    builder=fk_builder if refine_joints else None
    extra=np.zeros(3) if refine_joints else np.array([])
    X0=np.array(json.loads((ROOT/'calibration/table_registration/base_registration_candidate.json').read_text())['T_camera_from_base'])
    Ys=[np.linalg.inv(g)@np.linalg.inv(X0)@p for g,p in zip(G,poses)]
    Y0=np.eye(4);Y0[:3,:3]=Rotation.from_matrix(np.array([t[:3,:3] for t in Ys])).mean().as_matrix();Y0[:3,3]=np.median([t[:3,3] for t in Ys],axis=0)
    initial=np.r_[parameters(X0),parameters(Y0),extra]
    held=list(range(0,len(G),4));train=[i for i in range(len(G)) if i not in held]
    # Held-out views do not contribute to initialization or fitting of this validation fit.
    train_y=[Ys[i] for i in train];Ytrain=Y0.copy();Ytrain[:3,:3]=Rotation.from_matrix(np.array([t[:3,:3] for t in train_y])).mean().as_matrix();Ytrain[:3,3]=np.median([t[:3,3] for t in train_y],axis=0)
    validation=fit_transforms(G,uv,points,K,dist,np.r_[parameters(X0),parameters(Ytrain),extra],train,builder)
    def errors(p,indices):
        current_G=G if builder is None else builder(p[12:])
        return [float(np.sqrt(np.mean(np.sum((project(transform(p[:6])@current_G[i]@transform(p[6:12]),points,K,dist)-uv[i])**2,axis=1)))) for i in indices]
    held_errors=errors(validation.x,held)
    fit=fit_transforms(G,uv,points,K,dist,initial,range(len(G)),builder)
    err=errors(fit.x,range(len(G)))
    if builder is not None:G=builder(fit.x[12:])
    X,Y=transform(fit.x[:6]),transform(fit.x[6:])
    all_depths=np.array([(points@(X@g@Y)[:3,:3].T+(X@g@Y)[:3,3])[:,2] for g in G])
    result={'status':'candidate_requires_review','source_session':str(folder.resolve()),'method':'Joint corner reprojection: T_camera_from_base @ T_base_from_gripper(q) @ T_gripper_from_marker. Fixed recorded joint offsets and camera intrinsics.','view_count':len(G),'T_camera_from_base':X.tolist(),'T_gripper_from_marker':Y.tolist(),'overall_corner_rms_px':float(np.sqrt(np.mean(np.array(err)**2))),'per_view_rms_px':err,'held_out_indices':held,'held_out_rms_px':held_errors,'solver_success':bool(fit.success),'minimum_corner_depth_m':float(all_depths.min()),'joint_span_deg':np.rad2deg(np.ptp(joints,axis=0)).tolist(),'jacobian_singular_values_mixed_units':np.linalg.svd(fit.jac,compute_uv=False).tolist(),'limitations':['Assumes rigid marker attachment, correct robot kinematics and stationary samples.','Pixel fit does not independently establish metric accuracy or collision clearance.','No transforms, joint offsets or commands applied to working system.']}
    result['joint_offset_correction_deg']=np.rad2deg(fit.x[12:]).tolist()
    result['refines_three_bending_joints']=refine_joints
    if refine_joints:result['method']='Joint corner reprojection with camera/base transform, marker/gripper transform, and three bending-joint offsets bounded to +/-10 degrees; diagnostic only.'
    filename='registration_joint_refinement_candidate.json' if refine_joints else 'registration_candidate.json'
    (folder/filename).write_text(json.dumps(result,indent=2)+'\n')
    for i in range(len(G)):
        im=cv2.imread(str(folder/meta['views'][i]['file']));pred=project(X@G[i]@Y,points,K,dist)
        cv2.polylines(im,[np.rint(uv[i]).astype(np.int32)],True,(0,255,0),1);cv2.polylines(im,[np.rint(pred).astype(np.int32)],True,(0,0,255),1)
        cv2.putText(im,f'{i}: {err[i]:.2f}px',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        cv2.imwrite(str(folder/f'{"refined" if refine_joints else "fit"}_{i:03d}.jpg'),im)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path);p.add_argument('--refine-joints',action='store_true');a=p.parse_args();solve(a.session,a.refine_joints)
