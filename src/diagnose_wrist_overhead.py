"""Fit one wrist optical mounting across paired views, without changing commands."""
import argparse,json
from pathlib import Path
import cv2,mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from camera_calibration import detect,object_points
from estimate_wrist_extrinsics import board_pose
from mirror_robot import convert
from simulate import JOINTS
ROOT=Path(__file__).resolve().parents[1]

def transform(p):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix();T[:3,3]=p[3:6];return T

def solve(source, output=None):
    source=source.resolve()
    reg=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text());meta=json.loads((Path(reg['source_session'])/'session.json').read_text());X=np.array(reg['T_camera_from_base'])
    iw=json.loads((ROOT/'calibration/cameras/wrist_640x480.json').read_text());K=np.array(iw['matrix']);dist=np.array(iw['distortion']);obj=object_points(.0096).astype(np.float64)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/so101/so101.xml'));d=mujoco.MjData(m)
    folders=[];gripper_boards=[];observed=[];individual=[];inv_grippers=[];board_poses=[]
    groups=[]; representatives=[]
    for folder in sorted(source.glob('pose_*')):
        if not (folder/'wrist_mount_candidate.json').exists():continue
        positions=json.loads((folder/'joints.json').read_text())['positions']
        raw=np.array([positions[n] for n in JOINTS[:5]])
        match=next((i for i,(pose,_) in enumerate(representatives) if np.max(np.abs(raw-pose))<1.0),None)
        if match is None:
            representatives.append((raw,folder));groups.append([folder.name])
        else:groups[match].append(folder.name)
    for _,folder in representatives:
        if not (folder/'wrist_mount_candidate.json').exists():continue
        r=json.loads((folder/'wrist_mount_candidate.json').read_text());individual.append(np.array(r['T_gripper_from_wrist_optical']))
        over=cv2.imread(str(folder/'overhead.png'));wrist=cv2.imread(str(folder/'wrist.png'));A,_=board_pose(over,meta['intrinsics']);_,info=board_pose(wrist,iw);ok,uv=detect(wrist)
        if not ok:raise ValueError('Board not found')
        uv=uv.reshape(-1,2)
        if info['reversed_detector_order']:uv=uv[::-1].copy()
        q=convert(json.loads((folder/'joints.json').read_text())['positions'],meta['joint_mapping'])
        for n,v in zip(JOINTS,q):d.qpos[m.joint(n).qposadr[0]]=v
        mujoco.mj_forward(m,d);b=m.body('gripper').id;G=np.eye(4);G[:3,:3]=d.xmat[b].reshape(3,3);G[:3,3]=d.xpos[b]
        inv_grippers.append(np.linalg.inv(G));board_poses.append(A);gripper_boards.append(np.linalg.inv(G)@np.linalg.inv(X)@A);observed.append(uv);folders.append(folder)
    individual=np.array(individual);N=len(folders)
    if N<5:raise ValueError('At least five paired poses required')
    def initial(ids):
        return np.r_[Rotation.from_matrix(individual[ids,:3,:3]).mean().as_rotvec(),np.median(individual[ids,:3,3],axis=0),Rotation.from_matrix(X[:3,:3]).as_rotvec(),X[:3,3]]
    def project(p,i):
        T=np.linalg.inv(transform(p[:6]))@inv_grippers[i]@np.linalg.inv(transform(p[6:12]))@board_poses[i]
        return cv2.projectPoints(obj,Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3],K,dist)[0].reshape(-1,2)
    def fit(ids):return least_squares(lambda p:np.concatenate([(project(p,i)-observed[i]).ravel() for i in ids]),initial(ids),loss='soft_l1',f_scale=3,max_nfev=500)
    def error(p,i):return float(np.sqrt(np.mean(np.sum((project(p,i)-observed[i])**2,axis=1))))
    allfit=fit(list(range(N)));errors=[error(allfit.x,i) for i in range(N)];held=[]
    for i in range(N):held.append(error(fit([j for j in range(N) if j!=i]).x,i))
    Y=transform(allfit.x[:6]);rot=Y[:3,:3]@np.diag([1.,-1.,-1.]);q=Rotation.from_matrix(rot).as_quat()
    result={'status':'diagnostic_free_overhead_requires_review','T_overhead_from_base_refined':transform(allfit.x[6:12]).tolist(),'overhead_translation_change_mm':float(np.linalg.norm(transform(allfit.x[6:12])[:3,3]-X[:3,3])*1000),'overhead_rotation_change_deg':float(np.degrees(Rotation.from_matrix(X[:3,:3].T@transform(allfit.x[6:12])[:3,:3]).magnitude())),'pose_groups':groups,'saved_view_count':sum(map(len,groups)),'validation':'One representative per joint-pose group; entire repeated-pose group excluded from training when that representative is held out.','T_gripper_from_wrist_optical':Y.tolist(),'optical_axes':'OpenCV right/down/forward','mujoco_position_gripper_m':Y[:3,3].tolist(),'mujoco_quat_wxyz':np.r_[q[3],q[:3]].tolist(),'camera_intrinsics':iw,'view_count':N,'per_view_rms_px':errors,'leave_one_pose_out_rms_px':held,'individual_translation_span_mm':(np.ptp(individual[:,:3,3],axis=0)*1000).tolist(),'source_folders':[str(f.relative_to(ROOT)) for f in folders],'limitations':['Wrist intrinsics and overhead/base calibration remain provisional.','Individual estimates vary; not validated for precise reaching or collision avoidance.','Reprojection checks involve two cameras and robot kinematics, not just mount accuracy.','No joint offsets or robot commands changed.']}
    out=output or source/'joint_overhead_wrist_candidate.json';out.write_text(json.dumps(result,indent=2)+'\n')
    for i,folder in enumerate(folders):
        im=cv2.imread(str(folder/'wrist.png'));pred=project(allfit.x,i)
        for a,b in zip(observed[i],pred):cv2.circle(im,tuple(np.rint(a).astype(int)),2,(0,255,0),1);cv2.drawMarker(im,tuple(np.rint(b).astype(int)),(0,0,255),cv2.MARKER_CROSS,5,1)
        cv2.putText(im,f'Common mount fit: {errors[i]:.1f}px',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2);cv2.imwrite(str(folder/'joint_overhead_wrist_check.jpg'),im)
    print(json.dumps({k:v for k,v in result.items() if k not in ['camera_intrinsics','source_folders']},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args();solve(args.source,args.output)
