"""Estimate wrist optical mounting from a stationary dual-camera checkerboard pair."""
import argparse,json
from pathlib import Path
import cv2,mujoco,numpy as np
from camera_calibration import detect,object_points
from mirror_robot import convert
from simulate import JOINTS
ROOT=Path(__file__).resolve().parents[1]

def board_pose(frame,intrinsics):
    found,uv=detect(frame)
    if not found:raise ValueError('Complete board required')
    uv=uv.reshape(54,2)
    # Printed 10x7 board has a black outer corner at canonical top-left,
    # and a white outer corner at the opposite corner. Resolve 180-degree ordering.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    def intensity(p):
        x,y=np.rint(p).astype(int)
        if x<1 or y<1 or x>=gray.shape[1]-1 or y>=gray.shape[0]-1:raise ValueError('Pattern edge clipped')
        return float(np.median(gray[y-1:y+2,x-1:x+2]))
    dx=uv[1]-uv[0];dy=uv[9]-uv[0]
    a=intensity(uv[0]-.5*dx-.5*dy);b=intensity(uv[0]+.5*dx-.5*dy)
    if abs(a-b)<25:raise ValueError('Board colour orientation ambiguous')
    reversed_order=a>b
    if reversed_order:uv=uv[::-1].copy()
    K=np.array(intrinsics['matrix']);dist=np.array(intrinsics['distortion']);obj=object_points(.0096)
    ok,r,t=cv2.solvePnP(obj,uv,K,dist)
    if not ok:raise ValueError('Pose estimation failed')
    R=cv2.Rodrigues(r)[0]
    if np.any((obj@R.T+t.ravel())[:,2]<=0):raise ValueError('Board behind camera')
    pred=cv2.projectPoints(obj,r,t,K,dist)[0].reshape(-1,2)
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t.ravel()
    return T,{'rms_px':float(np.sqrt(np.mean(np.sum((pred-uv)**2,axis=1)))),'corner_colour_samples': [a,b],'reversed_detector_order':bool(reversed_order),'canonical_origin_pixel':uv[0].tolist()}

def main(folder):
    registration=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
    session=json.loads((Path(registration['source_session'])/'session.json').read_text())
    intr_o=session['intrinsics'];intr_w=json.loads((ROOT/'calibration/cameras/wrist_640x480.json').read_text())
    A,oa=board_pose(cv2.imread(str(folder/'overhead.png')),intr_o)
    B,wb=board_pose(cv2.imread(str(folder/'wrist.png')),intr_w)
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/so101/so101.xml'));d=mujoco.MjData(m)
    q=convert(json.loads((folder/'joints.json').read_text())['positions'],session['joint_mapping'])
    for n,v in zip(JOINTS,q):d.qpos[m.joint(n).qposadr[0]]=v
    mujoco.mj_forward(m,d);i=m.body('gripper').id;G=np.eye(4);G[:3,:3]=d.xmat[i].reshape(3,3);G[:3,3]=d.xpos[i]
    X=np.array(registration['T_camera_from_base']);Y=np.linalg.inv(G)@np.linalg.inv(X)@A@np.linalg.inv(B)
    result={'status':'single_pose_candidate_requires_multi_pose_validation','T_gripper_from_wrist_optical':Y.tolist(),'optical_axes':'OpenCV right/down/forward','overhead_board_fit':oa,'wrist_board_fit':wb,'square_m':.0096,'source':str(folder.resolve()),'limitations':['Both camera intrinsics and overhead/base transform contribute uncertainty.','Wrist intrinsic calibration is provisional.','Sequential camera/motor samples require stationary robot and board.','Board symmetry resolved by black/white outer-corner contrast.','No working camera transform or robot command mapping changed.']}
    (folder/'wrist_mount_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);args=p.parse_args();main(args.folder)
