"""Read-only MuJoCo/camera alignment preview. No robot commands or torque writes."""
import os
import argparse
import json
import queue
import signal
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from mirror_robot import convert
from simulate import JOINTS
ROOT=Path(__file__).resolve().parents[1]

class Preview:
    def __init__(self):
        self.registration=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
        session=Path(self.registration['source_session'])
        self.meta=json.loads((session/'session.json').read_text())
        self.K=np.array(self.meta['intrinsics']['matrix']);self.dist=np.array(self.meta['intrinsics']['distortion'])
        self.mapping=self.meta['joint_mapping'];self.size=tuple(self.meta['size'])
        self.T=np.array(self.registration['T_camera_from_base'])
        R=self.T[:3,:3];C=-R.T@self.T[:3,3]
        rotation=Rotation.from_matrix(R.T@np.diag([1.,-1.,-1.])).as_quat()
        quat=np.r_[rotation[3],rotation[:3]]
        root=ET.parse(ROOT/'assets/so101/so101.xml').getroot()
        root.find('compiler').set('meshdir',str(ROOT/'assets/so101/assets'))
        from wrist_mount_visual import apply_wrist_mount_visual
        apply_wrist_mount_visual(root)
        visual=root.find('visual');ET.SubElement(visual,'global',offwidth='800',offheight='640')
        for mat in root.findall('asset/material'):
            if 'sts3215' not in mat.get('name',''):mat.set('rgba','0.88 0.89 0.87 1')
        f=self.K[1,1];self.K0=np.array([[f,0,400],[0,f,320],[0,0,1.]])
        world=root.find('worldbody')
        ET.SubElement(world,'camera',name='calibrated_preview',pos=' '.join(map(str,C)),quat=' '.join(map(str,quat)),fovy=str(np.degrees(2*np.arctan(320/f))))
        ET.SubElement(world,'light',pos='0 0 1',dir='0 0 -1',directional='true',diffuse='0.8 0.8 0.8',ambient='0.4 0.4 0.4')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model);self.renderer=mujoco.Renderer(self.model,height=640,width=800)
        self.options=mujoco.MjvOption();self.options.geomgroup[:]=0;self.options.geomgroup[2]=1
        self.warp=self.K@np.linalg.inv(self.K0)
        w,h=self.size;self.maps=cv2.initUndistortRectifyMap(self.K,self.dist,None,self.K,(w,h),cv2.CV_32FC1)

    def draw(self,frame,positions,regions=(),show_marker=True,target=None,q_override=None):
        if tuple(frame.shape[1::-1])!=self.size:raise ValueError('Frame size differs from calibration')
        q=convert(positions,self.mapping) if q_override is None else np.asarray(q_override,float)
        for name,value in zip(JOINTS,q):self.data.qpos[self.model.joint(name).qposadr[0]]=value
        mujoco.mj_forward(self.model,self.data)
        self.renderer.update_scene(self.data,camera='calibrated_preview',scene_option=self.options)
        # Optional world-space change footprints, visual only: no model/collision edits.
        for center,size,rotation in regions:
            scene=self.renderer.scene
            if scene.ngeom>=scene.maxgeom:break
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_BOX,
                               np.asarray(size),np.asarray(center),np.asarray(rotation).ravel(),
                               np.array([1.,.45,.05,.7],dtype=np.float32))
            scene.geoms[scene.ngeom].segid=-1
            scene.geoms[scene.ngeom].objid=-1
            scene.geoms[scene.ngeom].objtype=mujoco.mjtObj.mjOBJ_UNKNOWN
            scene.ngeom+=1
        if target is not None:
            scene=self.renderer.scene
            if scene.ngeom<scene.maxgeom:
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.array([.006,.006,.006]),np.asarray(target),np.eye(3).ravel(),
                                   np.array([0.,1.,.2,1.],dtype=np.float32))
                scene.geoms[scene.ngeom].segid=-1
                scene.geoms[scene.ngeom].objid=-1
                scene.geoms[scene.ngeom].objtype=mujoco.mjtObj.mjOBJ_UNKNOWN
                scene.ngeom+=1
        render=cv2.cvtColor(self.renderer.render(),cv2.COLOR_RGB2BGR)
        self.renderer.enable_segmentation_rendering()
        try:
            seg=self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        mask=(seg[:,:,0]>=0).astype(np.uint8)*255
        sim=cv2.warpPerspective(render,self.warp,self.size)
        mask=cv2.warpPerspective(mask,self.warp,self.size,flags=cv2.INTER_NEAREST)
        self.robot_mask=mask.copy()
        real=cv2.remap(frame,*self.maps,cv2.INTER_LINEAR)
        overlay=real.copy();mixed=cv2.addWeighted(real,.5,sim,.5,0);overlay[mask>0]=mixed[mask>0]
        if show_marker:
            # Measured marker corners versus predicted marker in the undistorted view.
            detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,self.meta['marker']['dictionary'])))
            corners,ids,_=detector.detectMarkers(frame)
            if ids is not None:
                for uv,marker_id in zip(corners,ids.ravel()):
                    if marker_id==self.meta['marker']['id']:
                        measured=cv2.undistortPoints(uv,self.K,self.dist,P=self.K).reshape(4,2)
                        cv2.polylines(overlay,[np.rint(measured).astype(np.int32)],True,(0,255,0),2)
            b=self.model.body('gripper').id;G=np.eye(4);G[:3,:3]=self.data.xmat[b].reshape(3,3);G[:3,3]=self.data.xpos[b]
            T=self.T@G@np.array(self.registration['T_gripper_from_marker']);s=self.meta['marker']['actual_black_square_m']/2
            pts=np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]])
            predicted=cv2.projectPoints(pts,Rotation.from_matrix(T[:3,:3]).as_rotvec(),T[:3,3],self.K,np.zeros(5))[0]
            cv2.polylines(overlay,[np.rint(predicted).astype(np.int32)],True,(0,0,255),2)
        panels=[]
        for im,title in [(real,'REAL - lens corrected'),(sim,'MUJOCO - measured camera pose'),(overlay,'OVERLAY - green observed / red predicted' if show_marker else 'OVERLAY - robot alignment')]:
            panel=im.copy();cv2.rectangle(panel,(0,0),(self.size[0],28),(20,20,20),-1);cv2.putText(panel,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);panels.append(panel)
        result=np.hstack(panels)
        cv2.putText(result,'PROVISIONAL / READ ONLY | S: save | ESC: close',(8,result.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,255),1)
        return result

    def draw_side(self, plane, target, regions=()):
        """Side view of the current virtual pose and provisional table plane."""
        camera=mujoco.MjvCamera()
        camera.lookat[:]=[target[0]/2,target[1]/2,.12]
        camera.distance=.72
        camera.azimuth=np.degrees(np.arctan2(target[1],target[0]))+90
        camera.elevation=-14
        self.renderer.update_scene(self.data,camera=camera,scene_option=self.options)
        scene=self.renderer.scene
        basis=np.asarray(plane[:3,:3]).copy()
        if basis[2,2]<0:basis[:,1:]*=-1
        up=basis[:,2];point=plane[:3,3]
        center=np.array([.10,.10,0.]);center-=up*np.dot(center-point,up)
        def geom(kind,size,pos,rotation,color):
            if scene.ngeom>=scene.maxgeom:return
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,kind,np.asarray(size),np.asarray(pos),np.asarray(rotation).ravel(),np.asarray(color,np.float32))
            g.segid=-1;g.objid=-1;g.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN
            scene.ngeom+=1
        geom(mujoco.mjtGeom.mjGEOM_BOX,[.38,.38,.005],center-.005*up,basis,[.45,.43,.38,1])
        for position,size,rotation in regions:
            geom(mujoco.mjtGeom.mjGEOM_BOX,size,position,rotation,[1.,.45,.05,1.])
        geom(mujoco.mjtGeom.mjGEOM_SPHERE,[.006]*3,target,np.eye(3),[0.,1.,.2,1.])
        foot=target-up*np.dot(target-point,up)
        # Thin line marks the hover point's height above the table.
        if scene.ngeom<scene.maxgeom:
            g=scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([0.,1.,.2,1.],np.float32))
            mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,.001,foot,target)
            g.segid=-1;g.objid=-1;g.objtype=mujoco.mjtObj.mjOBJ_UNKNOWN
            scene.ngeom+=1
        render=cv2.cvtColor(self.renderer.render(),cv2.COLOR_RGB2BGR)
        side=cv2.resize(render,(640,512))[16:496].copy()
        cv2.rectangle(side,(0,0),(640,28),(20,20,20),-1)
        cv2.putText(side,f'SIDE VIEW | green line: {abs(np.dot(target-point,up))*100:.0f} cm above table',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,255,0),1)
        return side

    def close(self):self.renderer.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');parser.add_argument('--view',type=int,default=0);args=parser.parse_args()
    preview=Preview();proc=None;cap=None;out=ROOT/'captures/calibrated_preview';out.mkdir(exist_ok=True)
    try:
        if not args.live:
            view=preview.meta['views'][args.view];frame=cv2.imread(str(Path(preview.registration['source_session'])/view['file']))
            result=preview.draw(frame,view['positions']);path=out/f'view_{args.view:03d}.jpg';cv2.imwrite(str(path),result);print(path);return
        spec=preview.meta['camera'];cap=cv2.VideoCapture(spec['path'],cv2.CAP_V4L2)
        if not cap.isOpened():raise RuntimeError('Camera unavailable')
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*spec['fourcc']));cap.set(3,preview.size[0]);cap.set(4,preview.size[1])
        proc=subprocess.Popen([os.environ.get('LEROBOT_PYTHON','python'),str(ROOT/'src/robot_setup.py'),'stream'],stdout=subprocess.PIPE,text=True,bufsize=1)
        records=queue.Queue(maxsize=1)
        def reader():
            for line in proc.stdout:
                try:
                    item=(time.monotonic(),json.loads(line))
                    if records.full():records.get_nowait()
                    records.put_nowait(item)
                except (ValueError,queue.Empty,queue.Full):pass
        threading.Thread(target=reader,daemon=True).start()
        stamp,record=records.get(timeout=10)
        cv2.namedWindow('Calibrated MuJoCo preview - read only',cv2.WINDOW_NORMAL);cv2.resizeWindow('Calibrated MuJoCo preview - read only',1440,360)
        print('Preview open. No motion commands or torque writes. Stay still for accurate comparison.',flush=True)
        while True:
            start=time.monotonic();ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera read failed')
            try:stamp,record=records.get_nowait()
            except queue.Empty:pass
            if proc.poll() is not None or time.monotonic()-stamp>1:raise RuntimeError('Stale motor feedback; preview stopped')
            result=preview.draw(frame,record['positions']);cv2.imshow('Calibrated MuJoCo preview - read only',result)
            key=cv2.waitKey(1)&255
            if key==27 or cv2.getWindowProperty('Calibrated MuJoCo preview - read only',cv2.WND_PROP_VISIBLE)<1:break
            if key in [ord('s'),ord('S')]:cv2.imwrite(str(out/f'live_{time.time_ns()}.jpg'),result)
            time.sleep(max(0,1/15-(time.monotonic()-start)))
    finally:
        if cap is not None:cap.release()
        if proc is not None:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
            if proc.stdout:proc.stdout.close()
        preview.close();cv2.destroyAllWindows()
if __name__=='__main__':main()
