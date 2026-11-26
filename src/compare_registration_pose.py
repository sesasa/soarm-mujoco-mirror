"""Project the encoder-derived robot model into a saved image; no hardware I/O."""
import argparse
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from mirror_robot import convert
from simulate import JOINTS


def compare(image, observation, output):
    m=mujoco.MjModel.from_xml_path('assets/so101/so101.xml');d=mujoco.MjData(m)
    p=json.loads(observation.read_text());q=convert(p['positions'],json.load(open('config/joint_mapping.json')))
    for n,v in zip(JOINTS,q):d.qpos[m.joint(n).qposadr[0]]=v
    mujoco.mj_forward(m,d)
    a=json.load(open('calibration/table_registration/base_registration_candidate.json'))
    T=np.array(a['T_camera_from_base']);c=json.load(open('calibration/cameras/overhead_640x480.json'))
    K=np.array(c['matrix']);dist=np.array(c['distortion']);r=cv2.Rodrigues(T[:3,:3])[0]
    f=cv2.imread(str(image))
    if f is None or list(f.shape[1::-1])!=c['size']:raise ValueError('Wrong image dimensions')
    for body in range(1,m.nbody):
        points=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=body or m.geom_group[g]!=2 or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:continue
            mesh=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
            points.extend(v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
        if not points:continue
        uv=cv2.projectPoints(np.array(points),r,T[:3,3],K,dist)[0]
        h=cv2.convexHull(uv.astype(np.float32));cv2.polylines(f,[np.rint(h).astype(np.int32)],True,(255,180,0),1)
    projected=cv2.projectPoints(d.xanchor,r,T[:3,3],K,dist)[0].reshape(-1,2)
    output.parent.mkdir(exist_ok=True,parents=True);cv2.imwrite(str(output),f)
    report={'image':str(image),'observation':str(observation),'joint_positions_rad':q.tolist(),'joint_projected_pixels':dict(zip(JOINTS,projected.tolist())),'status':'visual_comparison_only','notes':'Per-body convex outlines include hidden surfaces; not exact visible silhouettes. Images and encoder readings acquired sequentially while arm assumed stationary.'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('image',type=Path);p.add_argument('observation',type=Path);p.add_argument('output',type=Path);a=p.parse_args();compare(a.image,a.observation,a.output)
