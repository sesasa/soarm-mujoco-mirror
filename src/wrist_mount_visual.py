"""Apply a provisional visual-only wrist-holder correction, separate from optics."""
import json
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]

def photo_holder(mount, shape):
    """Photo-informed primitives in the original holder's gripper frame."""
    white='0.88 0.89 0.87 1'
    def vec(x):return ' '.join(map(str,x))
    def geom(name, **kw):
        return ET.SubElement(mount,'geom',name='photo_holder_'+name,group='2',
                             contype='0',conaffinity='0',mass='0',**kw)
    def bar(name,a,b,width,thickness):
        a,b=np.array(a),np.array(b);axis=(b-a)/np.linalg.norm(b-a)
        x=np.array([1.,0,0]);y=np.cross(axis,x)
        q=Rotation.from_matrix(np.column_stack([x,y,axis])).as_quat()
        geom(name,type='box',pos=vec((a+b)/2),size=vec([width/2,thickness/2,np.linalg.norm(b-a)/2]),quat=vec(np.r_[q[3],q[:3]]),rgba=white)
    center=np.array(shape['frame_center_m']);width,height=shape['frame_size_m'];rim=shape['frame_rim_m']
    angle=np.deg2rad(shape['frame_tilt_deg'])
    x=np.array([1.,0,0]);up=np.array([0,np.cos(angle),-np.sin(angle)]);normal=np.cross(x,up)
    rot=Rotation.from_matrix(np.column_stack([x,up,normal])).as_quat();quat=vec(np.r_[rot[3],rot[:3]])
    base,knee=np.array(shape['foot_m']),np.array(shape['bend_m'])
    bar('foot',base,knee,shape['support_width_m'],shape['support_thickness_m'])
    bar('inclined_support',knee,center-up*height/2,shape['support_width_m'],shape['support_thickness_m'])
    for side in [-1,1]:
        geom('side_'+str(side),type='box',pos=vec(center+side*x*(width-rim)/2),size=vec([rim/2,height/2,shape['frame_thickness_m']/2]),quat=quat,rgba=white)
        geom('edge_'+str(side),type='box',pos=vec(center+side*up*(height-rim)/2),size=vec([(width-2*rim)/2,rim/2,shape['frame_thickness_m']/2]),quat=quat,rgba=white)
    geom('pcb',type='box',pos=vec(center+normal*.0025),size=vec([(width-2*rim)/2,(height-2*rim)/2,.001]),quat=quat,rgba='0.025 0.10 0.065 1')
    geom('lens_base',type='cylinder',pos=vec(center-normal*.004),size='0.009 0.004',quat=quat,rgba='0.035 0.035 0.035 1')
    geom('lens_barrel',type='cylinder',pos=vec(center-normal*.014),size='0.0065 0.006',quat=quat,rgba='0.025 0.025 0.025 1')
    geom('lens_glass',type='cylinder',pos=vec(center-normal*.0202),size='0.0055 0.0003',quat=quat,rgba='0.06 0.10 0.14 1')

def apply_wrist_mount_visual(root):
    path=ROOT/'config/wrist_mount_visual.json'
    if not path.exists():return
    cfg=json.loads(path.read_text());delta=np.array(cfg['translation_in_gripper_m'],float)
    if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('Invalid visual mount translation')
    quat=np.array(cfg.get('rotation_quat_wxyz',[1,0,0,0]),float)
    pivot=np.array(cfg.get('rotation_pivot_gripper_m',[0,0,0]),float)
    if quat.shape!=(4,) or not np.isfinite(quat).all() or np.linalg.norm(quat)<1e-9 or pivot.shape!=(3,) or not np.isfinite(pivot).all():
        raise ValueError('Invalid visual mount rotation')
    rotation=Rotation.from_quat(np.r_[quat[1:],quat[0]])
    final_shift=np.array(cfg.get('post_rotation_translation_gripper_m',[0,0,0]),float)
    if final_shift.shape!=(3,) or not np.isfinite(final_shift).all():raise ValueError('Invalid final visual translation')
    mount=root.find(".//body[@name='camera_mount']")
    if mount is None:raise ValueError('Camera mount missing')
    if cfg.get('shape'):
        # Keep the legacy mesh's mass contribution, but hide its appearance.
        # Existing collision proxies and the optical camera remain separate.
        old=mount.find("geom[@mesh='wrist_roll_follower_so101_camera_mount']")
        if old is not None:old.set('rgba','0 0 0 0');old.set('group','5')
        board=mount.find("geom[@name='wrist_camera_board']")
        if board is not None:mount.remove(board)
        photo_holder(mount,cfg['shape'])
    for geom in mount.findall('geom'):
        if geom.get('mesh')=='wrist_roll_follower_so101_camera_mount' or geom.get('name')=='wrist_camera_board' or geom.get('name','').startswith('photo_holder_'):
            old=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            geom.set('pos',' '.join(map(str,pivot+rotation.apply(old+delta-pivot)+final_shift)))
            original=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ')
            composed=(rotation*Rotation.from_quat(np.r_[original[1:],original[0]])).as_quat()
            geom.set('quat',' '.join(map(str,np.r_[composed[3],composed[:3]])))
