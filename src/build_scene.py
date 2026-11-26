"""Build a measured/estimated desk scene around the vendored SO-101 model."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def vec(values):
    return ' '.join(str(v) for v in values)

def build():
    cfg = json.loads((ROOT / 'config/scene.json').read_text())
    root = ET.parse(ROOT / 'assets/so101/so101.xml').getroot()
    root.set('model', 'SO101 desk — simulation only')
    root.find('compiler').set('meshdir', 'assets/so101/assets')
    root.find('option').set('timestep', '0.002')
    for mat in root.findall('asset/material'):
        if 'sts3215' not in mat.get('name', ''):
            mat.set('rgba', '0.88 0.89 0.87 1')
    # Keep actuator requests inside joint limits, including upstream rounding differences.
    for actuator in root.findall('actuator/position'):
        joint = root.find(f".//joint[@name='{actuator.get('joint')}']")
        actuator.set('ctrlrange', joint.get('range'))
    ET.SubElement(root.find('visual'), 'global', offwidth='960', offheight='720')
    world = root.find('worldbody')
    world.find("body[@name='base']").set('pos',vec(cfg.get('robot_base_position_m',[0,0,0])))
    ET.SubElement(world, 'light', pos='0.3 0 1.5', dir='0 0 -1', directional='true')
    def geom(parent, **attrs):
        return ET.SubElement(parent, 'geom', {k: str(v) for k,v in attrs.items()})
    from table_geometry import add_table
    add_table(root,cfg['table_outline_m'],cfg['table_thickness_m'])
    geom(world, name='floor', type='plane', size='2 2 0.01', pos='0 0 -0.75', rgba='0.22 0.25 0.29 1')
    remote_size = cfg['remote_size_m']
    remote = ET.SubElement(world, 'body', name='remote', pos=vec([*cfg['remote_position_m'],remote_size[2]/2]))
    ET.SubElement(remote, 'freejoint')
    geom(remote, name='remote_shell', type='box', size=vec([s/2 for s in remote_size]), mass='0.12', rgba='0.055 0.06 0.065 1')
    for row in range(9):
        for col in range(3):
            geom(remote, type='sphere', size='0.003', pos=vec([-0.085+row*0.021,(col-1)*0.012,remote_size[2]/2]),
                 rgba='0.7 0.7 0.7 1', contype='0', conaffinity='0', mass='0')
    for name,pos,size,color in [
        ('tape_measure',cfg['tape_measure_position_m'],[0.035,0.023],'0.52 0.54 0.55 1'),
        ('candle',cfg['candle_position_m'],[0.021,0.013],'0.86 0.85 0.8 1')]:
        body=ET.SubElement(world,'body',name=name,pos=vec(pos));ET.SubElement(body,'freejoint')
        geom(body,type='cylinder',size=vec(size),mass='0.07',rgba=color)
    # Candle inside the holder, visible in the latest phone image.
    inside=cfg['wire_holder_position_m']
    geom(world,name='holder_candle',type='cylinder',size='0.017 0.009',pos=vec([inside[0],inside[1],0.009]),rgba='0.8 0.8 0.76 1')
    # Approximate open wire holder, preserving its empty center.
    holder=ET.SubElement(world,'body',name='wire_holder',pos=vec(cfg['wire_holder_position_m']))
    import math
    for i in range(20):
        a,b=2*math.pi*i/20,2*math.pi*(i+1)/20
        geom(holder,type='capsule',size='0.002',fromto=vec([0.04*math.cos(a),0.04*math.sin(a),0.055,0.04*math.cos(b),0.04*math.sin(b),0.055]),rgba='0.06 0.06 0.06 1')
    for i in range(3):
        a=2*math.pi*i/3
        geom(holder,type='capsule',size='0.002',fromto=vec([0.04*math.cos(a),0.04*math.sin(a),0.055,0.045*math.cos(a),0.045*math.sin(a),0.003]),rgba='0.06 0.06 0.06 1')
    geom(world,name='camera_stand_base',type='cylinder',size='0.085 0.012',pos=vec([*cfg['stand_position_m'],0.012]),rgba='0.05 0.05 0.05 1')
    overhead=cfg['overhead_position_m']
    sx,sy=cfg['stand_position_m']
    geom(world,type='capsule',size='0.009',fromto=vec([sx,sy,0.025,sx-0.13,sy,0.40]),rgba='0.06 0.06 0.06 1')
    geom(world,type='capsule',size='0.009',fromto=vec([sx-0.13,sy,0.40,*overhead]),rgba='0.06 0.06 0.06 1')
    geom(world,name='overhead_camera_housing',type='box',size='0.022 0.016 0.01',pos=vec([overhead[0],overhead[1],overhead[2]+0.014]),rgba='0.07 0.09 0.07 1')
    import numpy as np
    z=np.array(overhead)-np.array(cfg['overhead_lookat_m']);z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    ET.SubElement(world,'camera',name='overhead',pos=vec(overhead),xyaxes=vec(cfg.get('overhead_xyaxes', [*x,*y])),fovy=str(cfg['overhead_fovy_deg']))
    # The wrist mount comes from the robot asset; add a visible camera board/lens.
    mount=root.find(".//body[@name='camera_mount']")
    geom(mount,name='wrist_camera_board',type='box',size='0.019 0.002 0.015',pos='0 0.061 -0.04',rgba='0.05 0.13 0.09 1',contype='0',conaffinity='0',mass='0')
    wrist=mount.find('camera')
    for attr in ['resolution','sensorsize','focal']:wrist.attrib.pop(attr,None)
    wrist.set('fovy',str(cfg.get('wrist_fovy_deg',65)))
    if 'wrist_camera_position_m' in cfg:wrist.set('pos',vec(cfg['wrist_camera_position_m']))
    if 'wrist_camera_euler_rad' in cfg:wrist.set('euler',vec(cfg['wrist_camera_euler_rad']))
    if 'wrist_camera_xyaxes' in cfg:
        wrist.attrib.pop('euler',None)
        wrist.set('xyaxes',vec(cfg['wrist_camera_xyaxes']))
    from wrist_mount_visual import apply_wrist_mount_visual
    apply_wrist_mount_visual(root)
    ET.indent(root)
    ET.ElementTree(root).write(ROOT/'scene.xml',encoding='unicode')
    return cfg

if __name__=='__main__':
    build()
    print(ROOT/'scene.xml')
