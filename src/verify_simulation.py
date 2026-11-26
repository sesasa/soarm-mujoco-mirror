"""Check all six controls near the starting pose, including command speed bounds."""
import numpy as np
from simulate import Simulation

s=Simulation()
assert s.model.nu==6
assert s.model.camera('overhead').id>=0
assert s.model.camera('wrist_cam').id>=0
np.testing.assert_allclose(s.model.geom('remote_shell').size[:2],[0.11,0.025])
for _ in range(1500):s.step()
for i in range(6):
    baseline=s.target.copy()
    s.target[i]+=0.15
    for _ in range(1000):
        old=s.command.copy()
        s.step()
        assert np.max(np.abs(s.command-old))<=s.config['command_speed_rad_s']*s.model.opt.timestep+1e-10
        assert np.isfinite(s.data.qpos).all()
    error=abs(s.data.qpos[s.addresses[i]]-s.target[i])
    assert error<0.02,(i,error)
    s.target=baseline
    for _ in range(1000):s.step()
# A concave outline must retain its area after decomposition.
from table_geometry import triangulate
points,triangles=triangulate(s.config['table_outline_m'])
def cross(a,b):return a[0]*b[1]-a[1]*b[0]
polygon_area=abs(sum(cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0))))/2
triangle_area=sum(abs(cross(points[b]-points[a],points[c]-points[a]))/2 for a,b,c in triangles)
np.testing.assert_allclose(triangle_area,polygon_area)
for name in ['remote','tape_measure','candle']:
    assert s.data.body(name).xpos[2]>0, f'{name} fell off the table'
assert not any(s.data.warning.number)
print('PASS: six controls, speed bounds, camera frames, remote dimensions, concave tabletop and stable local physics')
