"""Conservative CPU projection of moving robot meshes over camera targets."""
import json
import cv2,numpy as np,mujoco
from command_mapping import ROOT

class CameraVisibility:
 def __init__(self,checker):
  self.checker=checker
  registration=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
  meta=json.loads((__import__('pathlib').Path(registration['source_session'])/'session.json').read_text())
  self.camera=np.asarray(registration['T_camera_from_base']);self.K=np.asarray(meta['intrinsics']['matrix'])
  self.vertices=[]
  from itertools import product
  for i in checker.geoms:
   m=checker.model
   if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
    mesh=int(m.geom_dataid[i]);points=checker.mesh_hulls[mesh]
   else:
    size=m.geom_size[i].copy()
    if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_SPHERE:size[:]=size[0]
    elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE:size=np.array([size[0],size[0],size[0]+size[1]])
    elif m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CYLINDER:size=np.array([size[0],size[0],size[1]])
    points=np.array(list(product([-1,1],repeat=3)))*size
   self.vertices.append((i,points))
 def projected_mask(self,q):
  self.checker.set_pose(q);data=self.checker.data;mask=np.zeros((480,640),np.uint8)
  for i,points in self.vertices:
   world=points@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i]
   camera=world@self.camera[:3,:3].T+self.camera[:3,3]
   if np.any(camera[:,2]<=.005):return np.ones((480,640),np.uint8)
   pixels=camera@self.K.T;pixels=pixels[:,:2]/pixels[:,2,None]
   hull=cv2.convexHull(pixels.astype(np.float32));cv2.fillConvexPoly(mask,np.rint(hull).astype(np.int32),1)
  mask=cv2.dilate(mask,np.ones((9,9),np.uint8))
  return mask
 def coverage(self,q,polygons):
  mask=self.projected_mask(q)
  result=[]
  for polygon in polygons:
   target=np.zeros_like(mask);cv2.fillPoly(target,[np.rint(polygon).astype(np.int32)],1)
   result.append(float(np.count_nonzero(mask&target)/max(1,np.count_nonzero(target))))
  return result
