import unittest
import numpy as np,mujoco
from mesh_support import mesh_hulls
from command_mapping import ROOT

class Tests(unittest.TestCase):
 def test_support_matches_original_robot_meshes(self):
  model=mujoco.MjModel.from_xml_path(str(ROOT/'assets/so101/so101.xml'))
  hulls=mesh_hulls(model);self.assertIs(hulls,mesh_hulls(model))
  normals=np.random.default_rng(42).normal(size=(3,40))
  before=after=0
  for i,hull in enumerate(hulls):
   a=int(model.mesh_vertadr[i]);n=int(model.mesh_vertnum[i]);raw=model.mesh_vert[a:a+n].astype(float)
   np.testing.assert_allclose(np.min(raw@normals,axis=0),np.min(hull@normals,axis=0),atol=1e-12)
   before+=n;after+=len(hull)
  self.assertLess(after,before/2)
if __name__=='__main__':unittest.main()
