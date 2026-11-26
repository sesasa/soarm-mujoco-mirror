import json,unittest
import numpy as np
from command_mapping import ROOT
from scene_changes import Plane
from reach_retarget import camera_left_offset

class OffsetTests(unittest.TestCase):
 def test_one_cm_is_left_in_image_and_parallel_to_table(self):
  registration=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
  meta=json.loads((__import__('pathlib').Path(registration['source_session'])/'session.json').read_text())
  plane=Plane(np.array(meta['intrinsics']['matrix']),np.array(registration['T_camera_from_base']),np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
  for xy in ([0,0],[.1,.1],[-.1,.05]):
   center=(plane.base_from_board@np.r_[xy,0,1])[:3]
   offset=camera_left_offset(plane,center,.01)
   self.assertAlmostEqual(np.linalg.norm(offset),.01)
   self.assertAlmostEqual(offset@plane.base_from_board[:3,2],0.)
   def pixel(point):
    board=np.linalg.inv(plane.base_from_board)@np.r_[point,1.];p=plane.H@np.r_[board[:2],1.];return p[:2]/p[2]
   a,b=pixel(center),pixel(center+offset)
   self.assertLess(b[0],a[0]);self.assertAlmostEqual(a[1],b[1],places=7)
if __name__=='__main__':unittest.main()
