import json,unittest
import numpy as np
from basket_place import basket_visuals,BasketClearance
from command_mapping import ROOT

class BasketTests(unittest.TestCase):
 def test_measured_rim_matches_configured_height(self):
  registration=json.loads((ROOT/'calibration/robot_camera/overhead_from_base_provisional.json').read_text())
  board=json.loads((ROOT/'captures/scene_changes/plane.json').read_text())
  transform=np.linalg.inv(registration['T_camera_from_base'])@np.asarray(board['T_camera_from_board'])
  up=transform[:3,2];up=up if up[2]>0 else -up
  config=json.loads((ROOT/'config/basket.json').read_text())
  heights=(np.asarray(config['rim_polygon_base_m'])-transform[:3,3])@up
  np.testing.assert_allclose(heights,config['rim_height_m'],atol=1e-8)
  self.assertEqual(len(basket_visuals(config)),len(heights))
 def test_payload_must_fit_inside_rim(self):
  config={'center_base_m':[0,0,.04],'rotation':np.eye(3).tolist(),'rim_polygon_base_m':[[-.08,-.04,.04],[.08,-.04,.04],[.08,.04,.04],[-.08,.04,.04]]}
  checker=type('Checker',(),{'normal':np.array([0,0,1.])})()
  b=BasketClearance(checker,config)
  self.assertTrue(b.inside(np.array([[-.05,-.02,.08],[.05,.02,.08]])))
  self.assertFalse(b.inside(np.array([[0,.045,.08]])))
if __name__=='__main__':unittest.main()
