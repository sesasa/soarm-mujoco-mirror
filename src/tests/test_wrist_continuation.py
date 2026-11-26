import unittest
from wrist_local import WristContinuation

class Tests(unittest.TestCase):
 def test_requires_visible_patch_near_target(self):
  for features,stamp,remaining in [(None,1.,.03),({},.1,.03),({},1.,.07),({},1.,float('nan'))]:
   with self.assertRaises(ValueError):WristContinuation().start(object(),features,stamp,1.1,remaining)
 def test_bound_lifetime_and_frame_freshness(self):
  w=WristContinuation();w.start(object(),{'points':12},1.,1.1,.04)
  self.assertTrue(w.valid({'points':12},2.,2.1))
  self.assertFalse(w.valid(None,2.,2.1))
  self.assertFalse(w.valid({'points':12},2.,3.))
  self.assertFalse(w.valid({'points':12},22.,22.1))
 def test_unarmed_never_allows(self):
  self.assertFalse(WristContinuation().valid({},1.,1.1))

if __name__=='__main__':unittest.main()
