import json,unittest
import cv2,numpy as np
from command_mapping import ROOT
from basket_tracking import locate_basket

class TrackingTests(unittest.TestCase):
 def setUp(self):
  self.reference=json.loads((ROOT/'config/basket_reference.json').read_text())
  self.image=cv2.imread(str(ROOT/self.reference['source_image']))[:,:640]
 def test_fresh_translation_and_rotation(self):
  transform=cv2.getRotationMatrix2D((490,135),12,1.);transform[:,2]+=[-90,70]
  moved=cv2.warpAffine(self.image,transform,(640,480))
  pixels,evidence=locate_basket(moved,self.reference)
  expected=cv2.transform(np.asarray(self.reference['inner_rim_pixels'],np.float32)[None],transform)[0]
  self.assertLess(float(np.max(np.linalg.norm(pixels-expected,axis=1))),6)
  self.assertTrue(evidence.get('inliers',0)>=8 or evidence.get('match_score',0)>.5)
 def test_missing_basket_never_reuses_saved_position(self):
  with self.assertRaisesRegex(ValueError,'not clearly visible'):
   locate_basket(np.full((480,640,3),140,np.uint8),self.reference)
if __name__=='__main__':unittest.main()
