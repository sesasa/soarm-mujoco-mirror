import json,unittest
import cv2,numpy as np
from command_mapping import ROOT
from plate_detection import locate_coloured_plate,ellipses

class Tests(unittest.TestCase):
 def setUp(self):
  self.ref=json.loads((ROOT/'config/basket_reference.json').read_text())
  self.image=cv2.imread(str(ROOT/self.ref['source_image']))[:,:640]
  self.polygon=np.asarray(self.ref['inner_rim_pixels'],np.float32)
 def test_deposited_object_and_partial_rim_occlusion(self):
  c=self.polygon.mean(axis=0).astype(int)
  for dx in (0,35):
   frame=self.image.copy();cv2.rectangle(frame,(c[0]+dx-25,c[1]-35),(c[0]+dx+25,c[1]+35),(35,35,35),-1)
   result=locate_coloured_plate(self.image,frame,self.polygon,self.ref.get("plate_color","yellow"))
   self.assertIsNotNone(result)
   pixels,_=result
   self.assertLess(np.linalg.norm(pixels.mean(axis=0)-self.polygon.mean(axis=0)),8)
 def test_missing_plate_has_no_detection(self):
  self.assertIsNone(locate_coloured_plate(self.image,np.full_like(self.image,140),self.polygon,self.ref.get("plate_color","yellow")))
 def test_plate_moved_toward_camera(self):
  frame=cv2.imread(str(ROOT/'src/tests/fixtures/red_plate_near_camera.jpg'))
  result=locate_coloured_plate(self.image,frame,self.polygon,'red')
  self.assertIsNotNone(result)
  self.assertLess(np.linalg.norm(result[0].mean(axis=0)-[282,196]),8)
  from preview_camera_alignment import Preview
  from scene_changes import Plane
  from basket_tracking import measure_basket
  preview=Preview()
  try:
   plane=Plane(preview.K,preview.T,np.array(json.loads((ROOT/'captures/scene_changes/plane.json').read_text())['T_camera_from_board']))
   measured=measure_basket(frame,self.ref,plane,preview.T)
   self.assertGreater(measured['opening_size_m'][0],.10)
   self.assertLess(measured['opening_size_m'][0],.16)
  finally:preview.close()
if __name__=='__main__':unittest.main()
