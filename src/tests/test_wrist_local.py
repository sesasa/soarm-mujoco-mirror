import unittest
import cv2
import numpy as np
from wrist_local import PatchTracker

class WristTests(unittest.TestCase):
    def scene(self):
        im=np.zeros((480,640,3),np.uint8)
        rng=np.random.default_rng(5)
        for x,y in rng.integers([220,160],[350,290],(70,2)):
            cv2.circle(im,(int(x),int(y)),3,(240,240,240),-1)
        return im
    def test_translation_rotation(self):
        im=self.scene();t=PatchTracker();self.assertTrue(t.select(im,(210,150,360,300),1.))
        matrix=cv2.getRotationMatrix2D((285,225),3,1);matrix[:,2]+=[4,-3]
        moved=cv2.warpAffine(im,matrix,(640,480))
        f=t.update(moved,1.05,1.06)
        np.testing.assert_allclose(f['center_px'],[289,222],atol=.5)
        self.assertAlmostEqual(f['rotation_deg'],-3,delta=.4)
    def test_stale_and_blank(self):
        im=self.scene();t=PatchTracker();t.select(im,(210,150,360,300),1.)
        self.assertIsNone(t.update(im,1.,1.4));self.assertIsNone(t.box)
        t.select(im,(210,150,360,300),2.)
        self.assertIsNone(t.update(np.zeros_like(im),2.05,2.06))
    def test_repeated_and_reversed(self):
        im=self.scene();t=PatchTracker();t.select(im,(210,150,360,300),1.)
        self.assertIsNotNone(t.update(im,1.,1.01))
        self.assertIsNone(t.update(im,.99,1.01))
    def test_no_texture(self):
        t=PatchTracker();self.assertFalse(t.select(np.zeros((480,640,3),np.uint8),(100,100,200,200),1.))

if __name__=='__main__':unittest.main()
