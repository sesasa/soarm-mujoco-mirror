import unittest
import cv2
import numpy as np
from measured_objects import highlighter_proxy

class MeasuredObjectTest(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((160,220,3),(80,150,220),np.uint8)
        self.contours=[np.array([[30,30],[55,30],[55,130],[30,130]],np.int32),np.array([[100,30],[120,30],[120,130],[100,130]],np.int32)]
        cv2.drawContours(self.frame,self.contours,0,(90,40,240),-1)
        cv2.drawContours(self.frame,self.contours,1,(220,70,30),-1)
        self.regions=[(np.array([.1,.1,0]),np.array([.016,.05,.001]),np.diag([1.,-1.,-1.])),
                      (np.array([.2,.1,0]),np.array([.012,.06,.001]),np.eye(3))]
        self.profile={'height_m':.017,'width_m':.027}

    def test_measured_dimensions_rest_on_table_and_blue_unchanged(self):
        display,index,report=highlighter_proxy(self.frame,self.contours,self.regions,self.profile)
        self.assertEqual(index,0)
        center,size,R=display[0]
        self.assertAlmostEqual(center[2]-size[2],0)
        self.assertAlmostEqual(center[2]+size[2],.017)
        self.assertAlmostEqual(2*size[0],.027)
        self.assertAlmostEqual(2*size[1],.1)
        np.testing.assert_array_equal(display[1][1],self.regions[1][1])
        np.testing.assert_array_equal(self.regions[0][1],[.016,.05,.001])
        self.assertFalse(report['collision_validated'])

    def test_two_pink_candidates_are_not_assigned(self):
        cv2.drawContours(self.frame,self.contours,1,(90,40,240),-1)
        _,index,report=highlighter_proxy(self.frame,self.contours,self.regions,self.profile)
        self.assertIsNone(index);self.assertIsNone(report)

    def test_no_pink_does_not_assign_blue(self):
        cv2.drawContours(self.frame,self.contours,0,(220,70,30),-1)
        _,index,report=highlighter_proxy(self.frame,self.contours,self.regions,self.profile)
        self.assertIsNone(index);self.assertIsNone(report)

if __name__=='__main__':unittest.main()
