import unittest
import numpy as np
from simulated_reach import ReachProgress

class ReachProgressTest(unittest.TestCase):
    def setUp(self):
        self.region=(np.array([.15,.18,0]),np.array([.05,.015,.001]),np.eye(3))
        self.progress=ReachProgress(self.region)

    def advance(self, regions, count=6):
        for _ in range(count):self.progress.update(regions,.1)

    def test_transient_loss_pauses_and_resumes_without_time_jump(self):
        self.advance([self.region]);before=self.progress.elapsed
        self.assertFalse(self.progress.paused)
        self.advance([],20)
        self.assertTrue(self.progress.paused)
        self.assertEqual(self.progress.elapsed,before)
        self.advance([self.region],3)
        self.assertTrue(self.progress.paused)
        self.assertEqual(self.progress.elapsed,before)
        self.advance([self.region],1)
        self.assertFalse(self.progress.paused)
        self.assertAlmostEqual(self.progress.elapsed,before+.1)

    def test_moved_or_duplicate_target_stays_paused(self):
        self.advance([self.region]);before=self.progress.elapsed
        c,s,r=self.region
        for regions in ([(c+[.03,0,0],s,r)],[self.region,self.region]):
            self.advance(regions)
            self.assertTrue(self.progress.paused)
            self.assertEqual(self.progress.elapsed,before)
        np.testing.assert_array_equal(self.progress.center,c)

    def test_rotated_or_merged_footprint_stays_paused(self):
        c,s,r=self.region
        quarter_turn=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        for regions in ([(c,s,quarter_turn)],[(c,s*2,r)]):
            self.advance(regions)
            self.assertTrue(self.progress.paused)
            self.assertEqual(self.progress.elapsed,0)

    def test_detection_order_and_selection_ui_not_required(self):
        c,s,r=self.region
        other=(c+[.04,0,0],s,r)
        self.advance([other,self.region])
        self.assertFalse(self.progress.paused)
        self.advance([self.region,other])
        self.assertFalse(self.progress.paused)

if __name__=='__main__':unittest.main()
