import unittest
import numpy as np
from scene_changes import TargetSelection

class TargetSelectionTest(unittest.TestCase):
    def setUp(self):
        self.a=np.array([[10,10],[30,10],[30,60],[10,60]],np.float32)
        self.b=self.a+[60,0]
        self.selection=TargetSelection()

    def test_select_and_follow_reordered_detections(self):
        self.selection.click((20.,30.),[self.a,self.b])
        self.selection.update([self.b,self.a+[2,1]])
        self.assertEqual(self.selection.index,1)

    def test_disappearance_clears_without_switching_objects(self):
        self.selection.click((20.,30.),[self.a,self.b])
        self.selection.update([self.b])
        self.assertIsNone(self.selection.index)
        self.assertIsNone(self.selection.polygon)
        self.selection.update([self.a,self.b])
        self.assertIsNone(self.selection.index)

    def test_ambiguous_click_does_not_select(self):
        self.selection.click((20.,30.),[self.a,self.a+[2,0]])
        self.assertIsNone(self.selection.index)

    def test_split_or_duplicate_match_clears(self):
        self.selection.click((20.,30.),[self.a])
        self.selection.update([self.a,self.a+[1,0]])
        self.assertIsNone(self.selection.index)

if __name__=='__main__':unittest.main()
