import unittest
from adaptive_reach import TargetWatch,committed_grip_valid
from tests.test_named_targets import obs

class OcclusionTests(unittest.TestCase):
 def test_missing_allowed_only_with_commit_deadline(self):
  w=TargetWatch(obs(),0);w.update([],1)
  self.assertFalse(committed_grip_valid(w,1,None))
  self.assertTrue(committed_grip_valid(w,1,3))
  self.assertFalse(committed_grip_valid(w,4,3))
 def test_visible_moved_or_ambiguous_stops(self):
  w=TargetWatch(obs(),0);w.update([obs(.02)],.1)
  self.assertFalse(committed_grip_valid(w,.1,3))
  w=TargetWatch(obs(),0);w.update([obs(),obs(.01)],.1)
  self.assertFalse(committed_grip_valid(w,.1,3))
 def test_reappearance_same_pose(self):
  w=TargetWatch(obs(),0);w.update([],1);w.update([obs()],1.1)
  self.assertTrue(committed_grip_valid(w,1.1,3))

if __name__=='__main__':unittest.main()

class ApproachMemoryTests(unittest.TestCase):
 def stable(self):
  from adaptive_reach import FinalApproachMemory
  m=FinalApproachMemory();w=TargetWatch(obs(),0)
  for t in [.1,.2,.3,.4,.5,.6,.7]:w.update([obs()],t);m.update(w,t)
  return m,w
 def test_disappears_before_hover_confirmation(self):
  m,w=self.stable();w.update([], .8);m.update(w,.8)
  self.assertTrue(m.permits(w,.8,.03))
  self.assertTrue(m.permits(w,2.,.001))
  self.assertFalse(m.permits(w,.8,.10))
  self.assertFalse(m.permits(w,14.,.001))
 def test_motion_or_ambiguity_clears_memory(self):
  for objects in ([obs(.02)],[obs(),obs(.01)]):
   m,w=self.stable();w.update(objects,.8);m.update(w,.8);w.update([], .9)
   self.assertFalse(m.permits(w,.9,.01))
