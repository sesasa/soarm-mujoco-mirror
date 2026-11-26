import math,unittest
from command_mapping import Mapping,DirectTrajectory,DirectSegment
from control_worker import Controller
from tests.test_control import FakeBus
class Tests(unittest.TestCase):
 def setUp(self):
  self.m=Mapping();self.b=FakeBus(self.m);self.q=self.m.from_raw(self.b.raw)
  g=self.q.copy();g[0]+=.3;self.plan={'start':self.q,'goal':g,'duration':10.}
 def test_smooth_bounded_complete(self):
  t=DirectTrajectory(self.m,self.q);s=DirectSegment(t,self.plan);q=self.q;steps=[]
  for i in range(165):
   goal=s.step(1/15);n=t.step(goal,q,1/15);steps.append(abs(n[0]-q[0]));q=n
  self.assertLess(steps[0],max(steps)/100);self.assertLessEqual(max(steps),t.SPEED/15)
  self.assertEqual(q,self.plan['goal'])
 def test_invalid_start_speed_and_time(self):
  t=DirectTrajectory(self.m,self.q)
  with self.assertRaises(ValueError):DirectSegment(t,{**self.plan,'duration':4})
  bad=self.q.copy();bad[0]+=.01
  with self.assertRaises(ValueError):DirectSegment(t,{**self.plan,'start':bad})
  s=DirectSegment(t,self.plan)
  with self.assertRaises(ValueError):s.step(.5)
 def test_stop_cancels_segment(self):
  c=Controller(self.b,self.m,direct=True);c.arm();c.handle({'op':'direct_segment','time':1,'plan':self.plan,'segment_id':0},1)
  self.assertIsNotNone(c.segment);c.hold('stop');self.assertIsNone(c.segment)
 def test_opt_in(self):
  c=Controller(self.b,self.m);c.arm()
  with self.assertRaises(ValueError):c.handle({'op':'direct_segment','time':1,'plan':self.plan,'segment_id':0},1)
if __name__=='__main__':unittest.main()
