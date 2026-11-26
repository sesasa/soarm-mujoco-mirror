import math,unittest
from command_mapping import Mapping,DirectTrajectory
from control_worker import Controller
from tests.test_control import FakeBus
class Tests(unittest.TestCase):
 def setUp(self):self.m=Mapping();self.b=FakeBus(self.m);self.q=self.m.from_raw(self.b.raw)
 def test_large_slider_target_rate_limited(self):
  t=DirectTrajectory(self.m,self.q);g=self.q.copy();g[0]+=.5
  q=t.step(g,self.q,.1);self.assertAlmostEqual(q[0]-self.q[0],math.radians(.7))
  self.assertEqual(t.limits,self.m.limits)
 def test_resting_margin_only_inward(self):
  q=self.q.copy();q[1]=self.m.limits[1][0]-.03
  t=DirectTrajectory(self.m,q);g=q.copy();g[1]+=.02
  q2=t.step(g,q,.1);t.command_mapping.to_raw(q2)
  with self.assertRaises(ValueError):t.step(q,q2,.1)
 def test_no_far_outside_or_tracking_bypass(self):
  q=self.q.copy();q[1]-=3
  with self.assertRaises(ValueError):DirectTrajectory(self.m,q)
  t=DirectTrajectory(self.m,self.q);q=self.q.copy();q[0]+=.2
  with self.assertRaises(ValueError):t.step(self.q,q,.1)
 def test_resting_enable_order(self):
  q=self.q.copy();q[1]=self.m.limits[1][0]-.03;t=DirectTrajectory(self.m,q);self.b.raw=t.command_mapping.to_raw(q)
  c=Controller(self.b,self.m,direct=True);c.arm()
  self.assertEqual(c.state,'ACTIVE');self.assertEqual(self.b.calls[:2],[('write',self.b.raw),('enable',)])
  c.release();self.assertFalse(self.b.torque)
 def test_modes_exclusive(self):
  with self.assertRaises(ValueError):Controller(self.b,self.m,allow_grasp=True,direct=True)
if __name__=='__main__':unittest.main()
