import unittest,copy
from command_mapping import Mapping,DirectTrajectory,DirectSegment
from tests.test_control import FakeBus
from mirror_grasp_plan import accelerate_first_eighty
class Tests(unittest.TestCase):
 def test_double_tail_at_boundary(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);g=q.copy();g[0]+=.2
  t=DirectTrajectory(m,q);seg=DirectSegment(t,{'start':q,'goal':g,'duration':10.,'speedup_until_s':8.,'speedup_factor':4.,'tail_speedup_factor':2.})
  seg.elapsed=7.8;seg.step(.1)
  self.assertAlmostEqual(seg.elapsed,8.1)
  seg.step(.1);self.assertAlmostEqual(seg.elapsed,8.3)
 def test_invalid_tail_factor(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);t=DirectTrajectory(m,q)
  for factor in (0,5,float('nan')):
   with self.assertRaisesRegex(ValueError,'tail speedup'):
    DirectSegment(t,{'start':q,'goal':q,'duration':4.,'tail_speedup_factor':factor})
 def test_global_rate_doubles_both_sides_of_speed_boundary(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);g=q.copy();g[0]+=.2
  t=DirectTrajectory(m,q)
  plan={'start':q,'goal':g,'duration':10.,'speedup_until_s':8.,'speedup_factor':4.,'tail_speedup_factor':2.,'playback_rate':2.}
  seg=DirectSegment(t,plan);seg.elapsed=7.8;seg.step(.1)
  self.assertAlmostEqual(seg.elapsed,8.3)
  seg.step(.1);self.assertAlmostEqual(seg.elapsed,8.7)
  for _ in range(4):seg.step(.1)
  self.assertEqual(seg.step(.1),g)
 def test_global_rate_validation(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);t=DirectTrajectory(m,q)
  for rate in (0,3,float('nan')):
   with self.assertRaisesRegex(ValueError,'sequence playback rate'):
    DirectSegment(t,{'start':q,'goal':q,'duration':4.,'playback_rate':rate})
 def test_execution_rate_quarters_time_without_changing_path(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);g=q.copy();g[0]+=.2
  plan={'start':q,'goal':g,'duration':10.,'speedup_until_s':8.,'speedup_factor':4.,'tail_speedup_factor':2.,'playback_rate':2.}
  a=DirectSegment(DirectTrajectory(m,q),plan)
  b=DirectSegment(DirectTrajectory(m,q),{**plan,'execution_rate':4.})
  for _ in range(25):
   qa=a.step(.08);qb=b.step(.02)
   for x,y in zip(qa,qb):self.assertAlmostEqual(x,y)
  for rate in (0,5,float('nan')):
   with self.assertRaisesRegex(ValueError,'execution rate'):
    DirectSegment(DirectTrajectory(m,q),{**plan,'execution_rate':rate})
 def test_global_boundary(self):
  reports=[{'plan':{'duration':x}} for x in [10.,30.,10.]]
  accelerate_first_eighty(reports)
  self.assertEqual([r['plan']['speedup_until_s'] for r in reports],[10,30,0])
  self.assertEqual(sum(r['plan']['playback_duration'] for r in reports),20)
 def test_exact_crossing_and_unchanged_tail(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);g=q.copy();g[0]+=.2
  t=DirectTrajectory(m,q);s=DirectSegment(t,{'start':q,'goal':g,'duration':10.,'speedup_until_s':8.})
  for _ in range(39):s.step(.1)
  s.step(.15);self.assertAlmostEqual(s.elapsed,8.05)
  s.step(.1);self.assertAlmostEqual(s.elapsed,8.15)
  normal=DirectSegment(t,{'start':q,'goal':g,'duration':10.});normal.elapsed=8.15

  for a,b in zip(s.step(.1),normal.step(.1)):self.assertAlmostEqual(a,b,places=12)
 def test_four_times_crossing_and_tail(self):
  m=Mapping();q=m.from_raw(FakeBus(m).raw);g=q.copy();g[0]+=.2
  t=DirectTrajectory(m,q);plan={'start':q,'goal':g,'duration':10.,'speedup_until_s':8.,'speedup_factor':4.}
  s=DirectSegment(t,plan)
  for _ in range(19):s.step(.1)
  s.step(.15);self.assertAlmostEqual(s.elapsed,8.05)
  normal=DirectSegment(t,{'start':q,'goal':g,'duration':10.});normal.elapsed=s.elapsed
  for a,b in zip(s.step(.1),normal.step(.1)):self.assertAlmostEqual(a,b,places=12)
  with self.assertRaises(ValueError):DirectSegment(t,{**plan,'speedup_factor':5.})
if __name__=='__main__':unittest.main()
