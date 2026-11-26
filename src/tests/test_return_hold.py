import unittest
from command_mapping import Mapping
from control_worker import Controller
from tests.test_control import FakeBus
class Tests(unittest.TestCase):
 def test_stop_preserves_last_gripper_command_not_future_goal(self):
  m=Mapping();b=FakeBus(m);c=Controller(b,m,direct=True);c.arm();start=c.trajectory.command.copy();goal=start.copy();goal[5]-=.1
  c.handle({'op':'direct_segment','time':1,'segment_id':1,'plan':{'start':start,'goal':goal,'duration':5.,'hold_gripper':True}},1)
  last=m.to_raw(start)['gripper'];self.assertEqual(c.direct_grip_hold,last)
  b.raw['gripper']+=10;c.hold('stop');self.assertEqual(b.calls[-1][1]['gripper'],last);self.assertIsNone(c.segment)
  c.release();self.assertIsNone(c.direct_grip_hold)
 def test_release_attempts_every_motor_after_status_fault(self):
  m=Mapping();b=FakeBus(m);c=Controller(b,m,direct=True);c.enabled=True
  from types import SimpleNamespace
  b.motors={str(i):SimpleNamespace(id=i) for i in range(1,7)}
  writes=[]
  def fail():raise RuntimeError('Overload status')
  b.disable_torque=fail
  b._write=lambda address,length,motor,value,**kw:writes.append((motor,value))
  b._read=lambda address,length,motor,**kw:(0,0,32 if motor==6 else 0)
  c.release()
  self.assertEqual(writes,[(i,0) for i in range(1,7)])
  self.assertFalse(c.enabled)
 def test_unconfirmed_release_is_not_reported_successful(self):
  m=Mapping();b=FakeBus(m);c=Controller(b,m,direct=True);c.enabled=True
  from types import SimpleNamespace
  b.motors={'gripper':SimpleNamespace(id=6)}
  def fail():raise RuntimeError('Communication fault')
  b.disable_torque=fail;b._write=lambda *a,**kw:None
  b._read=lambda *a,**kw:(1,0,0)
  with self.assertRaisesRegex(RuntimeError,'unconfirmed'):c.release()
  self.assertTrue(c.enabled)
 def test_return_path_keeps_gap(self):
  import json
  from pathlib import Path
  plans=json.loads(Path('calibration/new_object_grasp_preflight.json').read_text())
  close=next(r for r in plans if r['stage']=='close')['plan']['goal']
  for r in plans:
   if r['stage'] in ('withdraw','return','basket_lift','basket_transfer'):
    self.assertEqual(r['plan']['start'][5],close[5]);self.assertEqual(r['plan']['goal'][5],close[5]);self.assertTrue(r['plan']['hold_gripper'])
if __name__=='__main__':unittest.main()
