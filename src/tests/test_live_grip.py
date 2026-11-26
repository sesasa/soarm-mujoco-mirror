import unittest,numpy as np
from live_grip import JointServo
from command_mapping import Mapping
from control_worker import Controller
from tests.test_control import FakeBus

class MotionTests(unittest.TestCase):
 def test_continuous_targets_bounded_velocity_and_acceleration(self):
  servo=JointServo(np.zeros(6));last=np.zeros(6)
  for i in range(1,240):
   now=i/30
   goal=np.full(6,.6 if i<70 else -.2 if i<150 else .1)
   servo.update(goal,now,now)
   q=servo.step(1/30,now)
   self.assertLessEqual(np.max(abs(servo.velocity)),servo.speed+1e-10)
   self.assertLessEqual(np.max(abs(servo.velocity-last)),servo.acceleration/30+1e-10)
   last=servo.velocity.copy()
  np.testing.assert_allclose(q,.1,atol=1e-3)
 def test_no_stale_or_repeated_observations(self):
  servo=JointServo(np.zeros(6))
  with self.assertRaises(ValueError):servo.update([0]*6,1,1.3)
  servo.update([.1]*6,1,1.1)
  with self.assertRaises(ValueError):servo.update([.1]*6,1,1.2)
  with self.assertRaises(ValueError):servo.step(1/30,1.31)
 def test_stale_control_interval_and_invalid_goal(self):
  servo=JointServo(np.zeros(6));servo.update([0]*6,1,1)
  for dt in [0,.3,float('nan')]:
   with self.assertRaises(ValueError):servo.step(dt,1)
  with self.assertRaises(ValueError):servo.update([float('nan')]*6,1.1,1.1)
 def test_recorded_wrist_endpoint_never_exceeds_controller_limit(self):
  from command_mapping import DirectTrajectory
  mapping=Mapping()
  start=np.array([.013042,-1.727876,1.529081,1.035594,-1.580770,.628326])
  goal=np.array([-.779839,.505569,-.331447,1.376029089937727,-1.580770,.628326])
  servo=JointServo(start);tr=DirectTrajectory(mapping,start);q=start.copy()
  for i in range(1,301):
   now=i/30;servo.update(goal,now,now)
   target=servo.step(1/30,now,limits=tr.limits)
   q=tr.step(target,q,1/30,speed_limit=servo.speed)
   self.assertLessEqual(q[3],mapping.limits[3][1])
  np.testing.assert_allclose(q,goal,atol=1e-4)

class WorkerTests(unittest.TestCase):
 def setUp(self):self.mapping=Mapping();self.bus=FakeBus(self.mapping)
 def test_opt_in_and_motor_off_rejected(self):
  c=Controller(self.bus,self.mapping,direct=True);c.arm()
  with self.assertRaises(ValueError):c.handle({'op':'servo_target','time':1,'frame_time':1,'q':c.trajectory.command},1)
  c=Controller(self.bus,self.mapping,direct=True,visual_servo=True)
  with self.assertRaises(ValueError):c.handle({'op':'servo_target','time':1,'frame_time':1,'q':[0]*6},1)
 def test_stream_pause_and_user_hold(self):
  c=Controller(self.bus,self.mapping,direct=True,visual_servo=True);c.arm()
  goal=c.trajectory.command.copy();goal[0]+=.1
  c.handle({'op':'servo_target','time':1,'frame_time':1,'q':goal},1)
  self.assertIsNotNone(c.servo)
  c.handle({'op':'direct_pause','time':1.1,'pause_id':4},1.1)
  self.assertIsNone(c.servo);self.assertEqual(c.state,'ACTIVE')
  c.handle({'op':'servo_target','time':1.2,'frame_time':1.2,'q':goal},1.2)
  c.hold('operator');self.assertIsNone(c.servo);self.assertEqual(c.state,'HOLD')
 def test_does_not_replace_an_active_segment(self):
  c=Controller(self.bus,self.mapping,direct=True,visual_servo=True);c.arm();q=c.trajectory.command.copy()
  c.handle({'op':'direct_segment','time':1,'segment_id':0,'plan':{'start':q,'goal':q,'duration':4}},1)
  with self.assertRaises(ValueError):c.handle({'op':'servo_target','time':1.1,'frame_time':1.1,'q':q},1.1)

if __name__=='__main__':unittest.main()
