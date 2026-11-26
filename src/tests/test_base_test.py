import math,unittest
from command_mapping import Mapping,Trajectory,BaseTestTrajectory
class BaseTestTests(unittest.TestCase):
    def setUp(self):self.m=Mapping();self.q=[-.9,-1.4,1.3,.4,-1.2,.4]
    def test_base_excursion_and_speed_only(self):
        t=BaseTestTrajectory(self.m,self.q);goal=self.q.copy();goal[0]-=math.radians(10)
        command=t.step(goal,self.q,.1)
        self.assertAlmostEqual(command[0]-self.q[0],-math.radians(.3))
        self.assertEqual(command[1:],self.q[1:])
        goal[0]-=.001
        with self.assertRaises(ValueError):t.step(goal,command,.1)
    def test_other_joints_locked(self):
        for i in range(1,6):
            t=BaseTestTrajectory(self.m,self.q);goal=self.q.copy();goal[i]+=.01
            with self.assertRaises(ValueError):t.step(goal,self.q,.1)
    def test_default_window_unchanged(self):
        t=Trajectory(self.m,self.q);goal=self.q.copy();goal[0]+=.12
        with self.assertRaises(ValueError):t.step(goal,self.q,.1)
    def test_base_window_intersects_model_limits(self):
        q=self.q.copy();q[0]=self.m.limits[0][1]-.01;t=BaseTestTrajectory(self.m,q)
        self.assertEqual(t.limits[0][1],self.m.limits[0][1])
