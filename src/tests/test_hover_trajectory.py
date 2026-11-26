import copy,math,unittest
from hover_trajectory import HoverTrajectory
from command_mapping import Mapping

class HoverTests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.q=[0,0,0,0,-1,.4];self.goal=self.q.copy();self.goal[1]=math.radians(30)
        self.r={'mode':'position_only_high_hover','height_above_table_m':.12,'joint_mapping':self.m.mapping,'prepared_at_unix':100.,'plan':{'start':self.q,'goal':self.goal,'duration':20.},'clearance':{'path_minimum_m':.009,'gripper_holder_path_minimum_m':.1,'samples':161}}
    def test_speed_endpoint_and_settle(self):
        t=HoverTrajectory(self.r,self.m,self.q,self.q,100.);q=self.q.copy()
        for _ in range(221):
            nxt=t.step(q,.1)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(q,nxt)),math.radians(.3)+1e-9)
            self.assertEqual(nxt[5],self.q[5]);q=nxt
        self.assertTrue(t.done)
        self.assertEqual(q,self.goal)
    def test_stale_wrong_mapping_start_and_jaw_refused(self):
        for kind in ['age','mapping','command','jaw','speed','travel','clearance','nan','samples']:
            r=copy.deepcopy(self.r);cmd=self.q.copy()
            if kind=='age':r['prepared_at_unix']=0
            if kind=='mapping':r['joint_mapping']={}
            if kind=='command':cmd[0]+=.01
            if kind=='jaw':r['plan']['goal'][5]+=.01
            if kind=='speed':r['plan']['duration']=5
            if kind=='travel':r['plan']['goal'][1]=math.radians(65);r['plan']['duration']=60
            if kind=='clearance':r['clearance']['gripper_holder_path_minimum_m']=.07
            if kind=='nan':r['plan']['duration']=float('nan')
            if kind=='samples':r['clearance']['samples']=80
            with self.subTest(kind=kind),self.assertRaises(ValueError):HoverTrajectory(r,self.m,cmd,self.q,100)
    def test_feedback_lag_stale_interval_and_bad_settle(self):
        for dt in [0,.3,float('nan')]:
            t=HoverTrajectory(self.r,self.m,self.q,self.q,100)
            with self.assertRaises(ValueError):t.step(self.q,dt)
        t=HoverTrajectory(self.r,self.m,self.q,self.q,100);q=self.q.copy();q[0]+=.1
        with self.assertRaises(ValueError):t.step(q,.1)
        t=HoverTrajectory(self.r,self.m,self.q,self.q,100);t.elapsed=22;t.command=self.goal.copy();q=self.goal.copy();q[0]+=math.radians(2.5)
        with self.assertRaises(ValueError):t.step(q,.1)
