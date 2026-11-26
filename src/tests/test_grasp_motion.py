import copy,math,unittest
from command_mapping import Mapping
from grasp_motion import GraspMotion
class GraspMotionTests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.q=[-.9,-1.,1.,.4,-1.,.4];g=self.q.copy();g[1]+=.5
        self.r={'mode':'supervised_grasp','stage':'approach','joint_mapping':self.m.mapping,'prepared_at_unix':100.,'reference_height_m':.04,'plan':{'start':self.q,'goal':g,'duration':20.},'clearance':{'samples':161,'path_minimum_m':.009,'gripper_holder_path_minimum_m':.04}}
    def test_approach_finishes_without_next_stage(self):
        t=GraspMotion(self.r,self.m,self.q,self.q,None,100);q=self.q.copy()
        for _ in range(222):
            nxt=t.step(q,.1);self.assertLessEqual(max(abs(a-b) for a,b in zip(q,nxt)),math.radians(.3)+1e-9);q=nxt
        self.assertTrue(t.done);self.assertEqual(t.stage,'approach');self.assertEqual(q[5],self.q[5])
    def test_wrong_order_and_expired_preview(self):
        with self.assertRaises(ValueError):GraspMotion(self.r,self.m,self.q,self.q,'lower',100)
        with self.assertRaises(ValueError):GraspMotion(self.r,self.m,self.q,self.q,None,161)
    def test_close_locks_arm_and_allows_contact_error(self):
        r=copy.deepcopy(self.r);r['stage']='close';r['reference_height_m']=None;r['plan']['goal']=self.q.copy();r['plan']['goal'][5]-=.1
        t=GraspMotion(r,self.m,self.q,self.q,'lower',100);t.command=r['plan']['goal'].copy();t.elapsed=22
        measured=t.command.copy();measured[5]+=math.radians(4);t.step(measured,.1);self.assertTrue(t.done)
        r['plan']['goal'][0]+=.01
        with self.assertRaises(ValueError):GraspMotion(r,self.m,self.q,self.q,'lower',100)
    def test_speed_clearance_jaw_changes_and_lag_rejected(self):
        for kind in ['speed','clearance','jaw','nan']:
            r=copy.deepcopy(self.r)
            if kind=='speed':r['plan']['duration']=4
            if kind=='clearance':r['clearance']['gripper_holder_path_minimum_m']=.02
            if kind=='jaw':r['plan']['goal'][5]+=.01
            if kind=='nan':r['plan']['goal'][0]=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):GraspMotion(r,self.m,self.q,self.q,None,100)
        t=GraspMotion(self.r,self.m,self.q,self.q,None,100);bad=self.q.copy();bad[0]+=.1
        with self.assertRaises(ValueError):t.step(bad,.1)
        with self.assertRaises(ValueError):t.step(self.q,.3)
