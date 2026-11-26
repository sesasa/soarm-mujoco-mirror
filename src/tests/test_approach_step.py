import unittest
from unittest.mock import Mock
import numpy as np
from approach_step import prepare_step
from command_mapping import Mapping

class ApproachStepTest(unittest.TestCase):
    def setUp(self):
        self.q=np.array([0,0,0,0,-1,.4]);self.mapping=Mapping()
        self.report={'plan':{'start':self.q.tolist(),'goal':(self.q+np.array([.1,.8,-.2,-.1,.02,.3])).tolist()}}
        self.clearance=Mock();self.clearance.path.return_value={'path_minimum_m':.009,'gripper_holder_path_minimum_m':.10}
    def test_bounded_step_and_unchanged_gripper(self):
        p=prepare_step(self.report,self.q,self.mapping,self.clearance)
        self.assertLessEqual(np.max(np.abs(p['goal']-self.q)),np.deg2rad(3)+1e-12)
        self.assertEqual(p['goal'][5],self.q[5])
        self.assertLessEqual(1.875*np.max(np.abs(p['goal']-self.q))/p['duration'],np.deg2rad(1)+1e-12)
    def test_changed_pose_refused(self):
        q=self.q.copy();q[0]+=.02
        with self.assertRaises(ValueError):prepare_step(self.report,q,self.mapping,self.clearance)
    def test_clearance_refused(self):
        self.clearance.path.return_value['gripper_holder_path_minimum_m']=.01
        with self.assertRaises(ValueError):prepare_step(self.report,self.q,self.mapping,self.clearance)
    def test_invalid_plan_refused(self):
        self.report['plan']['goal'][0]=float('nan')
        with self.assertRaises(ValueError):prepare_step(self.report,self.q,self.mapping,self.clearance)

    def test_playback_validation(self):
        from approach_step import validate_step
        plan=prepare_step(self.report,self.q,self.mapping,self.clearance)
        validate_step(plan,self.q,self.mapping.limits)
        for field,value in [('duration',.1),('goal',self.q+1),('start',[0]*5)]:
            broken={**plan,field:value}
            with self.assertRaises(ValueError):validate_step(broken,self.q,self.mapping.limits)
        limits=np.array(self.mapping.limits);limits[1,1]=.001
        with self.assertRaises(ValueError):validate_step(plan,self.q,limits)
