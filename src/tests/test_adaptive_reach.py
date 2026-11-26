import unittest
import numpy as np
import cv2
from adaptive_reach import (Observation,TargetWatch,observe_regions,planning_result_current,pause_acknowledged)
from command_mapping import Mapping
from control_worker import Controller
from tests.test_control import FakeBus


def observation(x=0.,angle=0.,colour=0):
    a=np.deg2rad(angle);r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    h=np.zeros((12,8,8),np.float32);h[colour,3,3]=1
    return Observation((np.array([x,0.,0.]),np.array([.025,.05,.001]),r),h)


class WatchTests(unittest.TestCase):
    def test_small_jitter_does_not_replan(self):
        w=TargetWatch(observation(),0.)
        for i in range(1,21):w.update([observation(.0005*(-1)**i)],i*.05)
        self.assertTrue(w.can_descend(1.));self.assertIsNone(w.action('approach',1.))

    def test_moving_target_pauses_then_needs_new_plan(self):
        w=TargetWatch(observation(),0.)
        moved=observation(.02)
        for i in range(1,20):w.update([moved],i*.05)
        self.assertEqual(w.action('approach',.95),'pause');self.assertFalse(w.can_descend(.95))
        self.assertTrue(w.accepts_plan(moved,.95));w.commit(moved)
        self.assertTrue(w.can_descend(.95))

    def test_rotating_box_and_reversed_axis(self):
        w=TargetWatch(observation(),0.);w.update([observation(angle=180)],.1)
        self.assertFalse(w.moved())
        w.update([observation(angle=20)],.2);self.assertEqual(w.action('approach',.2),'pause')

    def test_occluded_and_ambiguous_never_commit(self):
        w=TargetWatch(observation(),0.);w.update([],.1)
        self.assertFalse(w.can_descend(.1));self.assertIsNone(w.action('approach',.1))
        self.assertEqual(w.action('approach',.3),'pause')
        w.update([observation(.01),observation(-.01)],.4)
        self.assertEqual(w.problem,'ambiguous target');self.assertFalse(w.stable(.4))

    def test_does_not_switch_to_different_appearance_or_far_object(self):
        w=TargetWatch(observation(),0.);w.update([observation(.01,colour=4)],.3)
        self.assertEqual(w.action('approach',.3),'pause')
        w.update([observation(.3)],.4);self.assertIsNotNone(w.problem)

    def test_final_descent_stops_but_carried_object_is_not_retargeted(self):
        w=TargetWatch(observation(),0.);w.update([observation(.02)],.1)
        self.assertEqual(w.action('lower',.1),'stop')
        for stage in ('close','basket_lift','basket_transfer','basket_open','return_sitting'):
            self.assertIsNone(w.action(stage,.1))

    def test_moving_slowly_never_accumulates_stationary_time(self):
        w=TargetWatch(observation(),0.)
        for i in range(1,15):
            w.update([observation(i*.001)],i*.1)
            self.assertFalse(w.stable(i*.1))

    def test_old_planning_result_and_changed_command_rejected(self):
        old=observation();new=observation(.02);w=TargetWatch(old,0.)
        for i in range(1,20):w.update([new],i*.05)
        state={'state':'ACTIVE','command':[0.]*6}
        self.assertFalse(planning_result_current(state,[0.]*6,w,old,.95))
        self.assertTrue(planning_result_current(state,[0.]*6,w,new,.95))
        self.assertFalse(planning_result_current({**state,'state':'HOLD'},[0.]*6,w,new,.95))
        self.assertFalse(planning_result_current(state,[.01]*6,w,new,.95))
        self.assertFalse(planning_result_current(state,[0.]*6,w,new,2.))

    def test_histogram_follows_texture_and_contour_order(self):
        frame=np.zeros((100,160,3),np.uint8)
        c1=np.array([[10,10],[50,10],[50,70],[10,70]])
        c2=c1+[80,0];cv2.fillPoly(frame,[c1],(0,0,220));cv2.fillPoly(frame,[c2],(0,220,0))
        a,b=observe_regions(frame,[c1,c2],[observation().region,observation(.05).region])
        w=TargetWatch(a,0.);w.update([b,a],.1)
        self.assertIs(w.current,a)


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.bus=FakeBus(self.m);self.c=Controller(self.bus,self.m,direct=True)

    def test_worker_pause_keeps_exact_command_without_rearming(self):
        self.c.arm();start=list(self.c.trajectory.command);goal=start.copy();goal[0]+=.2
        self.c.handle({'op':'direct_segment','time':1,'plan':{'start':start,'goal':goal,'duration':10.},'segment_id':3},1)
        # Simulate command progression while encoder feedback is a little behind.
        command=start.copy();command[0]+=.04;self.c.trajectory.command=command.copy()
        calls=list(self.bus.calls)
        self.c.handle({'op':'direct_pause','time':1.1,'pause_id':4},1.1)
        self.assertEqual(self.c.target,command);self.assertEqual(self.c.state,'ACTIVE')
        self.assertIsNone(self.c.segment);self.assertIsNone(self.c.segment_id)
        self.assertEqual(self.bus.calls,calls);self.assertTrue(self.c.enabled)
        self.assertTrue(pause_acknowledged({'state':'ACTIVE','direct_pause_id':4,'segment_id':None},4))
        self.assertFalse(pause_acknowledged({'state':'ACTIVE','direct_pause_id':3,'segment_id':None},4))
        self.c.handle({'op':'direct_segment','time':1.2,'plan':{'start':command,'goal':goal,'duration':10.},'segment_id':0},1.2)
        self.assertIsNone(self.c.direct_pause_id)

    def test_pause_is_not_an_arm_command(self):
        with self.assertRaises(ValueError):self.c.handle({'op':'direct_pause','time':1,'pause_id':1},1)
        self.assertFalse(self.c.enabled);self.assertEqual(self.bus.calls,[])
        self.c.arm();self.c.hold('operator stop')
        with self.assertRaises(ValueError):self.c.handle({'op':'direct_pause','time':2,'pause_id':2},2)
        self.assertEqual(self.c.state,'HOLD')

    def test_user_stop_clears_pause_authorization(self):
        self.c.arm();self.c.handle({'op':'direct_pause','time':1,'pause_id':7},1)
        self.c.hold('user stop');self.assertIsNone(self.c.direct_pause_id)
        self.assertFalse(pause_acknowledged({'state':self.c.state,'direct_pause_id':7,'segment_id':None},7))
        self.c.release();self.assertFalse(self.c.enabled)

    def test_pause_rejects_wrong_mode_and_stale_message(self):
        c=Controller(self.bus,self.m);c.arm()
        with self.assertRaises(ValueError):c.handle({'op':'direct_pause','time':1,'pause_id':1},1)
        with self.assertRaises(ValueError):self.c.handle({'op':'direct_pause','time':1,'pause_id':1},2)


if __name__=='__main__':unittest.main()
