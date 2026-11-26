import json,os,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from contact_cycle import SetDownCycle
from contact_preview import run,OUT

class SetDownTests(unittest.TestCase):
    def setUp(self):
        self.q=np.zeros(6)
        self.patch=patch('contact_cycle.make_lift_waypoints',return_value=[self.q.copy(),self.q.copy()])
        self.patch.start();self.addCleanup(self.patch.stop)
        self.c=SetDownCycle(None,self.q,np.array([0,0,1]),np.array([0,1,0]),-.1,.3,0.)
    def observe(self,t,dt=.1,table=.3,finger=0.,bottom=0.,speed=0.):
        self.c.observe(t,dt,self.q,{'table':table,'fixed_finger':finger,'moving_finger':finger},bottom,speed)
    def test_no_support_prevents_release_and_times_out(self):
        self.observe(5.1,table=0.)
        self.assertTrue(self.c.failed)
        self.assertEqual(self.c.stage,'lower')
        self.assertIsNone(self.c.release_started_at)
    def test_support_must_be_continuous(self):
        self.observe(.1);self.observe(.2,table=0.);self.observe(.3);self.observe(.4)
        self.assertEqual(self.c.stage,'lower')
        self.observe(.5)
        self.assertEqual(self.c.stage,'release')
        self.assertTrue(self.c.supported_before_release)
    def test_finger_contact_prevents_withdrawal(self):
        self.observe(.3,dt=.3)
        self.observe(2.5,dt=2.2,finger=.1)
        self.assertEqual(self.c.stage,'release')
        self.observe(5.4,finger=.1)
        self.assertTrue(self.c.failed)
    def test_completion_needs_settled_independent_support(self):
        self.observe(.3,dt=.3);self.observe(2.4,dt=.3)
        self.assertEqual(self.c.stage,'withdraw')
        self.observe(6.5,dt=.6);self.observe(6.6,speed=.1)
        self.assertEqual(self.c.final_support_hold,0.)
        self.assertFalse(self.c.done)
        self.observe(7.2,dt=.6);self.observe(7.7,dt=.5)
        self.assertTrue(self.c.done)

class FullCycleTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('MUJOCO_GL') in ('egl','osmesa'),'Rendering context required')
    def test_free_object_pickup_release_and_withdrawal(self):
        result=run(headless=True,source_path=Path(__file__).parent/'fixtures/contact_start.json',cycle=True)
        self.assertTrue(result['cycle_passed']);self.assertTrue(result['lift_passed'])
        self.assertTrue(result['supported_before_release'])
        self.assertFalse(result['hardware_commands']);self.assertFalse(result['unexpected_contact'])
        self.assertGreaterEqual(result['final_table_support_hold_s'],1.)
        self.assertLess(abs(result['final_object_bottom_clearance_m']),.0015)
        self.assertGreater(result['final_forces_N']['table'],.1)
        for finger in ('fixed_finger','moving_finger'):
            self.assertLess(result['final_forces_N'][finger],.02)
        trace=json.loads((OUT/'cycle_trace.json').read_text())
        self.assertTrue({'lower','release','withdraw'}.issubset({r['cycle_stage'] for r in trace}))
        release=next(r for r in trace if r['cycle_stage']=='release')
        self.assertGreater(release['forces_N']['table'],.1)

if __name__=='__main__':unittest.main()
