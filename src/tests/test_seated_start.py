import json,math,unittest
from command_mapping import Mapping,ROOT
from seated_start import SeatedStart
from tests.test_control import FakeBus
from control_worker import Controller
class Tests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.p=json.loads((ROOT/'calibration/seated_start.json').read_text());self.q=self.p['start']
    def test_inward_rate_and_finish(self):
        t=SeatedStart(self.m,self.q,self.p);q=self.q
        for _ in range(165):
            nextq=t.step(q,.1)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(q,nextq)),math.radians(.1))
            t.mapping.to_raw(nextq);q=nextq
        self.assertTrue(t.done);self.m.check(q)
    def test_bad_pose_and_tracking(self):
        q=self.q.copy();q[0]+=.1
        with self.assertRaises(ValueError):SeatedStart(self.m,q,self.p)
        t=SeatedStart(self.m,self.q,self.p)
        with self.assertRaises(ValueError):t.step(q,.1)
    def test_enable_and_stop(self):
        t=SeatedStart(self.m,self.q,self.p);b=FakeBus(self.m);b.raw=t.mapping.to_raw(self.q)
        c=Controller(b,self.m,allow_grasp=True);c.handle({'op':'seated_start','time':1},1)
        self.assertEqual(c.state,'RISING');self.assertEqual(b.calls[:2],[('write',b.raw),('enable',)])
        c.hold('stop');self.assertEqual(c.state,'HOLD');c.release();self.assertFalse(b.torque)
    def test_opt_in(self):
        b=FakeBus(self.m);c=Controller(b,self.m)
        with self.assertRaises(ValueError):c.handle({'op':'seated_start','time':1},1)
        self.assertEqual(b.calls,[])
if __name__=='__main__':unittest.main()
