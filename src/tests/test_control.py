import copy
import math
import unittest
from command_mapping import Mapping,Trajectory,JOINTS
from control_worker import Controller

class FakeBus:
    def __init__(self,mapping):
        self.mapping=mapping
        self.raw=mapping.to_raw([0,0,-.096,-.079,-math.pi/2,.5])
        self.calls=[];self.torque=0
    def sync_read(self,name,normalize=False):
        if name=='Present_Position':return self.raw.copy()
        if name=='Operating_Mode':return dict.fromkeys(JOINTS,0)
        if name=='Torque_Enable':return dict.fromkeys(JOINTS,self.torque)
        raise AssertionError(name)
    def sync_write(self,name,values,normalize=False):
        assert name=='Goal_Position' and normalize is False
        self.calls.append(('write',values.copy()))
    def enable_torque(self):self.calls.append(('enable',));self.torque=1
    def disable_torque(self):self.calls.append(('disable',));self.torque=0

class Tests(unittest.TestCase):
    def setUp(self):self.mapping=Mapping();self.bus=FakeBus(self.mapping)
    def test_roundtrip(self):
        for fraction in [.01,.25,.5,.75,.99]:
            q=[lo+(hi-lo)*fraction for lo,hi in self.mapping.limits]
            got=self.mapping.from_raw(self.mapping.to_raw(q))
            self.assertLess(max(abs(a-b) for a,b in zip(q,got)),.001)
    def test_offsets_inverse(self):
        q=self.mapping.from_raw(self.bus.raw)
        self.assertLess(abs(q[4]+math.pi/2),.002)
        self.assertEqual(self.mapping.to_raw(q),self.bus.raw)
    def test_invalid_targets(self):
        for values in [[float('nan')]*6,[float('inf')]*6,[0]*5,[100]*6]:
            with self.assertRaises(ValueError):self.mapping.to_raw(values)
    def test_enable_order(self):
        c=Controller(self.bus,self.mapping);c.arm()
        self.assertEqual(self.bus.calls,[('write',self.bus.raw),('enable',)])
    def test_rate_and_excursion(self):
        q=self.mapping.from_raw(self.bus.raw);t=Trajectory(self.mapping,q)
        target=q.copy();target[0]+=.05
        out=t.step(target,q,.1)
        self.assertAlmostEqual(out[0]-q[0],math.radians(.5))
        target[0]=q[0]+math.radians(6)
        with self.assertRaises(ValueError):t.step(target,q,.1)
    def test_tracking_and_stale(self):
        q=self.mapping.from_raw(self.bus.raw);t=Trajectory(self.mapping,q)
        for dt in [0,1,float('nan')]:
            with self.assertRaises(ValueError):t.step(q,q,dt)
        bad=q.copy();bad[0]+=.2
        with self.assertRaises(ValueError):t.step(q,bad,.1)
    def test_hold_release(self):
        c=Controller(self.bus,self.mapping);c.arm();c.hold('test')
        self.assertEqual(c.state,'HOLD');self.assertTrue(c.enabled)
        c.release();self.assertEqual(c.state,'MONITOR');self.assertFalse(c.enabled)
        self.assertEqual(self.bus.calls[-1],('disable',))
    def test_stale_and_monitor_no_commands(self):
        c=Controller(self.bus,self.mapping)
        with self.assertRaises(ValueError):c.handle({'op':'arm','time':1},2)
        c.handle({'op':'target','time':1,'q':[0]*6},1.1)
        c.hold('test');c.release();self.assertEqual(self.bus.calls,[])
    def test_existing_torque_rejected(self):
        self.bus.torque=1;c=Controller(self.bus,self.mapping)
        with self.assertRaises(ValueError):c.arm()
        self.assertEqual(self.bus.calls,[])
    def test_worker_watchdog_and_eof_release(self):
        import subprocess,sys,tempfile,json,time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            audit=Path(folder)/'calls.json'
            code="from tests.test_control import FakeBus; from command_mapping import Mapping; from control_worker import run; import json,sys; m=Mapping(); b=FakeBus(m); run(b,m,2); open(sys.argv[1],'w').write(json.dumps(b.calls))"
            p=subprocess.Popen([sys.executable,'-u','-c',code,str(audit)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
            try:
                first=json.loads(p.stdout.readline())
                self.assertEqual(first['state'],'MONITOR')
                p.stdin.write(json.dumps({'op':'arm','time':time.monotonic()})+'\n');p.stdin.flush()
                states=[]
                for line in p.stdout:
                    record=json.loads(line);states.append(record['state'])
                    if record['state']=='HOLD':break
                self.assertIn('ACTIVE',states);self.assertIn('HOLD',states)
                p.stdin.close();p.stdout.read();p.wait(timeout=4)
                calls=json.loads(audit.read_text())
                self.assertEqual(calls[0][0],'write');self.assertEqual(calls[1][0],'enable')
                self.assertEqual(calls[-1][0],'disable')
            finally:
                if p.poll() is None:p.kill();p.wait()
                if not p.stdin.closed:p.stdin.close()
                p.stdout.close()

    def test_outside_limits_no_enable(self):
        self.bus.raw['shoulder_lift']=self.mapping.calibration['shoulder_lift']['range_min']
        c=Controller(self.bus,self.mapping)
        with self.assertRaises(ValueError):c.arm()
        self.assertEqual(self.bus.calls,[])

if __name__=='__main__':unittest.main()
