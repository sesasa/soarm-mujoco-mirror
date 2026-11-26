import copy,json,math,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from command_mapping import Mapping
from control_worker import Controller
from tests.test_control import FakeBus

class HoverWorkerTests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.b=FakeBus(self.m);self.c=Controller(self.b,self.m,allow_hover=True);self.c.arm()
        q=self.c.target.copy();goal=q.copy();goal[1]+=.2
        self.r={'mode':'position_only_high_hover','height_above_table_m':.12,'joint_mapping':self.m.mapping,'prepared_at_unix':time.time(),'plan':{'start':q,'goal':goal,'duration':10},'clearance':{'path_minimum_m':.009,'gripper_holder_path_minimum_m':.1,'samples':161}}
    def start(self):self.c.handle({'op':'hover','time':time.monotonic(),'report':self.r},time.monotonic())
    def test_requires_opt_in_and_arm(self):
        for c in [Controller(self.b,self.m),Controller(self.b,self.m,True)]:
            with self.assertRaises(ValueError):c.handle({'op':'hover','time':time.monotonic(),'report':self.r},time.monotonic())
    def test_single_use_manual_targets_ignored_hold_cancels(self):
        self.start();self.assertEqual(self.c.state,'HOVER')
        old=self.c.target.copy();self.c.handle({'op':'target','time':time.monotonic(),'q':[0]*6},time.monotonic());self.assertEqual(old,self.c.target)
        self.c.hold('stop');self.assertIsNone(self.c.hover);self.assertEqual(self.c.state,'HOLD')
        self.c.arm()
        with self.assertRaises(ValueError):self.start()
    def test_release_cancels(self):
        self.start();self.c.release();self.assertIsNone(self.c.hover);self.assertEqual(self.b.torque,0)
    def test_watchdog_stops_hover_and_eof_releases(self):
        code='from tests.test_control import FakeBus; from command_mapping import Mapping; from control_worker import run; m=Mapping(); b=FakeBus(m); run(b,m,4,allow_hover=True); print("FINAL_TORQUE",b.torque)'
        p=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            json.loads(p.stdout.readline())
            def send(op,**kw):p.stdin.write(json.dumps({'op':op,'time':time.monotonic(),**kw})+'\n');p.stdin.flush()
            send('arm')
            while json.loads(p.stdout.readline())['state']!='ACTIVE':pass
            self.r['prepared_at_unix']=time.time();send('hover',report=self.r)
            states=[]
            while True:
                r=json.loads(p.stdout.readline());states.append(r['state'])
                if r['state']=='HOLD':break
            self.assertIn('HOVER',states);self.assertIn('heartbeat',r['reason'])
            p.stdin.close();out=p.stdout.read();p.wait(timeout=4);self.assertIn('FINAL_TORQUE 0',out)
        finally:
            if p.poll() is None:p.kill();p.wait()
            if not p.stdin.closed:p.stdin.close()
            p.stdout.close()

if __name__=='__main__':unittest.main()
