import time,unittest
from command_mapping import Mapping
from control_worker import Controller
from tests.test_control import FakeBus
class GraspWorkerTests(unittest.TestCase):
    def setUp(self):
        self.m=Mapping();self.b=FakeBus(self.m);self.c=Controller(self.b,self.m,allow_grasp=True);self.c.arm();q=self.c.target.copy();goal=q.copy();goal[1]+=.2
        self.r={'mode':'supervised_grasp','stage':'approach','joint_mapping':self.m.mapping,'prepared_at_unix':time.time(),'reference_height_m':.04,'plan':{'start':q,'goal':goal,'duration':10},'clearance':{'samples':161,'path_minimum_m':.009,'gripper_holder_path_minimum_m':.04}}
    def start(self):self.c.handle({'op':'grasp','time':time.monotonic(),'report':self.r},time.monotonic())
    def test_explicit_mode_and_single_use(self):
        c=Controller(self.b,self.m)
        with self.assertRaises(ValueError):c.handle({'op':'grasp','time':time.monotonic(),'report':self.r},time.monotonic())
        self.start();self.assertEqual(self.c.state,'GRASP');self.c.hold('stop')
        self.assertIsNone(self.c.grasp)
        with self.assertRaises(ValueError):self.start()
    def test_stop_preserves_existing_grip_but_release_does_not(self):
        self.c.grasp_stage='close';self.c.grip_goal_raw=self.b.raw['gripper']-10;self.c.hold('stop')
        self.assertEqual(self.b.calls[-1][1]['gripper'],self.c.grip_goal_raw)
        self.c.release();self.assertEqual(self.b.torque,0);self.assertIsNone(self.c.grasp_stage)
    def test_cannot_rearm_during_grasp(self):
        self.start()
        with self.assertRaises(ValueError):self.c.arm()
    def test_modes_mutually_exclusive(self):
        with self.assertRaises(ValueError):Controller(self.b,self.m,allow_hover=True,allow_grasp=True)
    def test_watchdog_stops_grasp_and_eof_releases(self):
        import json,subprocess,sys
        code='from tests.test_control import FakeBus; from command_mapping import Mapping; from control_worker import run; m=Mapping(); b=FakeBus(m); run(b,m,4,allow_grasp=True); print("FINAL_TORQUE",b.torque)'
        p=subprocess.Popen([sys.executable,'-u','-c',code],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            json.loads(p.stdout.readline())
            def send(op,**kw):p.stdin.write(json.dumps({'op':op,'time':time.monotonic(),**kw})+'\n');p.stdin.flush()
            send('arm')
            while json.loads(p.stdout.readline())['state']!='ACTIVE':pass
            self.r['prepared_at_unix']=time.time();send('grasp',report=self.r)
            states=[]
            while True:
                r=json.loads(p.stdout.readline());states.append(r['state'])
                if r['state']=='HOLD':break
            self.assertIn('GRASP',states);self.assertIn('heartbeat',r['reason'])
            self.assertIsNone(r['grasp_stage'])
            p.stdin.close();out=p.stdout.read();p.wait(timeout=4);self.assertIn('FINAL_TORQUE 0',out)
        finally:
            if p.poll() is None:p.kill();p.wait()
            if not p.stdin.closed:p.stdin.close()
            p.stdout.close()

if __name__=='__main__':unittest.main()
