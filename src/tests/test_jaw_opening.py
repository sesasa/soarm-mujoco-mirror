import unittest,json
from pathlib import Path
import mujoco
import numpy as np
from simulated_reach import ReachSolver,open_for_object,fingertip_gap,trajectory,vertical_step
from table_clearance import TableClearance

class JawOpeningTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]
        fixture=json.loads((root/'src/tests/fixtures/descent_start.json').read_text())
        self.previous=fixture['plan'];self.previous['measured_object']={'width_m':.027,'height_m':.017}
        model=mujoco.MjModel.from_xml_path(str(root/'assets/so101/so101.xml'))
        self.solver=ReachSolver(model);plane=np.array(fixture['plane'])
        self.clearance=TableClearance(model,plane[:3,3],plane[:3,2])

    def test_opens_to_measured_width_plus_margin_without_moving_arm(self):
        original=json.dumps(self.previous)
        plan=open_for_object(self.solver,self.previous,self.clearance)
        self.assertAlmostEqual(fingertip_gap(self.solver,plan['goal']),.037,places=8)
        self.assertGreater(plan['goal'][5],plan['start'][5])
        gaps=[]
        for t in np.linspace(0,plan['duration'],30):
            q=trajectory(plan,t)
            np.testing.assert_allclose(q[:5],self.previous['goal'][:5])
            gaps.append(fingertip_gap(self.solver,q))
        self.assertTrue(np.all(np.diff(gaps)>=-1e-10))
        self.assertEqual(json.dumps(self.previous),original)
        self.assertFalse(plan['jaw_opening']['physical_gap_validated'])

    def test_already_open_does_not_close(self):
        self.previous['goal'][5]=.8
        plan=open_for_object(self.solver,self.previous,self.clearance)
        self.assertAlmostEqual(plan['goal'][5],.8)

    def test_unmeasured_and_too_wide_refused(self):
        previous=dict(self.previous);previous.pop('measured_object')
        with self.assertRaises(ValueError):open_for_object(self.solver,previous,self.clearance)
        self.previous['measured_object']['width_m']=.2
        with self.assertRaises(ValueError):open_for_object(self.solver,self.previous,self.clearance)

    def test_clearance_refused(self):
        class Low:
            def path(self,plan):return {'path_minimum_m':.008,'gripper_holder_path_minimum_m':.019}
        with self.assertRaises(ValueError):open_for_object(self.solver,self.previous,Low())

    def test_descent_keeps_opening_and_measured_object(self):
        opened=open_for_object(self.solver,self.previous,self.clearance)
        lowered=vertical_step(self.solver,opened,-.01,self.clearance)
        self.assertAlmostEqual(lowered['goal'][5],opened['goal'][5])
        self.assertAlmostEqual(fingertip_gap(self.solver,lowered['goal']),.037,places=8)
        self.assertEqual(lowered['measured_object']['width_m'],.027)
        self.assertIn('jaw_opening',lowered)

if __name__=='__main__':unittest.main()
