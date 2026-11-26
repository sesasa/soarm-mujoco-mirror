import unittest,json
from pathlib import Path
import numpy as np
import mujoco
from simulated_reach import ReachSolver,vertical_step
from table_clearance import TableClearance

class VerticalStepTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[2]
        fixture=json.loads((root/'src/tests/fixtures/descent_start.json').read_text())
        self.plan=fixture['plan'];plane=np.array(fixture['plane'])
        model=mujoco.MjModel.from_xml_path(str(root/'assets/so101/so101.xml'))
        self.solver=ReachSolver(model)
        self.checker=TableClearance(model,plane[:3,3],plane[:3,2])

    def test_step_down_and_up_continuity(self):
        down=vertical_step(self.solver,self.plan,-.01,self.checker)
        np.testing.assert_allclose(down['start'],self.plan['goal'],atol=2e-6)
        self.assertAlmostEqual(down['height_m'],.07)
        self.assertLess(down['vertical_path_checks']['maximum_lateral_error_m'],.002)
        self.assertGreater(down['table_clearance']['gripper_holder_path_minimum_m'],.01)
        up=vertical_step(self.solver,down,.01,self.checker)
        self.assertAlmostEqual(up['height_m'],.08)
        np.testing.assert_allclose(up['start'],down['goal'],atol=2e-6)
        self.assertAlmostEqual(up['goal'][5],down['goal'][5])

    def test_floor_and_ceiling(self):
        with self.assertRaises(ValueError):vertical_step(self.solver,self.plan,.01,self.checker)
        plan=self.plan
        for _ in range(5):plan=vertical_step(self.solver,plan,-.01,self.checker)
        self.assertAlmostEqual(plan['height_m'],.03)
        with self.assertRaises(ValueError):vertical_step(self.solver,plan,-.01,self.checker)

    def test_insufficient_clearance_refused_without_modifying_previous(self):
        class Blocked:
            def path(self,plan):return {'path_minimum_m':.003,'gripper_holder_path_minimum_m':.02}
        previous=json.dumps(self.plan)
        with self.assertRaises(ValueError):vertical_step(self.solver,self.plan,-.01,Blocked())
        self.assertEqual(json.dumps(self.plan),previous)

    def test_wrong_mode_or_step_size(self):
        with self.assertRaises(ValueError):vertical_step(self.solver,{'mode':'position_only'},-.01,self.checker)
        with self.assertRaises(ValueError):vertical_step(self.solver,self.plan,-.03,self.checker)

if __name__=='__main__':unittest.main()
