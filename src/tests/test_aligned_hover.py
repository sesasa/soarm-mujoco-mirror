import unittest
from pathlib import Path
import mujoco
import numpy as np
from simulated_reach import ReachSolver

class AlignedHoverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2]/'assets/so101/so101.xml'))

    def test_reachable_orientation_and_pen_axis_symmetry(self):
        solver=ReachSolver(self.model)
        reference=np.array([-.93,.32,-.37,1.65,.8,.2])
        target=solver.position(reference)
        orientation=solver.data.xmat[self.model.body('gripper').id].reshape(3,3).copy()
        up,along=orientation[:,2],orientation[:,1]
        for sign in (1,-1):
            plan=solver.solve_grasp(reference+[.05,-.05,.05,-.05,.05,0],target,up,sign*along)
            self.assertLess(plan['error_m'],.002)
            self.assertLess(plan['downward_tilt_deg'],1)
            self.assertLess(plan['jaw_alignment_error_deg'],1)
            solver.position(plan['goal'])
            actual=solver.data.xmat[self.model.body('gripper').id].reshape(3,3)
            # Jaw closing direction is across the pen, not along it.
            self.assertLess(abs(actual[:,0]@along),.02)
            self.assertGreater((-actual[:,2])@(-up),.999)
            self.assertAlmostEqual(plan['goal'][5],reference[5])
            self.assertTrue(np.all(plan['goal']>=solver.bounds[:,0]))
            self.assertTrue(np.all(plan['goal']<=solver.bounds[:,1]))

    def test_invalid_axes_and_unreachable_pose(self):
        solver=ReachSolver(self.model)
        for target,up,along in [([.2,.1,.1],[0,0,1],[0,0,1]),
                                ([.2,.1,.1],[0,0,0],[1,0,0]),
                                ([2,2,2],[0,0,1],[1,0,0])]:
            with self.assertRaises(ValueError):solver.solve_grasp(np.zeros(6),target,up,along)

if __name__=='__main__':unittest.main()
