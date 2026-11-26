import unittest
from pathlib import Path
import mujoco
import numpy as np
from simulated_reach import ReachSolver,trajectory,hover_point

class ReachTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2]/'assets/so101/so101.xml'))

    def test_known_reachable_pose_and_limits(self):
        solver=ReachSolver(self.model)
        target=solver.position(np.array([.2,-.5,.6,.4,-.8,.2]))
        plan=solver.solve(np.array([0,-.4,.4,.3,-.6,.2]),target)
        self.assertLess(np.linalg.norm(solver.position(plan['goal'])-target),.002)
        for t in np.linspace(0,plan['duration'],100):
            q=trajectory(plan,t)
            self.assertTrue(np.all(q>=solver.bounds[:,0]))
            self.assertTrue(np.all(q<=solver.bounds[:,1]))
            self.assertAlmostEqual(q[5],.2)
        np.testing.assert_allclose(trajectory(plan,0),plan['start'])
        np.testing.assert_allclose(trajectory(plan,plan['duration']),plan['goal'])
        times=np.linspace(0,plan['duration'],500)
        velocity=np.diff([trajectory(plan,t) for t in times],axis=0)/np.diff(times)[:,None]
        self.assertLessEqual(np.max(np.abs(velocity)),np.deg2rad(25)+1e-6)

    def test_unreachable_refused(self):
        with self.assertRaises(ValueError):ReachSolver(self.model).solve(np.zeros(6),[2,2,2])

    def test_nonfinite_refused(self):
        with self.assertRaises(ValueError):ReachSolver(self.model).solve(np.zeros(6),[np.nan,0,0])

    def test_hover_is_above_plane_with_downward_normal(self):
        normal=np.array([.1,0,-1.]);normal/=np.linalg.norm(normal)
        surface=np.array([.2,.1,-.01]);destination=hover_point(surface,normal)
        self.assertGreater(destination[2],surface[2])
        self.assertAlmostEqual(np.linalg.norm(destination-surface),.08)
        np.testing.assert_allclose(destination-surface,-.08*normal)

    def test_out_of_limit_start_is_reported(self):
        solver=ReachSolver(self.model)
        start=np.array([0,-1.8,.8,1.0,-1,.1])
        limited=np.clip(start,solver.bounds[:,0],solver.bounds[:,1])
        plan=solver.solve(start,solver.position(limited))
        self.assertGreater(plan['start_adjustment_deg'],3)

if __name__=='__main__':unittest.main()
