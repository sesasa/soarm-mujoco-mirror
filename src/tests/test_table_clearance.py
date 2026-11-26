import unittest
import mujoco
import numpy as np
from simulated_reach import JOINTS
from table_clearance import TableClearance

class TableClearanceTest(unittest.TestCase):
    def model(self,mesh=False):
        asset='<asset><mesh name="tetra" vertex="0 0 0 .02 0 0 0 .02 0 0 0 .02"/></asset>' if mesh else ''
        geom='<geom type="mesh" mesh="tetra" group="2"/>' if mesh else '<geom type="box" size=".01 .02 .03" group="2"/>'
        joints=''.join(f'<joint name="{name}" type="slide" axis="0 0 1" armature=".001"/>' for name in JOINTS)
        return mujoco.MjModel.from_xml_string(f'<mujoco>{asset}<worldbody><geom type="box" size=".1 .1 .1"/><body name="gripper" pos="0 0 .1">{joints}{geom}<geom type="sphere" size=".2" group="5" mass="0"/></body></worldbody></mujoco>')

    def test_box_distance_fixed_and_hidden_geometry_excluded(self):
        checker=TableClearance(self.model(),[0,0,0],[0,0,-1])
        result=checker.at(np.zeros(6))
        self.assertAlmostEqual(result['minimum_m'],.07)
        self.assertAlmostEqual(result['gripper_holder_minimum_m'],.07)
        self.assertEqual(result['body'],'gripper')

    def test_compiled_mesh_coordinates(self):
        checker=TableClearance(self.model(mesh=True),[0,0,0],[0,0,1])
        self.assertAlmostEqual(checker.at(np.zeros(6))['minimum_m'],.1,places=6)

    def test_sampled_intersection_and_endpoint(self):
        checker=TableClearance(self.model(),[0,0,0],[0,0,1])
        plan={'start':np.zeros(6),'goal':np.array([-.09,0,0,0,0,0]),'duration':2.}
        result=checker.path(plan,samples=11)
        self.assertAlmostEqual(result['path_minimum_m'],-.02)
        self.assertEqual(result['status'],'sampled intersection')
        self.assertEqual(result['minimum_at_fraction'],1.)
        self.assertFalse(result['collision_validated'])

if __name__=='__main__':unittest.main()
