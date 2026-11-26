import json,unittest
from pathlib import Path
import numpy as np,mujoco
from real_hover import scene_matches,make_report
from scene_changes import Plane
from command_mapping import Mapping,ROOT

class HoverPlanningTests(unittest.TestCase):
    def test_scene_change_and_ambiguity(self):
        r=(np.zeros(3),np.array([.01,.05,.001]),np.eye(3))
        self.assertTrue(scene_matches([r],[r]))
        self.assertFalse(scene_matches([r],[]))
        self.assertFalse(scene_matches([r],[r,r]))
        self.assertFalse(scene_matches([r],[(np.array([.02,0,0]),r[1],r[2])]))
        self.assertFalse(scene_matches([r],[(r[0],r[1],np.array([[0,-1,0],[1,0,0],[0,0,1]]))]))
    def test_high_hover_respects_hardware_ranges(self):
        import xml.etree.ElementTree as ET
        from wrist_mount_visual import apply_wrist_mount_visual
        candidate=json.loads((Path(__file__).parent/'fixtures/high_hover_start.json').read_text())
        tree=ET.parse(ROOT/'assets/so101/so101.xml').getroot();tree.find('compiler').set('meshdir',str(ROOT/'assets/so101/assets'));apply_wrist_mount_visual(tree)
        model=mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding='unicode'))
        plane=Plane(np.eye(3),np.array(candidate['T_camera_from_base']),np.array(candidate['T_camera_from_board']))
        surface=np.array(candidate['surface']);q=candidate['q'];state={'command':q,'q':q}
        report=make_report(model,Mapping(),plane,state,(surface,np.ones(3),np.eye(3)))
        Mapping().check(report['plan']['goal'])
        self.assertEqual(report['plan']['goal'][5],q[5])
        self.assertGreater(report['clearance']['gripper_holder_path_minimum_m'],.08)
        self.assertFalse(report['grasp_enabled'])
        self.assertGreaterEqual(report['plan']['duration'],20)

    def test_contained_pen_fragments_do_not_imply_motion(self):
        original=(np.zeros(3),np.array([.01,.05,.001]),np.eye(3))
        fragments=[(np.array([0,y,0]),np.array([.01,.022,.001]),np.eye(3)) for y in [-.028,.028]]
        self.assertTrue(scene_matches([original],fragments))
        self.assertFalse(scene_matches([original],fragments[:1]))
        self.assertFalse(scene_matches([original],fragments+[(np.array([.1,0,0]),original[1],original[2])]))

    def test_selected_target_ignores_other_red_labels(self):
        import cv2
        from real_hover import selected_target_visible
        image=np.zeros((100,100,3),np.uint8)
        contours=[np.array([[10,10],[30,10],[30,60],[10,60]]),np.array([[60,10],[80,10],[80,60],[60,60]])]
        for c in contours:cv2.fillPoly(image,[c],(0,0,255))
        regions=[(np.array([x,0.,0.]),np.array([.01,.05,.001]),np.eye(3)) for x in [0.,.1]]
        report={'surface_center_base_m':[0.,0.,0.]}
        self.assertTrue(selected_target_visible(report,regions,contours,image))
        self.assertFalse(selected_target_visible(report,regions[1:],contours[1:],image))
        image[:,:40]=0
        self.assertFalse(selected_target_visible(report,regions,contours,image))
