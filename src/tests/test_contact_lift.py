import json,os,unittest
from pathlib import Path
import mujoco
from contact_preview import build,run,OUT

class ContactLiftTest(unittest.TestCase):
    def test_object_is_free_not_attached(self):
        source=json.loads((Path(__file__).parent/'fixtures/contact_start.json').read_text())
        model,_=build(source)
        self.assertEqual(model.joint('object_free').type[0],mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(model.neq,0)

    @unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('MUJOCO_GL') in ('egl','osmesa'),'Rendering context required')
    def test_lift_and_sustained_airborne_support(self):
        result=run(headless=True,source_path=Path(__file__).parent/'fixtures/contact_start.json',lift=True)
        self.assertTrue(result['lift_attempted'])
        self.assertTrue(result['lift_passed'])
        self.assertFalse(result['hardware_commands'])
        self.assertFalse(result['unexpected_contact'])
        self.assertGreater(result['final_object_bottom_clearance_m'],.01)
        self.assertGreater(result['airborne_hold_s'],1.9)
        self.assertLess(result['final_forces_N']['table'],.01)
        self.assertGreater(result['final_forces_N']['fixed_finger'],.02)
        self.assertGreater(result['final_forces_N']['moving_finger'],.02)
        trace=json.loads((OUT/'lift_trace.json').read_text())
        airborne=[r for r in trace if r['lift_elapsed_s'] is not None and r['lift_elapsed_s']>=4.]
        self.assertGreater(len(airborne),90)
        self.assertTrue(all(r['object_bottom_clearance_m']>.01 and r['bilateral'] and r['forces_N']['table']<.01 for r in airborne))

if __name__=='__main__':unittest.main()
