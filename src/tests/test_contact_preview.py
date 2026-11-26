import json,os,unittest
from pathlib import Path
from contact_preview import build,run,OUT

class ContactPreviewTest(unittest.TestCase):
    def test_missing_measured_opening_refused(self):
        with self.assertRaises(ValueError):build({})

    @unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('MUJOCO_GL') in ('egl','osmesa'),'Rendering context required')
    def test_free_object_establishes_both_finger_contacts_without_lift(self):
        source=Path(__file__).parent/'fixtures/contact_start.json'
        result=run(headless=True,source_path=source)
        self.assertTrue(result['bilateral_contact_established'])
        self.assertFalse(result['unexpected_contact'])
        self.assertFalse(result['hardware_commands'])
        self.assertFalse(result['lift_attempted'])
        self.assertGreater(result['final_forces_N']['fixed_finger'],.02)
        self.assertGreater(result['final_forces_N']['moving_finger'],.02)
        self.assertGreater(result['final_forces_N']['table'],.02)
        self.assertLess(result['object_displacement_m'],.02)
        trace=json.loads((OUT/'contact_trace.json').read_text())
        bilateral=[row for row in trace if row['bilateral']]
        self.assertGreater(len(bilateral),12)
        self.assertTrue(all(row['forces_N']['fixed_finger']>.02 and row['forces_N']['moving_finger']>.02 for row in bilateral))
        self.assertTrue(any(row['forces_N']['moving_finger']>.02 and not row['bilateral'] for row in trace))

if __name__=='__main__':unittest.main()
