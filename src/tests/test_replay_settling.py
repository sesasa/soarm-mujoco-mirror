import unittest
import numpy as np
from replay_settling import endpoint_settled

class Tests(unittest.TestCase):
 def test_biscuit_contact_can_continue_through_carry(self):
  error=np.deg2rad([.45,.13,.50,1.09,.33,5.63])
  for stage in ('close','basket_lift','basket_transfer'):
   self.assertTrue(endpoint_settled(stage,error))
  self.assertFalse(endpoint_settled('basket_open',error))
 def test_contact_allowance_does_not_relax_arm_or_large_jaw_errors(self):
  self.assertFalse(endpoint_settled('close',np.deg2rad([0,3.1,0,0,0,5.63])))
  self.assertFalse(endpoint_settled('close',np.deg2rad([0,0,0,0,0,6.1])))
 def test_recorded_plate_exit_sag_can_continue(self):
  self.assertTrue(endpoint_settled('plate_exit',np.deg2rad([.41,3.04,.23,.18,.2,1.01])))
 def test_release_still_requires_tighter_position(self):
  self.assertFalse(endpoint_settled('basket_open',np.deg2rad([.41,3.04,.23,.18,.2,1.01])))
 def test_large_error_and_bad_feedback_do_not_continue(self):
  self.assertFalse(endpoint_settled('plate_exit',np.deg2rad([0,5,0,0,0,0])))
  self.assertFalse(endpoint_settled('plate_exit',[0,0,0,0,0,float('nan')]))
if __name__=='__main__':unittest.main()
