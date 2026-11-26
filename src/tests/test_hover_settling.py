import unittest
import numpy as np
from live_grip import hover_settled

class Tests(unittest.TestCase):
 def test_recorded_sag_can_finish(self):
  delta=np.deg2rad([.19,.32,1.38,-.09,.1,0])
  self.assertTrue(hover_settled(.0063,delta,np.zeros(6)))
 def test_excess_position_or_joint_error_still_refused(self):
  self.assertFalse(hover_settled(.009,np.zeros(6),np.zeros(6)))
  q=np.zeros(6);q[2]=np.deg2rad(4)
  self.assertFalse(hover_settled(.006,q,np.zeros(6)))
 def test_nonfinite_or_negative_refused(self):
  for error in [float('nan'),float('inf'),-.001]:
   self.assertFalse(hover_settled(error,np.zeros(6),np.zeros(6)))

if __name__=='__main__':unittest.main()
