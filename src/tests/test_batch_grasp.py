import unittest
import numpy as np
from batch_grasp import table_targets

class BatchTests(unittest.TestCase):
 def region(self,width,length):return (np.zeros(3),np.array([width/2,length/2,.001]),np.eye(3))
 def test_selects_both_sizes_and_excludes_plate_contents(self):
  plate={'inner_rim_pixels':[[200,100],[350,100],[350,250],[200,250]]}
  packet=np.array([[30,30],[80,30],[80,130],[30,130]])
  box=np.array([[100,40],[140,40],[140,100],[100,100]])
  deposited=packet+[200,90]
  candidates=table_targets([packet,box,deposited],[self.region(.054,.105),self.region(.043,.061),self.region(.054,.105)],plate)
  self.assertEqual([c['key'] for c in candidates],['packet','box'])
 def test_full_preview_keeps_second_leg_out_of_initial_execution(self):
  from unittest.mock import patch
  from batch_grasp import prepare_batch
  packet={'key':'packet','region':self.region(.054,.105),'width_m':.054}
  box={'key':'box','region':self.region(.043,.061),'width_m':.043,'contour':np.array([[100,40],[140,40],[140,100],[100,100]])}
  calls=[]
  def build(model,mapping,plane,state,region,**kwargs):
   calls.append((state,kwargs))
   return [{'stage':'return_sitting' if kwargs['return_to_sitting'] else 'plate_exit','plan':{'start':state['command'],'goal':[len(calls)]*6}}]
  with patch('batch_grasp.measure_basket',return_value={}),patch('batch_grasp.table_targets',return_value=[packet,box]),patch('batch_grasp.build_sequence',side_effect=build):
   answer=prepare_batch(None,None,None,{'command':[0]*6},None,[],[],None,[])
  self.assertEqual(len(answer['plans']),1)
  self.assertEqual([r['preview_target'] for r in answer['preview_plans']],['packet','box'])
  self.assertEqual(calls[1][0]['command'],answer['plans'][-1]['plan']['goal'])
  self.assertFalse(calls[0][1]['return_to_sitting'])
  self.assertTrue(calls[1][1]['return_to_sitting'])
  self.assertAlmostEqual(calls[1][1]['target_profile_override']['closing_gap_m'],.039)
  calls.clear()
  with patch('batch_grasp.measure_basket',return_value={}),patch('batch_grasp.table_targets',return_value=[packet,box]),patch('batch_grasp.build_sequence',side_effect=build):
   refreshed=prepare_batch(None,None,None,{'command':[0]*6},None,[],[],None,[],preview_remaining=False)
  self.assertEqual(len(calls),1)
  self.assertEqual(len(refreshed['preview_plans']),len(refreshed['plans']))
  self.assertFalse(calls[0][1]['return_to_sitting'])
 def test_partial_plate_overlap_not_reselected(self):
  plate={'inner_rim_pixels':[[100,100],[200,100],[200,200],[100,200]]}
  contour=np.array([[80,110],[140,110],[140,170],[80,170]])
  self.assertEqual(table_targets([contour],[self.region(.04,.06)],plate),[])
if __name__=='__main__':unittest.main()
