import unittest
import numpy as np
from gpt_targets import request_payload,parse_selection


def response(value,status='completed'):
 import json
 return {'status':status,'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps(value)}]}]}

class GPTSelectionTests(unittest.TestCase):
 def test_valid_id(self):
  self.assertEqual(parse_selection(response({'status':'selected','candidate_id':2,'description':'matchbox'}),3),(1,'matchbox'))
 def test_invalid_and_ambiguous(self):
  for value in [0,4,True,None,'2']:
   with self.assertRaises(ValueError):parse_selection(response({'status':'selected','candidate_id':value}),3)
  for status in ['not_found','ambiguous']:
   with self.assertRaises(ValueError):parse_selection(response({'status':status,'candidate_id':1}),3)
 def test_incomplete_and_refusal(self):
  with self.assertRaises(ValueError):parse_selection(response({},'incomplete'),3)
  with self.assertRaises(ValueError):parse_selection({'status':'completed','output':[{'type':'message','content':[{'type':'refusal'}]}]},3)
 def test_payload(self):
  frame=np.zeros((480,640,3),np.uint8);cs=[np.int32([[10,30],[50,30],[50,80],[10,80]])]
  p,_=request_payload(frame,cs,'matchbox')
  self.assertFalse(p['store']);self.assertTrue(p['text']['format']['strict'])
  self.assertEqual(p['input'][0]['content'][1]['type'],'input_image')
  with self.assertRaises(ValueError):request_payload(frame,[],'box')

if __name__=='__main__':unittest.main()
