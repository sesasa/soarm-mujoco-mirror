import unittest
from pick_place_command import parse_pick_place_command

class Tests(unittest.TestCase):
 def test_explicit_execution_and_target(self):
  for text,target in [('pick and place the highlighter','highlighter'),('Please pick and place matchbox in the plate.','matchbox'),('PICK AND PLACE the red box on the plate','red box')]:
   self.assertEqual(parse_pick_place_command(text),(True,target))
 def test_bare_name_and_noncommands_do_not_execute(self):
  for text in ['highlighter','reach for the highlighter','do not pick and place highlighter','can you pick and place highlighter?']:
   self.assertEqual(parse_pick_place_command(text),(False,text))
 def test_empty_target(self):
  for text in ['pick and place','pick and place the']:
   with self.assertRaises(ValueError):parse_pick_place_command(text)

if __name__=='__main__':unittest.main()
