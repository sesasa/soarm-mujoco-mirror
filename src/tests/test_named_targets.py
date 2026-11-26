import unittest
import numpy as np
from adaptive_reach import Observation
from named_targets import NamedTargets


def obs(x=0.,color=0):
    hist=np.zeros((12,8,8),np.float32);hist[color,0,0]=1
    return Observation((np.array([x,0.,0.]),np.array([.02,.05,.02]),np.eye(3)),hist)

class NamedTests(unittest.TestCase):
    def test_commands_use_updated_position(self):
        names=NamedTargets();names.label('Highlighter',obs(),0)

        for t in np.arange(.1,.81,.1):names.update([obs(.02)],float(t))
        for command in ['highlighter','reach for the highlighter','Please reach for the highlighter.','go to highlighter']:
            name,watch=names.resolve(command,.8)
            self.assertEqual(name,'highlighter');self.assertAlmostEqual(watch.current.center[0],.02)
    def test_unknown_missing_ambiguous(self):
        names=NamedTargets();names.label('matchbox',obs(),0)
        with self.assertRaises(ValueError):names.resolve('highlighter',0)
        names.update([],1)
        with self.assertRaises(ValueError):names.resolve('matchbox',1)
        names.update([obs(),obs(.01)],2)
        with self.assertRaises(ValueError):names.resolve('matchbox',2)
    def test_two_distinct_objects(self):
        names=NamedTargets();names.label('highlighter',obs(),0);names.label('matchbox',obs(.05,5),0)

        for t in np.arange(.1,1.01,.1):names.update([obs(.05,5),obs()],float(t))
        self.assertEqual(names.resolve('matchbox',1)[1].current.histogram[5,0,0],1)
    def test_stale(self):
        names=NamedTargets();names.label('box',obs(),0)
        with self.assertRaises(ValueError):names.resolve('box',1)

if __name__=='__main__':unittest.main()
