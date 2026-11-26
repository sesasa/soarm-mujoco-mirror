"""Regression: custom marker segids must not retain uninitialized memory."""
import os
import unittest
import numpy as np
from preview_camera_alignment import Preview

@unittest.skipUnless(os.environ.get('DISPLAY') or os.environ.get('MUJOCO_GL') in ('egl','osmesa'),'Rendering context required')
class PreviewRenderingTest(unittest.TestCase):
    def test_alternating_windows_and_new_marker_slots(self):
        previews=[]
        try:
            previews=[Preview(),Preview()]
            frame=np.zeros((480,640,3),np.uint8)
            region=(np.array([.15,.15,0]),np.array([.04,.01,.001]),np.eye(3))
            for n in range(8):
                preview=previews[n%2]
                # Reproduce the stale value observed in the crash deterministically.
                # update_scene resets model geoms; custom slots must be reset by us.
                for geom in preview.renderer.scene.geoms[:100]:geom.segid=336860180
                result=preview.draw(frame,{},regions=[region]*(n%3),show_marker=False,
                                    target=np.array([.15,.15,.08]) if n%3 else None,
                                    q_override=np.array([0,-.5,.5,.3,-1.,.2]))
                self.assertEqual(result.shape,(480,1920,3))
                plane=np.eye(4);plane[2,3]=-.02
                side=preview.draw_side(plane,np.array([.15,.15,.08]),[region])
                self.assertEqual(side.shape,(480,640,3))
                # Visual-only geometry must never index the segmentation lookup.
                for geom in preview.renderer.scene.geoms[:preview.renderer.scene.ngeom]:
                    if geom.objid==-1:self.assertEqual(geom.segid,-1)
                self.assertFalse(preview.renderer._segmentation_rendering)
        finally:
            for preview in previews:preview.close()

if __name__=='__main__':unittest.main()
