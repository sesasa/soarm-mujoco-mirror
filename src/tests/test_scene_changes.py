import unittest
import cv2
import numpy as np
from scene_changes import Changes, Plane, refine_current_contours, merge_contained_fragments

class SceneChangesTest(unittest.TestCase):
    def setUp(self):
        self.image=np.full((160,200,3),150,np.uint8)
        self.mask=np.full((160,200),255,np.uint8)
        self.detector=Changes(self.mask)
        self.detector.reset(self.image)

    def update(self, frame, excluded=None):
        for _ in range(8):boxes,status=self.detector.update(frame,.1,excluded)
        return boxes,status

    def test_addition_persists_and_reset_clears(self):
        frame=self.image.copy();frame[50:80,60:100]=30
        self.assertEqual(self.detector.update(frame,.1)[0],[])
        self.assertEqual(len(self.update(frame)[0]),1)
        self.detector.reset(frame)
        self.assertEqual(self.update(frame)[0],[])

    def test_uniform_brightness(self):
        self.assertEqual(self.update(self.image+20)[0],[])

    def test_move_marks_both_locations(self):
        first=self.image.copy();first[30:50,30:60]=20
        self.detector.reset(first)
        second=self.image.copy();second[90:110,120:150]=20
        self.assertEqual(len(self.update(second)[0]),2)

    def test_excluded_robot(self):
        frame=self.image.copy();frame[50:80,60:100]=20
        excluded=np.zeros_like(self.mask);excluded[45:85,55:105]=255
        self.assertEqual(self.update(frame,excluded)[0],[])

    def test_plane_mapping(self):
        K=np.array([[400.,0,320],[0,410,240],[0,0,1]])
        T=np.eye(4);T[:3,:3]=cv2.Rodrigues(np.array([.12,-.08,.2]))[0];T[:3,3]=[.1,-.2,.5]
        base=np.eye(4);base[:3,3]=[.2,.1,.03]
        plane=Plane(K,base,T)
        points=np.array([[-.1,.1],[.02,-.03],[.12,.1]],np.float64)
        pixels=cv2.perspectiveTransform(points[None],plane.H)[0]
        np.testing.assert_allclose(plane.board_xy(pixels),points,atol=1e-12)
        center,size,rot=plane.footprint([[250,180],[280,180],[280,200],[250,200]])
        cam=base@np.r_[center,1]
        board=np.linalg.inv(T)@cam
        self.assertAlmostEqual(board[2],0,places=12)
        self.assertTrue((size>0).all())
        np.testing.assert_allclose(rot.T@rot,np.eye(3),atol=1e-12)

    def test_current_appearance_rejects_shadow_and_vacated_position(self):
        frame=np.full((160,200,3),180,np.uint8)
        # A broad weak shadow around a narrow pen; candidate also includes old space.
        frame[35:125,70:110]=155
        frame[45:115,82:96]=(100,40,20)
        candidate=np.array([[65,30],[120,30],[120,130],[65,130]],np.int32)
        empty=np.array([[15,35],[40,35],[40,115],[15,115]],np.int32)
        refined=refine_current_contours(frame,[candidate,candidate.copy(),empty])
        self.assertEqual(len(refined),1)
        x,y,w,h=cv2.boundingRect(refined[0])
        self.assertLessEqual(w,16)
        self.assertGreaterEqual(h,68)
        self.assertGreaterEqual(x,80)

    def test_interior_fragment_merges_without_widening(self):
        # Concave outer pen silhouette and an isolated interior label patch.
        outer=np.array([[20,10],[40,10],[40,90],[20,90],[20,65],[35,65],[35,35],[20,35]],np.int32)
        label=np.array([[23,40],[30,40],[30,60],[23,60]],np.int32)
        nearby=np.array([[44,10],[54,10],[54,90],[44,90]],np.int32)
        groups=merge_contained_fragments([label,nearby,outer])
        self.assertEqual(len(groups),2)
        self.assertEqual(sorted(cv2.boundingRect(c) for c in groups),[(20,10,21,81),(44,10,11,81)])

    def test_end_to_end_objects_stay_separate(self):
        first=np.array([[20,10],[35,10],[35,50],[20,50]],np.int32)
        second=first+np.array([0,43])
        self.assertEqual(len(merge_contained_fragments([first,second])),2)

    def test_recorded_adjacent_pens_separate(self):
        from pathlib import Path
        frame=cv2.imread(str(Path(__file__).resolve().parents[2]/'captures/scene_changes/adjacent_pens.png'))
        self.assertIsNotNone(frame)
        # Both objects inside one change candidate, as in the live failure.
        candidate=np.array([[300,155],[380,155],[380,310],[300,310]],np.int32)
        for seed in (0,1,2):
            cv2.setRNGSeed(seed)
            contours=refine_current_contours(frame,[candidate])
            self.assertEqual(len(contours),2)
            sizes=[sorted(cv2.minAreaRect(c)[1]) for c in contours]
            self.assertTrue(all(80<long<140 and short<40 for short,long in sizes))

    def test_recorded_adjacent_tip_fragment(self):
        import json
        from pathlib import Path
        path=Path(__file__).resolve().parents[2]/'captures/scene_changes/adjacent_fragments.json'
        data=json.loads(path.read_text())
        contours=[np.array(c,np.int32) for c in data['contours_undistorted_px']]
        self.assertEqual(len(contours),3)
        result=merge_contained_fragments(contours)
        self.assertEqual(len(result),2)
        self.assertTrue(all(min(cv2.minAreaRect(c)[1])<40 for c in result))

    def test_rotated_metric_footprint(self):
        K=np.array([[400.,0,320],[0,410,240],[0,0,1]])
        T=np.eye(4);T[:3,:3]=cv2.Rodrigues(np.array([.3,-.2,.1]))[0];T[:3,3]=[.1,-.1,.6]
        plane=Plane(K,np.eye(4),T)
        xy=cv2.boxPoints(((.05,.02),(.10,.018),37)).astype(np.float64)
        pixels=cv2.perspectiveTransform(xy[None],plane.H)[0]
        region=plane.footprint(pixels)
        np.testing.assert_allclose(sorted(region[1][:2]*2),[.018,.10],atol=1e-7)
        projected=plane.projected_footprint(region)
        for point in pixels:
            self.assertLess(np.linalg.norm(projected-point,axis=1).min(),1e-4)
        # The rotated footprint is substantially smaller than its axis-aligned box.
        self.assertLess(np.prod(region[1][:2]*2),np.prod(np.ptp(xy,axis=0))*.6)

if __name__=='__main__':unittest.main()
