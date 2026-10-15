import importlib
import unittest

import numpy as np


class SourceAnchorTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('geometry_ablation')
        except ModuleNotFoundError:
            self.fail('source-anchor experimental helper is missing')
        self.guide = np.full((134,2),np.nan)
        self.guide[[0,1,2,5,8,11,9,10]] = [(5,2),(5,5),(4,6),(6,6),(4,10),(6,10),(4,14),(4,18)]
        self.source = np.full((133,2),np.nan)
        self.source[[5,6,11,12]] = [(-10,10),(10,10),(-8,22),(8,22)]
        self.scores = np.ones(133)

    def test_uses_shoulders_not_neck_and_places_hips_at_source_anchor(self):
        result, meta = self.module.fit_to_source_anchor(self.guide,self.source,self.scores)
        self.assertAlmostEqual(meta['scale_multiplier'],3)
        np.testing.assert_allclose(result[[8,11]].mean(0),(0,22))
        np.testing.assert_allclose(result[[2,5]].mean(0),(0,10))
        np.testing.assert_allclose(result[1],(0,7))

    def test_preserves_relative_pose_missing_points_and_off_canvas_coordinates(self):
        result, _ = self.module.fit_to_source_anchor(self.guide,self.source,self.scores)
        np.testing.assert_allclose(result[9]-result[8],3*(self.guide[9]-self.guide[8]))
        self.assertTrue(np.isnan(result[50]).all())
        self.assertGreater(result[10,1],32)
        self.assertTrue(np.isnan(self.guide[50]).all())

    def test_rejects_unreliable_or_missing_source_anchor(self):
        self.scores[11]=0.29
        with self.assertRaises(ValueError):
            self.module.fit_to_source_anchor(self.guide,self.source,self.scores)
        self.scores[11]=1
        self.source[11]=np.nan
        with self.assertRaises(ValueError):
            self.module.fit_to_source_anchor(self.guide,self.source,self.scores)

    def test_raster_clips_crossing_bone_without_moving_its_endpoint(self):
        points=np.full((134,2),np.nan)
        points[1]=(15,8);points[8]=(15,44)
        image=self.module.render_clipped_guide(points,(32,32))
        self.assertNotEqual(image.getpixel((15,31)),(0,0,0))
        self.assertEqual(image.getpixel((0,31)),(0,0,0))
        np.testing.assert_allclose(points[8],(15,44))


if __name__=='__main__':unittest.main()
