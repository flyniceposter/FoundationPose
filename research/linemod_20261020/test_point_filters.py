import unittest
import numpy as np
from point_filter_study import filter_mask
from point_audit import depth_erosion


class PointFilterTests(unittest.TestCase):
    def test_component(self):
        mask = np.zeros((20, 20), bool); mask[2:7, 2:7] = True; mask[15,15] = True
        selected = filter_mask(mask, 'largest_component')
        self.assertEqual(int(selected.sum()),25)
        self.assertTrue(mask[15,15])

    def test_depth_gate(self):
        mask = np.ones((10,10),bool); depth = np.ones((10,10),np.float32)
        depth[0,0]=0; depth[0,1]=np.nan; depth[0,2]=2
        selected = filter_mask(mask,'depth_half_diameter',depth,.2)
        self.assertEqual(int(selected.sum()),97)
        self.assertFalse(filter_mask(mask,'depth_diameter',np.zeros_like(depth),.2).any())
        self.assertEqual(int(filter_mask(mask,'raw').sum()),100)

    def test_existing_erosion_cpu(self):
        d = np.ones((9,9),np.float32); d[4,4]=2
        out = depth_erosion(d)
        self.assertEqual(out[4,4],0)
        self.assertEqual(out[0,0],1)
        self.assertTrue(np.array_equal(depth_erosion(np.zeros_like(d)),np.zeros_like(d)))


if __name__=='__main__':
    unittest.main()
