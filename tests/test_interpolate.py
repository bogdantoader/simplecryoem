import unittest
import numpy as np 
import sys, site

site.addsitedir('..')

from src.interpolate import *

class TestInterpolate(unittest.TestCase):

    def test_find_nearest_one_grid_point_idx(self):    
        x_freq = np.fft.fftfreq(10,2)
        y_freq = np.fft.fftfreq(5,0.1)
        z_freq = np.fft.fftfreq(6,1)

        #print(x_freq)
        #print(y_freq)
        #print(z_freq)

        pts = [np.array([0, 0, 0]),
                np.array([0.04, 0, 0]),
                np.array([-0.24, -2, 0.168]),
                np.array([-0.01, 4, -0.48])]
        pts_grid_idxs = [np.array([0, 0, 0]),
                np.array([1, 0, 0]),
                np.array([5, 4, 1]),
                np.array([0, 2, 3])]

        for p, pt_grid_idx in zip(pts, pts_grid_idxs):
            #print(p, pt_grid_idx)
            idx_found = find_nearest_one_grid_point_idx(p, x_freq, y_freq, z_freq)
            #print(idx_found)
            self.assertEqual(sum((idx_found-pt_grid_idx)**2), 0)

        return
           

    def test_interpolate_nn(self):
        x_freq = np.fft.fftfreq(10,2)
        y_freq = np.fft.fftfreq(5,0.1)
        z_freq = np.fft.fftfreq(6,1)
        
        vol = np.random.randn(len(x_freq), len(y_freq), len(z_freq))

        i_coords = np.array([[0, 0.04, -0.24, -0.01],
                             [0, 0, -2, 4],
                             [0, 0, 0.168, -0.48]])
            
        i_vol_correct = np.array([vol[0,0,0],
                    vol[1,0,0],
                    vol[5,4,1],
                    vol[0,2,3]])

        i_vol = interpolate(i_coords, x_freq, y_freq, z_freq, vol, "nn")

        self.assertEqual(sum((i_vol - i_vol_correct)**2), 0)

        return

if __name__ == '__main__':
    unittest.main()
