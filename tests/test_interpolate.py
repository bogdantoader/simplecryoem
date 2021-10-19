import unittest
import numpy as np 
import sys, site

site.addsitedir('..')

from src.interpolate import *
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

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
                #np.array([1, 0, 0]),
                #np.array([5, 4, 1]),
                #np.array([0, 2, 3])]
                np.array([0, 1, 0]),
                np.array([4, 5, 1]),
                np.array([2, 0, 3])]
            # x and y indices swapped

        for p, pt_grid_idx in zip(pts, pts_grid_idxs):
            #print(p, pt_grid_idx)
            idx_found = find_nearest_one_grid_point_idx(p, x_freq, y_freq, z_freq)
            #print(idx_found)
            assert_array_equal(idx_found, pt_grid_idx)

        return
           

    def test_interpolate_nn(self):
        x_freq = np.fft.fftfreq(10,2)
        y_freq = np.fft.fftfreq(5,0.1)
        z_freq = np.fft.fftfreq(6,1)
        
        vol = np.random.randn(len(y_freq), len(x_freq), len(z_freq))

        i_coords = np.array([[0, 0.04, -0.24, -0.01],
                             [0, 0, -2, 4],
                             [0, 0, 0.168, -0.48]])
            
        i_vol_correct = np.array([vol[0,0,0],
                    vol[0,1,0],
                    vol[4,5,1],
                    vol[2,0,3]])

        i_vol = interpolate(i_coords, x_freq, y_freq, z_freq, vol, "nn")

        assert_array_equal(i_vol, i_vol_correct)
        return

    def test_interpolate_tri(self):
         
        vol, coords_vals, freqs = self.get_data_for_tri_tests()

        x_freq, y_freq, z_freq = freqs

        coords = [coord for coord, val in coords_vals]
        vals = [val for coord, val in coords_vals]

        i_coords = np.array(coords).T

        assert_array_equal(interpolate(i_coords, x_freq, y_freq, z_freq, vol,
            "tri"), vals)
       

        return

    def test_tri_interp_point(self):
        xyz = np.array([[0,10],[0,100],[0,1000]])
        xyz_idx = np.array([[0,1],[0,1],[0,1]])

        vol, coords_vals, _ = self.get_data_for_tri_tests()
           
        for coord, val in coords_vals:
            self.assertEqual(tri_interp_point(np.array(coord),vol,(xyz,xyz_idx)),val)

        return

    def get_data_for_tri_tests(self):
        """This function returns what is required to test trilinear
        interpolation functions: the grid (x_freq, y_freq, z_freq),
        the volume defined on the grid, the coordinates of the points to
        interpolate at and the correct values of the interpolation.
        """

        # The grids
        x_freq = np.array([0,10,20,-20,10])
        y_freq = np.array([0,100,200,-200,100])
        z_freq = np.array([0,1000,2000,-2000,1000])

        freqs = (x_freq, y_freq, z_freq)

        # The volume
        # Note that tri_interp_point assumes that x and y axes are swapped,
        # i.e. in vol[i,j], the index i changes y and the index j changes x
        # positions, so that it works with the coordinates given by meshgrid.
        # Hence, to obtain vol(x,y), we need to call vol[y_idx, x_idx]
        vol = np.zeros([5,5,5])
        vol[0,0,0] = 1
        vol[1,0,0] = 4
        vol[1,1,0] = 3
        vol[0,1,0] = 2
        vol[0,0,1] = 5
        vol[1,0,1] = 8
        vol[1,1,1] = 7
        vol[0,1,1] = 6
       
        # The interpolation coordinates and the target values
        coords_vals = []

        # First the corners of the cube
        coords_vals += [([0,0,0],1),
            ([0,0,1000],5),
            ([0,100,0],4),
            ([0,100,1000],8),
            ([10,0,0],2),
            ([10,0,1000],6),
            ([10,100,0],3),
            ([10,100,1000],7)]

        # Interpolate on edges 
        coords_vals += [([5,0,0],1.5),
            ([0,50,0],2.5),
            ([10,50,0],2.5),
            ([5,100,0],3.5),
            ([5,0,1000],5.5),
            ([0,50,1000],6.5),
            ([10,50,1000],6.5),
            ([5,100,1000],7.5),
            ([0,0,500],3),
            ([10,0,500],4),
            ([10,100,500],5),
            ([0,100,500],6)]

        # Interpolate on cube faces
        coords_vals += [([5,0,500],3.5),
            ([10,50,500],4.5),
            ([5,100,500],5.5),
            ([0,50,500],4.5),
            ([5,50,0],2.5),
            ([5,50,1000],6.5)]

        # Interpolate in the interior
        coords_vals += [([5,50,500],4.5)]

        return vol, coords_vals, freqs

    def test_find_nearest_eight_grid_points_idx(self):
        x_freq = [0, 1, 2, -2, -1]
        y_freq = x_freq
        z_freq = x_freq
    
        coords = np.array([1.5, 1.5, 1.5])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[1,2],[1,2]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[1,2],[1,2]]))

        coords = np.array([1.5, 1.5, 2.5])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[1,2],[2,-2]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[1,2],[2,3]]))

        coords = np.array([1.5, 1.5, 3.5])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[1,2],[-2,-1]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[1,2],[3,4]]))

        coords = np.array([1.5, 1.5, 4.5])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[1,2],[-1,0]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[1,2],[4,0]]))

        coords = np.array([1.5, -0.9, 0.2])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[-1,0],[0,1]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[4,0],[0,1]]))

        coords = np.array([1.5, -1.9, 0.2])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[-2,-1],[0,1]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[3,4],[0,1]]))

        coords = np.array([1.5, -2.9, 0.2])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[1,2],[2,-2],[0,1]]))
        assert_array_equal(xyz_idx, np.array([[1,2],[2,3],[0,1]]))

        coords = np.array([2, -0.6, 0.2])
        xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, 
            x_freq, y_freq, z_freq)
        assert_array_equal(xyz, np.array([[2,-2],[-1,0],[0,1]]))
        assert_array_equal(xyz_idx, np.array([[2,3],[4,0],[0,1]]))

        return

    def test_find_adjacent_grid_points_idx(self):
        grid = np.array([0, 0.2, 0.4, -0.4, -0.2])

        assert_equal(find_adjacent_grid_points_idx(0.02, grid),(0,1))
        assert_equal(find_adjacent_grid_points_idx(0.25, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(0.48, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(0.62, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(0.79, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(0.89, grid),(4,0))
        
        assert_equal(find_adjacent_grid_points_idx(-0.1, grid),(4,0))
        assert_equal(find_adjacent_grid_points_idx(-0.25, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(-0.38, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(-0.43, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(-0.57, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(-0.62, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(-0.87, grid),(0,1))
        assert_equal(find_adjacent_grid_points_idx(-0.94, grid),(0,1))

        assert_equal(find_adjacent_grid_points_idx(0, grid),(0,1))
        assert_equal(find_adjacent_grid_points_idx(0.2, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(0.4, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(0.6, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(0.8, grid),(4,0))
        assert_equal(find_adjacent_grid_points_idx(-0.2, grid),(4,0))
        assert_equal(find_adjacent_grid_points_idx(-0.4, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(-0.6, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(-0.8, grid),(1,2))
       
        eps = 1e-18
        assert_equal(find_adjacent_grid_points_idx(0-eps, grid),(0,1))
        assert_equal(find_adjacent_grid_points_idx(0.2-eps, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(0.4-eps, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(0.6-eps, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(0.8-eps, grid),(4,0))
        assert_equal(find_adjacent_grid_points_idx(1-eps, grid),(0,1))
        assert_equal(find_adjacent_grid_points_idx(1.2-eps, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(-0.2-eps, grid),(4,0))
        assert_equal(find_adjacent_grid_points_idx(-0.4-eps, grid),(3,4))
        assert_equal(find_adjacent_grid_points_idx(-0.6-eps, grid),(2,3))
        assert_equal(find_adjacent_grid_points_idx(-0.8-eps, grid),(1,2))
        assert_equal(find_adjacent_grid_points_idx(-1-eps, grid),(0,1))

        return

if __name__ == '__main__':
    unittest.main()
