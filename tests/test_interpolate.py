import unittest
import numpy as np 
import jax.numpy as jnp
import sys, site
from jax.config import config

config.update("jax_enable_x64", True)
site.addsitedir('..')

from src.interpolate import *
from numpy.testing import assert_array_equal, assert_equal

class TestInterpolate(unittest.TestCase):

    def test_find_nearest_one_grid_point_idx(self):   
        # Tested here for both even and odd number of grid points.
        x_freq = jnp.fft.fftfreq(10,2)
        y_freq = jnp.fft.fftfreq(5,0.1)
        z_freq = jnp.fft.fftfreq(6,1)

        # The grid spacing and length of the above grids.
        x_grid = jnp.array([x_freq[1], len(x_freq)])
        y_grid = jnp.array([y_freq[1], len(y_freq)])
        z_grid = jnp.array([z_freq[1], len(z_freq)])

        pts = [jnp.array([0, 0, 0]),
                jnp.array([0.04, 0, 0]),
                jnp.array([-0.24, -2, 0.168]),
                jnp.array([-0.01, 4, -0.48]),
                jnp.array([-0.04, 0, 0])]
        pts_grid_idxs = [np.array([0, 0, 0]),
                #jnp.array([1, 0, 0]),
                #jnp.array([5, 4, 1]),
                #jnp.array([0, 2, 3])]
                jnp.array([0, 1, 0]),
                jnp.array([4, 5, 1]),
                jnp.array([2, 0, 3]),
                jnp.array([0, 9, 0])]
            # x and y indices swapped

        for p, pt_grid_idx in zip(pts, pts_grid_idxs):
            idx_found = find_nearest_one_grid_point_idx(p, x_grid, y_grid, z_grid)
            assert_array_equal(idx_found, pt_grid_idx)

        return
           
    def test_find_nearest_one_grid_point_idx_bug(self):
        coord = jnp.array([0.282842712e+00, 2.22044605e-16, 0.00000000e+00])
        x_freq = jnp.array([ 0.,  0.1,  0.2, -0.2, -0.1])
        y_freq = x_freq
        z_freq = x_freq
        target_idx = jnp.array([0, 3, 0])

        # The grid spacing and length of the above grids.
        x_grid = jnp.array([x_freq[1], len(x_freq)])
        y_grid = jnp.array([y_freq[1], len(y_freq)])
        z_grid = jnp.array([z_freq[1], len(z_freq)])

        found_idx = find_nearest_one_grid_point_idx(coord,x_grid,y_grid,z_grid)
        assert_array_equal(found_idx, target_idx)
        return

    def test_interpolate_nn(self):
        x_freq = jnp.fft.fftfreq(10,2)
        y_freq = jnp.fft.fftfreq(5,0.1)
        z_freq = jnp.fft.fftfreq(6,1)

        # The grid spacing and length of the above grids.
        x_grid = jnp.array([x_freq[1], len(x_freq)])
        y_grid = jnp.array([y_freq[1], len(y_freq)])
        z_grid = jnp.array([z_freq[1], len(z_freq)])
        
        vol = jnp.array(np.random.randn(len(y_freq), len(x_freq), len(z_freq)))

        i_coords = jnp.array([[0, 0.04, -0.24, -0.01],
                             [0, 0, -2, 4],
                             [0, 0, 0.168, -0.48]])
            
        i_vol_correct = jnp.array([vol[0,0,0],
                    vol[0,1,0],
                    vol[4,5,1],
                    vol[2,0,3]])

        i_vol = interpolate(i_coords, x_grid, y_grid, z_grid, vol, "nn")

        assert_array_equal(i_vol, i_vol_correct)
        return

    def test_interpolate_tri(self):
         
        vol, coords_vals, grids = self.get_data_for_tri_tests()

        x_grid, y_grid , z_grid = grids 

        coords = [coord for coord, val in coords_vals]
        vals = [val for coord, val in coords_vals]

        i_coords = jnp.array(coords).T

        assert_array_equal(interpolate(i_coords, x_grid, y_grid, z_grid, vol, "tri"), vals)

        return

    def test_tri_interp_point(self):
        xyz = jnp.array([[0,10],[0,100],[0,1000]])
        xyz_idx = jnp.array([[0,1],[0,1],[0,1]])

        vol, coords_vals, _ = self.get_data_for_tri_tests()
           
        for coord, val in coords_vals:
            self.assertEqual(tri_interp_point(jnp.array(coord),vol,(xyz,xyz_idx)),val)

        return

    def get_data_for_tri_tests(self):
        """This function returns what is required to test trilinear
        interpolation functions: the grid (x_freq, y_freq, z_freq),
        the volume defined on the grid, the coordinates of the points to
        interpolate at and the correct values of the interpolation.
        """

        # The grids
        x_freq = jnp.array([0,10,20,-20,10])
        y_freq = jnp.array([0,100,200,-200,100])
        z_freq = jnp.array([0,1000,2000,-2000,1000])

        x_grid = jnp.array([x_freq[1], len(x_freq)])
        y_grid = jnp.array([y_freq[1], len(y_freq)])
        z_grid = jnp.array([z_freq[1], len(z_freq)])

        grids = (x_grid, y_grid, z_grid)

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
        vol = jnp.array(vol) 

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

        return vol, coords_vals, grids

    def test_find_nearest_eight_grid_points_idx(self):
        # This should also work for even number of points, as long as 
        # find_adjacent_grid_points_idx does (tested below).

        # The grid spacing and length of each frequency grid. NOT jax objects
        x_grid = np.array([1, 7])
        y_grid = x_grid
        z_grid = x_grid

        # The spacing and lengths correspond to the following grids: 
        # x_freq = [0, 1, 2, -2, -1]
        # y_freq = x_freq
        # z_freq = x_freq
    
        coords = jnp.array([1.5, 1.5, 1.5])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[1,2],[1,2]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[1,2],[1,2]]))

        coords = jnp.array([1.5, 1.5, 2.5])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[1,2],[2,3]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[1,2],[2,3]]))

        coords = jnp.array([1.5, 1.5, 3.5])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[1,2],[3,4]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[1,2],[3,4]]))

        coords = jnp.array([1.5, 1.5, 4.5])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[1,2],[4,5]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[1,2],[4,5]]))

        coords = jnp.array([1.5, -0.9, 0.2])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[6,7],[0,1]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[6,0],[0,1]]))

        coords = jnp.array([1.5, -1.9, 0.2])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[5,6],[0,1]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[5,6],[0,1]]))

        coords = jnp.array([1.5, -2.9, 0.2])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[1,2],[4, 5],[0,1]]))
        assert_array_equal(xyz_idx, jnp.array([[1,2],[4,5],[0,1]]))

        coords = jnp.array([2, -0.6, 0.2])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[2,3],[6, 7],[0,1]]))
        assert_array_equal(xyz_idx, jnp.array([[2,3],[6,0],[0,1]]))

        coords = jnp.array([5.7, 8.2, -12.3])
        coords, (xyz, xyz_idx) = find_nearest_eight_grid_points_idx(coords, 
            x_grid, y_grid, z_grid)
        assert_array_equal(xyz, jnp.array([[5, 6],[1,2],[1,2]]))
        assert_array_equal(xyz_idx, jnp.array([[5,6],[1,2],[1,2]]))

        return

    def test_find_adjacent_grid_points_idx(self):
        # Odd number of points
        grid, pts_adjacent_nearest = self.get_data_adjacent_and_nearest_grid_point_idx_odd()

        for test_pt, adj_idx, _ in pts_adjacent_nearest:
            assert_equal(find_adjacent_grid_points_idx(test_pt, grid[1], len(grid)),
                    (jnp.array(adj_idx[0]), jnp.array(adj_idx[1])))

        # Even number of points
        grid, pts_adjacent_nearest = self.get_data_adjacent_and_nearest_grid_point_idx_even()

        for test_pt, adj_idx, _ in pts_adjacent_nearest:
            assert_equal(find_adjacent_grid_points_idx(test_pt, grid[1], len(grid)),
                    (jnp.array(adj_idx[0]), jnp.array(adj_idx[1])))
        return

    def test_find_nearest_grid_point_idx(self):
        # Odd number of points
        grid, pts_adjacent_nearest = self.get_data_adjacent_and_nearest_grid_point_idx_odd()

        for test_pt, _, nearest_idx in pts_adjacent_nearest:
            assert_equal(
                find_nearest_grid_point_idx(test_pt, grid[1], len(grid)), 
                jnp.array(nearest_idx))
       
        # Even number of points
        grid, pts_adjacent_nearest = self.get_data_adjacent_and_nearest_grid_point_idx_even()

        for test_pt, _, nearest_idx in pts_adjacent_nearest:
            assert_equal(
                find_nearest_grid_point_idx(test_pt, grid[1], len(grid)), 
                jnp.array(nearest_idx))
        return

        return

    def get_data_adjacent_and_nearest_grid_point_idx_even(self):
        grid = jnp.array([0., 0.25, -0.5, -0.25])

        eps = 1e-18

        # List of tuples containing (test_pt, adjacent_point_idx, closest_idx)
        pts_adjacent_nearest = [
            # Well between grid points, positive
            (0.1, (0,1), 0),
            (0.4, (1,2), 2),
            (0.6, (2,3), 2),
            (0.85, (3,0), 3),
            (0.9, (3,0), 0),
            (1.05, (0,1), 0),
            #  Well between grid points, negative
            (-0.1, (3,0), 0),
            (-0.2, (3,0), 3),
            (-0.3, (2,3), 3),
            (-0.43, (2,3), 2),
            (-0.57, (1,2), 2),
            (-0.7, (1,2), 1),
            (-0.87, (0,1), 1),
            (-0.94, (0,1), 0),
            # On the grid points
            (0, (0,1), 0),
            (0.25, (1,2), 1),
            (0.5, (2,3), 2),
            (0.75, (3,0), 3),
            (1, (0,1), 0),
            (-0.25, (3,0), 3),
            (-0.5, (2,3), 2),
            (-0.75, (1,2), 1),
            (-1, (0,1), 0),
            # Epislon on the left of the grid points
            # e.g. due to floating point errors - we consider these
            # points to be on the grid and want to consistently select
            # the grid point itself and the one to its right.
            (0-eps, (0,1), 0),
            (0.25-eps, (1,2), 1),
            (0.5-eps, (2,3), 2),
            (0.75-eps, (3,0), 3),
            (1-eps, (0,1), 0),
            (-0.25-eps, (3,0), 3),
            (-0.5-eps, (2,3), 2),
            (-0.75-eps, (1,2), 1),
            (-1-eps, (0,1), 0),
            # Mid-distance between grid points, plus/minus epislon
            # We want to consistently select the left point as the closest.
            (0.125, (0,1), 0),
            (0.375, (1,2), 1),
            (0.625, (2,3), 2),
            (0.875, (3,0), 3),
            (1.125, (0,1), 0),
            (-0.125, (3,0), 3),
            (-0.375, (2,3), 2),
            (-0.625, (1,2), 1),
            (-0.875, (0,1), 0),
            (-0.125, (3,0), 3),
            (0.125+eps, (0,1), 0),
            (0.375+eps, (1,2), 1),
            (0.625+eps, (2,3), 2),
            (0.875+eps, (3,0), 3),
            (1.125+eps, (0,1), 0),
            (-0.125+eps, (3,0), 3),
            (-0.375+eps, (2,3), 2),
            (-0.625+eps, (1,2), 1),
            (-0.875+eps, (0,1), 0),
            (-0.125+eps, (3,0), 3)
        ]
        return grid, pts_adjacent_nearest


    def get_data_adjacent_and_nearest_grid_point_idx_odd(self):
        grid = jnp.array([0, 0.2, 0.4, -0.4, -0.2])

        eps = 1e-18

        # List of tuples containing (test_pt, adjacent_point_idx, closest_idx)
        pts_adjacent_nearest = [
            # Well between grid points, positive
            (0.02, (0,1), 0),
            (0.25, (1,2), 1),
            (0.48, (2,3), 2),
            (0.62, (3,4), 3),
            (0.79, (3,4), 4),
            (0.89, (4,0), 4),
            #  Well between grid points, negative
            (-0.1, (4,0), 4),
            (-0.25, (3,4), 4),
            (-0.38, (3,4), 3),
            (-0.43, (2,3), 3),
            (-0.57, (2,3), 2),
            (-0.62, (1,2), 2),
            (-0.87, (0,1), 1),
            (-0.94, (0,1), 0),
            # On the grid points
            (0, (0,1), 0),
            (0.2, (1,2), 1),
            (0.4, (2,3), 2),
            (0.6, (3,4), 3),
            (0.8, (4,0), 4),
            (-0.2, (4,0), 4),
            (-0.4, (3,4), 3),
            (-0.6, (2,3), 2),
            (-0.8, (1,2), 1),
            # Epislon on the left of the grid points
            # e.g. due to floating point errors - we consider these
            # points to be on the grid and want to consistently select
            # the grid point itself and the one to its right
            (0-eps, (0,1), 0),
            (0.2-eps, (1,2), 1),
            (0.4-eps, (2,3), 2),
            (0.6-eps, (3,4), 3),
            (0.8-eps, (4,0), 4),
            (1-eps, (0,1), 0),
            (1.2-eps, (1,2), 1),
            (-0.2-eps, (4,0), 4),
            (-0.4-eps, (3,4), 3),
            (-0.6-eps, (2,3), 2),
            (-0.8-eps, (1,2), 1),
            (-1-eps, (0,1), 0),
            # Mid-distance between grid points, plus/minus epislon
            # We want to consistently select the left point as the closest.
            (0.1, (0,1), 0),
            (0.3, (1,2), 1),
            (0.5, (2,3), 2),
            (0.7, (3,4), 3),
            (0.9, (4,0), 4),
            (1.1, (0,1), 0),
            (1.3, (1,2), 1),
            (-0.1, (4,0), 4),
            (-0.3, (3,4), 3),
            (-0.5, (2,3), 2),
            (-0.7, (1,2), 1),
            (-0.9, (0,1), 0),
            (-1.1, (4,0), 4),
            (-1.3, (3,4), 3),
            (0.1+eps, (0,1), 0),
            (0.3+eps, (1,2), 1),
            (0.5+eps, (2,3), 2),
            (0.7+eps, (3,4), 3),
            (0.9+eps, (4,0), 4),
            (1.1+eps, (0,1), 0),
            (1.3+eps, (1,2), 1),
            (-0.1+eps, (4,0), 4),
            (-0.3+eps, (3,4), 3),
            (-0.5+eps, (2,3), 2),
            (-0.7+eps, (1,2), 1),
            (-0.9+eps, (0,1), 0),
            (-1.1+eps, (4,0), 4),
            (-1.3+eps, (3,4), 3)
        ]
        return grid, pts_adjacent_nearest

if __name__ == '__main__':
    unittest.main()
