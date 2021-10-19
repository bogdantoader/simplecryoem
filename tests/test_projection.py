import unittest
import sys, site

site.addsitedir('..')

import numpy as np
from src.projection import *
from numpy.testing import assert_array_almost_equal


class TestProjection(unittest.TestCase):
    # Point mass on a small Fourier domain
    # Dimensions are odd numbers for now

    def test_project_odd_dims_xy_fourier(self):
        # Some rotations in the xy plane where the rotated 
        # coordinates fall on the grid

        # Indices of the point mass
        i, j, k = (0, 2, 2)
        v, X, Y, Z = self.get_volume_and_coords_odd(i,j,k)
        
        angles = [np.pi/2, -3*np.pi/2, -np.pi/2, 3*np.pi/2, np.pi, -np.pi,
                2*np.pi, -2*np.pi]

        target_points = [(2,0), (2,0), (2,4), (2,4), (4,2), (4,2), (0,2),
                (0,2)]

        for (ang, target_p) in zip(angles, target_points):
            a = np.array([0,0,ang])
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, X, Y, Z, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        # Rotations in the xy plane where the rotated coordinates fall 
        # between two grid points. Take different point for variety
        i, j, k = (2, 3, 2)
        v, X, Y, Z = self.get_volume_and_coords_odd(i,j,k)

        #  For nearest neightbour it should be one
        angles = [np.pi/4, np.pi/4+np.pi/2, np.pi/4+np.pi, -np.pi/4]
        target_points = [(1,3), (1,1), (3,1), (3,3)]
        
        for (ang, target_p) in zip(angles, target_points):
            a = np.array([0,0,ang])
            vp_nn, _ = self.do_nn_and_tri_projection(v, X, Y, Z, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)

        # Check the trilinear interpolation rotation for each angle 
        target_vp_tri = np.zeros([5,5,4])
        target_vp_tri[1,3,0] = 2-np.sqrt(2)
        target_vp_tri[2,3,0] = 1/np.sqrt(2) - 1/2 
        target_vp_tri[1,2,0] = target_vp_tri[2,3,0]

        target_vp_tri[1,1,1] = 2-np.sqrt(2)
        target_vp_tri[1,2,1] = 1/np.sqrt(2)-1/2
        target_vp_tri[2,1,1] = 1/np.sqrt(2)-1/2

        target_vp_tri[3,1,2] = 2-np.sqrt(2)
        target_vp_tri[3,2,2] = 1/np.sqrt(2)-1/2
        target_vp_tri[2,1,2] = 1/np.sqrt(2)-1/2

        target_vp_tri[3,3,3] = 2-np.sqrt(2)
        target_vp_tri[3,2,3] = 1/np.sqrt(2)-1/2
        target_vp_tri[2,3,3] = 1/np.sqrt(2)-1/2

        for i in range(len(angles)):
            _, vp_tri = self.do_nn_and_tri_projection(v, X, Y, Z, (0,0,angles[i]))
            assert_array_almost_equal(vp_tri, target_vp_tri[:,:,i],decimal=15)


    def test_project_odd_dims_yz_fourier(self):
        # Some rotations in the yz plane where the rotated 
        # coordinates fall on the grid

        # Indices of the point mass
        i, j, k = (2, 2, 1)
        v, X, Y, Z = self.get_volume_and_coords_odd(i,j,k)

        angles = [np.pi/2, -3*np.pi/2, -np.pi/2, 3*np.pi/2]

        target_points = [(1,2), (1,2), (3,2), (3,2)]

        for (ang, target_p) in zip(angles, target_points):
            a = np.array([ang,0,0])
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, X, Y, Z, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        return

    def test_project_odd_dims_xz_fourier(self):
        # Some rotations in the xz plane where the rotated 
        # coordinates fall on the grid

        # Indices of the point mass
        i, j, k = (0,2,4)
        v, X, Y, Z = self.get_volume_and_coords_odd(i,j,k)

        angles = [np.pi/2, -3*np.pi/2, -np.pi/2, 3*np.pi/2]

        target_points = [(0,0), (0,0), (0,4), (0,4)]

        for (ang, target_p) in zip(angles, target_points):
            a = np.array([0,ang,0])
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, X, Y, Z, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        return

    def get_volume_and_coords_odd(self, i, j, k):
        nx = 5
        dx = 0.2 # to have exactly 1 between grid points
        v = np.zeros([nx, nx, nx])
        v[i,j,k] = 1

        x_freq = np.fft.fftfreq(nx, dx)
        y_freq = np.fft.fftfreq(nx, dx)
        z_freq = np.fft.fftfreq(nx, dx)
        X, Y, Z = np.meshgrid(x_freq, y_freq, z_freq, indexing = 'xy')

        return v, X, Y, Z

    def assert_point_mass_proj_one(self, v, idx):
        self.assertAlmostEqual(v[idx], 1, places = 14)
        self.assertAlmostEqual(np.sum(abs(v)), 1, places = 14)

    def do_nn_and_tri_projection(self, v, X, Y, Z, angles):
        vp_nn, _, _, _ = project(np.fft.ifftshift(v), X, Y, Z, angles, "nn")
        vp_nn = np.fft.fftshift(vp_nn)

        vp_tri, _, _, _ = project(np.fft.ifftshift(v), X, Y, Z, angles, "tri")
        vp_tri = np.fft.fftshift(vp_tri)

        return vp_nn, vp_tri

if __name__ == '__main__':
    unittest.main()
