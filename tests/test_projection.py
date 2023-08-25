import unittest
from jax.config import config
import numpy as np
import jax.numpy as jnp
from simplecryoem.projection import project, project_spatial
from simplecryoem.utils import spherical_volume
from numpy.testing import assert_array_almost_equal

config.update("jax_enable_x64", True)


class TestProjection(unittest.TestCase):
    # Point mass on a small Fourier domain
    # Dimensions are odd numbers for now

    def test_project_odd_dims_xy_fourier(self):
        """Point source in the Fourier space, rotation in the z=0 plane.
        Testing nearest neighbour and trilinear interpolation, odd dimensions."""

        # The rotated coordinates fall on the grid.

        # Indices of the point mass
        i, j, k = (0, 2, 2)
        v, x_grid = self.get_volume_and_coords_odd(i, j, k)

        angles = [
            jnp.pi / 2,
            -3 * jnp.pi / 2,
            -jnp.pi / 2,
            3 * jnp.pi / 2,
            jnp.pi,
            -jnp.pi,
            2 * jnp.pi,
            -2 * jnp.pi,
        ]

        target_points = [(2, 0), (2, 0), (2, 4), (2, 4), (4, 2), (4, 2), (0, 2), (0, 2)]

        for ang, target_p in zip(angles, target_points):
            a = jnp.array([0, 0, ang])
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, x_grid, a)

            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        # Rotations in the xy plane where the rotated coordinates fall
        # between two grid points. Take different point for variety
        i, j, k = (2, 3, 2)
        v, x_grid = self.get_volume_and_coords_odd(i, j, k)

        #  For nearest neightbour it should be one
        angles = [jnp.pi / 4, jnp.pi / 4 + jnp.pi / 2, jnp.pi / 4 + jnp.pi, -jnp.pi / 4]
        target_points = [(1, 3), (1, 1), (3, 1), (3, 3)]

        for ang, target_p in zip(angles, target_points):
            a = jnp.array([0, 0, ang])
            vp_nn, _ = self.do_nn_and_tri_projection(v, x_grid, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)

        # Check the trilinear interpolation rotation for each angle
        target_vp_tri = np.zeros([5, 5, 4])
        target_vp_tri[1, 3, 0] = 2 - np.sqrt(2)
        target_vp_tri[2, 3, 0] = 1 / np.sqrt(2) - 1 / 2
        target_vp_tri[1, 2, 0] = target_vp_tri[2, 3, 0]

        target_vp_tri[1, 1, 1] = 2 - np.sqrt(2)
        target_vp_tri[1, 2, 1] = 1 / np.sqrt(2) - 1 / 2
        target_vp_tri[2, 1, 1] = 1 / np.sqrt(2) - 1 / 2

        target_vp_tri[3, 1, 2] = 2 - np.sqrt(2)
        target_vp_tri[3, 2, 2] = 1 / np.sqrt(2) - 1 / 2
        target_vp_tri[2, 1, 2] = 1 / np.sqrt(2) - 1 / 2

        target_vp_tri[3, 3, 3] = 2 - np.sqrt(2)
        target_vp_tri[3, 2, 3] = 1 / np.sqrt(2) - 1 / 2
        target_vp_tri[2, 3, 3] = 1 / np.sqrt(2) - 1 / 2

        target_vp_tri = jnp.array(target_vp_tri)

        for i in range(len(angles)):
            _, vp_tri = self.do_nn_and_tri_projection(v, x_grid, (0, 0, angles[i]))
            assert_array_almost_equal(vp_tri, target_vp_tri[:, :, i], decimal=15)

    def test_project_odd_dims_yz_fourier(self):
        """Rotations of a Fourier point mass in the yz plane where the rotated
        coordinates fall on the grid."""

        # Indices of the point mass
        i, j, k = (2, 2, 1)
        v, x_grid = self.get_volume_and_coords_odd(i, j, k)

        # Euler angles corresponding to rotation arond the x axis
        # by pi/2, -3pi/2, -pi/2, 3pi/2 respectively.
        angles = [
            [jnp.pi / 2, jnp.pi / 2, jnp.pi / 2],
            [-3 * jnp.pi / 2, -3 * jnp.pi / 2, -3 * jnp.pi / 2],
            [-jnp.pi / 2, jnp.pi / 2, -jnp.pi / 2],
            [3 * jnp.pi / 2, -3 * jnp.pi / 2, 3 * jnp.pi / 2],
        ]

        target_points = [(1, 2), (1, 2), (3, 2), (3, 2)]

        for ang, target_p in zip(angles, target_points):
            a = jnp.array(ang)
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, x_grid, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        return

    def test_project_odd_dims_xz_fourier(self):
        """Rotations of a Fourier point mass in the xz plane where the rotated
        coordinates fall on the grid."""

        # Indices of the point mass
        i, j, k = (0, 2, 4)
        v, x_grid = self.get_volume_and_coords_odd(i, j, k)

        angles = [jnp.pi / 2, -3 * jnp.pi / 2, -jnp.pi / 2, 3 * jnp.pi / 2]

        target_points = [(0, 0), (0, 0), (0, 4), (0, 4)]

        for ang, target_p in zip(angles, target_points):
            a = jnp.array([0, ang, 0])
            vp_nn, vp_tri = self.do_nn_and_tri_projection(v, x_grid, a)
            self.assert_point_mass_proj_one(vp_nn, target_p)
            self.assert_point_mass_proj_one(vp_tri, target_p)

        return

    def test_project_odd_dims_xy_spatial_pi2_p4(self):
        """Projections of a point mass in the spatial domain on the z=0 plane,
        rotated at -pi/2 and -pi/4, compared against explicit calculation.
        Nearest neighbour interpolation for -pi/2 and -pi/4, trilinear
        inteerpolation for -pi/2 only."""

        nx = 5
        shape = jnp.array([nx, nx, nx])
        dimensions = jnp.array([1, 1, 1])
        pixel_size = dimensions[0] / shape[0]
        radius = 1 / (2 * nx)
        intensity = 1

        # The centres of the pixels are at -0.4, -0.2, 0, 0.2, 0.4
        centres = [
            [0, 0, 0],
            [0.2, 0, 0],
            [0.4, 0, 0],
            [-0.2, 0, 0],
            [-0.4, 0, 0],
            [0, 0.2, 0],
            [0.2, 0.2, 0],
            [0.4, 0.2, 0],
            [-0.2, 0.2, 0],
            [-0.4, 0.2, 0],
            [0, -0.4, 0],
            [0.2, -0.4, 0],
            [0.4, -0.4, 0],
            [-0.2, -0.4, 0],
            [-0.4, -0.4, 0],
            # same as above but at different depths
            [0, 0, -0.2],
            [0.2, 0, -0.4],
            [0.4, 0, 0.2],
            [-0.2, 0, 0.4],
            [-0.4, 0, 0.4],
            [0, 0.2, 0.4],
            [0.2, 0.2, 0.4],
            [0.4, 0.2, -0.4],
            [-0.2, 0.2, 0.2],
            [-0.4, 0.2, -0.2],
            [0, -0.4, -0.2],
            [0.2, -0.4, 0.2],
            [0.4, -0.4, -0.4],
            [-0.2, -0.4, 0.4],
            [-0.4, -0.4, 0.2],
        ]

        Kxr2, Kyr2 = self.get_pi2_rotated_coordinates()
        Kxr4, Kyr4 = self.get_pi4_rotated_coordinates()

        for centre in centres:
            centre = jnp.array(centre)
            v = spherical_volume(shape, dimensions, centre, radius, intensity, False)

            # Calculate the -pi/2 projections
            v_proj2_nn = project_spatial(
                v, jnp.array([0, 0, -jnp.pi / 2]), pixel_size, method="nn"
            )
            v_proj2_tri = project_spatial(
                v, [0, 0, -jnp.pi / 2], pixel_size, method="tri"
            )

            # And the -pi/4 projection
            v_proj4_nn = project_spatial(
                v, [0, 0, -jnp.pi / 4], pixel_size, method="nn"
            )

            # The analytically calculated projections - for pi/2 both nn and
            # tri are the same, while for pi/4 we only have nn.
            point_idx = jnp.array(list(np.where(v == 1))).flatten()
            v_proj2_true = self.calculate_rotated_point_mass_projection(
                point_idx, Kxr2, Kyr2
            )
            v_proj4_true = self.calculate_rotated_point_mass_projection(
                point_idx, Kxr4, Kyr4
            )

            # And check that all's good
            # print("BB", v_proj4_nn)
            # print(v_proj4_true)
            assert_array_almost_equal(v_proj2_nn, v_proj2_true, decimal=15)
            assert_array_almost_equal(v_proj2_tri, v_proj2_true, decimal=15)
            assert_array_almost_equal(v_proj4_nn, v_proj4_true, decimal=15)
        return

    def calculate_rotated_point_mass_projection(self, idx, Kxr, Kyr):
        """Return the analytically calculated projection of a point mass
        in 5 x 5 x 5 spatial domain, rotated pi/2 or pi/4
        (depending on Kxr and Kyr - the rotated coordinate matrices)."""

        # kx, ky Fourier coordinates
        Ky = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [-2, -2, -2, -2, -2],
                [-1, -1, -1, -1, -1],
            ]
        )
        Kx = Ky.T

        # We need the iffthifted indices in idx
        # When ifftshifted, the indices (2, 3) become (0, 1)
        xy_freq = jnp.array([0, 1, 2, -2, -1])
        new_idx1 = jnp.fft.fftshift(xy_freq)[idx[0]]
        new_idx2 = jnp.fft.fftshift(xy_freq)[idx[1]]

        # And calculate the actual ifftn(fftn(v))
        vr0_a = jnp.zeros([5, 5], dtype=jnp.complex128)
        for i in range(5):
            for j in range(5):
                vr0_a = vr0_a.at[i, j].set(
                    1
                    / 25
                    * jnp.sum(
                        jnp.exp(
                            1j
                            * 2
                            * jnp.pi
                            / 5
                            * (i * Kx + j * Ky - new_idx1 * Kxr - new_idx2 * Kyr)
                        )
                    )
                )

        vr0_a = jnp.real(jnp.fft.fftshift(vr0_a))

        return vr0_a

    def get_pi2_rotated_coordinates(self):
        Ky = jnp.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [-2, -2, -2, -2, -2],
                [-1, -1, -1, -1, -1],
            ]
        )
        Kx = Ky.T

        # Get the pi/2 rotated coordinates. To rotate the object by -pi/2, we need to
        # rotate the coordinates by pi/2, so negate the angle.
        gamma = jnp.pi / 2
        Kxr, Kyr = self.rotate_coordinates_in_xy(Kx, Ky, gamma)

        return Kxr, Kyr

    def get_pi4_rotated_coordinates(self):
        # Get the pi/4 rotated coordinates, calculated on paper.
        # Note that to rotate the object by -pi/4, we need to
        # rotate the coordinates by pi/4.
        # So these coordinates will lead to an object rotated by -pi/4.

        Kxr = jnp.array(
            [
                [0, 1, 1, -1, -1],
                [-1, 0, 1, -2, -1],
                [-1, -1, 0, 2, -2],
                [1, 2, -2, 0, 1],
                [1, 1, 2, -1, 0],
            ]
        )

        Kyr = jnp.array(
            [
                [0, 1, 1, -1, -1],
                [1, 1, 2, -1, 0],
                [1, 2, -2, 0, 1],
                [-1, -1, 0, 2, -2],
                [-1, 0, 1, -2, -1],
            ]
        )

        return Kxr, Kyr

    def rotate_coordinates_in_xy(self, X, Y, gamma):
        R = jnp.array([[np.cos(gamma), -np.sin(gamma)], [np.sin(gamma), np.cos(gamma)]])

        rc = []
        for x, y in zip(X.flatten(), Y.flatten()):
            xy2 = R @ np.array([x, y])
            rc.append(list(xy2))

        rc = jnp.array(rc)

        Xr = rc[:, 0].reshape(X.shape)
        Yr = rc[:, 1].reshape(Y.shape)

        return Xr, Yr

    def get_volume_and_coords_odd(self, i, j, k):
        nx = 5
        dx = 0.2  # to have exactly 1 between grid points
        v = np.zeros([nx, nx, nx])
        v[i, j, k] = 1
        v = jnp.array(v)

        x_freq = jnp.fft.fftfreq(nx, dx)

        x_grid = np.array([x_freq[1], len(x_freq)])

        return v, x_grid

    def assert_point_mass_proj_one(self, v, idx):
        self.assertAlmostEqual(v[idx], 1, places=14)
        self.assertAlmostEqual(jnp.sum(abs(v)), 1, places=14)

    def do_nn_and_tri_projection(self, v, x_grid, angles):
        vp_nn = project(
            jnp.fft.ifftshift(v), angles, [0, 0], None, x_grid, x_grid, "nn"
        )
        vp_tri = project(
            jnp.fft.ifftshift(v), angles, [0, 0], None, x_grid, x_grid, "tri"
        )

        vp_nn = vp_nn.reshape(v.shape[0], v.shape[1])
        vp_tri = vp_tri.reshape(v.shape[0], v.shape[1])

        vp_nn = np.fft.fftshift(vp_nn)
        vp_tri = np.fft.fftshift(vp_tri)

        return vp_nn, vp_tri


if __name__ == "__main__":
    unittest.main()
