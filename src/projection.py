import numpy as np
import jax.numpy as jnp
import itertools
from src.interpolate import interpolate
from src.utils import volume_fourier 
import jax



def project_spatial(v, angles, dimensions, method = "tri"):
    """Takes a centred object in the spatial domain and returns the centred
    projection in the spatial domain.
    If N is the number of pixels in one dimension, then the origin is 
    assumed to be in the pixel with index (N-1)/2 if N is odd 
    and N/2 is N is even, similar to an fftshifted Fourier grid.
    """ 
    
    # First ifftshift in the spatial domain 
    v = jnp.fft.ifftshift(v)
    V, X, Y, Z, _, _, _ = volume_fourier(v, dimensions)

    # Not bothered about using the full X, Y, Z here since for the big 
    # experiments we do it all in Fourier anyway.
    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    x_grid = [x_freq[1], len(x_freq)]
    y_grid = [y_freq[1], len(y_freq)]
    z_grid = [z_freq[1], len(z_freq)]

    V_slice, coords_slice = project(V, x_grid, y_grid, z_grid, angles, method)
   
    # Make it 2D
    V_slice = V_slice.reshape(V.shape[0], V.shape[1])

    # Back to spatial domain
    v_proj = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(V_slice)))
    
    return v_proj


def project(vol, x_grid, y_grid, z_grid, angles, interpolation_method = "tri"):
    """Projection in the Fourier domain.
    Assumption: the frequencies are in the 'standard' order for vol and the
    coordinates X, Y, Z."""
   
    # Get the rotated coordinates in the z=0 plane.
    slice_coords = rotate(x_grid, y_grid, angles)
    
    slice_interp = interpolate(slice_coords, x_grid, y_grid, z_grid, vol,
            interpolation_method)
    
    return slice_interp, slice_coords

def rotate(x_grid, y_grid, angles):
    """Rotate the coordinates given by X, Y, Z=0
    with Euler angles alpha, beta, gamma
    around axes x, y, z respectively.

    Parameters
    ----------
    x_grid, y_grid: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which 
        the volume is defined. The full grids can be obtained by running
        x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).
    angles:  3 x 1 array
        [alpha, beta, gamma] Euler angles

    Returns
    -------
    X_r, Y_r, Z_r : Nx x Ny x Nz arrays
        The coordinates after rotaion.
    """

    # Change later if too inefficient. Do not use itertools, it triggers
    # an XLA 'too slow' bug when calling with jit.

    # Generate the x and y grids.
    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
    y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1/(y_grid[0]*y_grid[1]))

    X,Y = jnp.meshgrid(x_freq,y_freq)
    coords = jnp.array([X.ravel(), Y.ravel(), jnp.zeros(len(x_freq)*len(y_freq))])

    # And finally rotate
    rotated_coords = get_rotation_matrix(angles) @ coords

    return rotated_coords

def get_rotation_matrix(angles):
    """Given the Euler angles alpha, beta, gamma, return 
    the rotation matrix."""

    alpha, beta, gamma = angles

    Rx = jnp.array([[1, 0, 0],
                   [0, jnp.cos(alpha), -jnp.sin(alpha)],
                   [0, jnp.sin(alpha), jnp.cos(alpha)]])

    Ry = jnp.array([[jnp.cos(beta), 0, jnp.sin(beta)],
                   [0, 1, 0],
                   [-jnp.sin(beta), 0, jnp.cos(beta)]])

    Rz = jnp.array([[jnp.cos(gamma), -jnp.sin(gamma), 0],
                    [jnp.sin(gamma), jnp.cos(gamma), 0],
                    [0, 0, 1]])

    return Rz @ Ry @ Rx
