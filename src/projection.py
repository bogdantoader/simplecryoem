import numpy as np
import jax.numpy as jnp
import itertools
from src.interpolate import interpolate
from src.utils import volume_fourier, create_mask, get_rotation_matrix 
from src.ctf import eval_ctf
import jax
from jax.config import config
from external.pyem.pyem.vop import grid_correct
from external.pyem.pyem import star
from matplotlib import pyplot as plt

config.update("jax_enable_x64", True)


def project_spatial(v, angles, pixel_size, shifts = [0,0], method = "tri", ctf_params = None):
    """Takes a centred object in the spatial domain and returns the centred
    projection in the spatial domain.
    If N is the number of pixels in one dimension, then the origin is 
    assumed to be in the pixel with index (N-1)/2 if N is odd 
    and N/2 is N is even, similar to an fftshifted Fourier grid.
    """ 
    
    # First ifftshift in the spatial domain 
    v = jnp.fft.ifftshift(v)
    V, X, Y, Z = volume_fourier(v, pixel_size)

    # Added mask to compare with pyem - needs more fiddling
    #V = V * create_mask(X,Y,Z, (0,0,0), np.max(X))

    # Not bothered about using the full X, Y, Z here since for the big 
    # experiments we do it all in Fourier anyway.
    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    # IMPORTANT: do not make this a Jax array
    x_grid = np.array([x_freq[1], len(x_freq)])
    y_grid = np.array([y_freq[1], len(y_freq)])
    z_grid = np.array([z_freq[1], len(z_freq)])

    V_slice = project(V, angles, shifts, ctf_params, x_grid, y_grid, z_grid, method)
   
    # Make it 2D
    V_slice = V_slice.reshape(V.shape[0], V.shape[1])

    # Back to spatial domain
    v_proj = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(V_slice)))
    
    return v_proj

# TODO: write the doc string properly
def project(vol, angles, shifts, ctf_params, x_grid, y_grid, z_grid, interpolation_method = "tri"):
    """Projection in the Fourier domain.
    Assumption: the frequencies are in the 'standard' order for vol and the
    coordinates X, Y, Z.

    Parameters:G
    -----------
    vol: 
        Volume in Fourier domain, in standard order. 

    angles: [psi, tilt, rot]
        Proper Euler angles.

    shifts : [originx, originy]

    ctf_params : 9 x 1 array or None
        As in the ctf file.

    x_grid, y_grid, z_grid : [grid_spacing, grid_length]

    interp_method : "tri" or "nn"

    Returns:
    --------

    """
   
    # Get the rotated coordinates in the z=0 plane.
    proj_coords = rotate(x_grid, y_grid, angles)
  

    proj = interpolate(proj_coords, x_grid, y_grid, z_grid, vol, interpolation_method)

    shift = get_shift_term(x_grid, y_grid, shifts)
    proj *= shift

    if ctf_params is not None:
        x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
        y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1/(y_grid[0]*y_grid[1]))

        X,Y = jnp.meshgrid(x_freq,y_freq)
        r = jnp.sqrt(X**2 + Y**2)
        theta  = jnp.arctan2(Y, X)

        ctf = eval_ctf(r, theta, ctf_params)

        #plt.imshow(r); plt.colorbar()

        proj *= ctf.ravel()
    
    #return proj, proj_coords
    return proj

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
    
    angles = -jnp.array(angles)

    # And finally rotate
    rotated_coords = get_rotation_matrix(*angles) @ coords

    return rotated_coords

def get_shift_term(x_grid, y_grid, shifts):
    """Generate the phase term corresponding to the shifts in 
    units (e.g. Angstroms)."""
    
    # Generate the x and y grids.
    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
    y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1/(y_grid[0]*y_grid[1]))

    X,Y = jnp.meshgrid(x_freq,y_freq)
    shift = jnp.exp(2 * jnp.pi * 1j * (X * shifts[0] + Y * shifts[1])) 

    return shift.ravel()


def project_star_params(vol, p, pfac = 1):
    """Spatial domain projection of vol with parameters from one row of
    a star file given by dictionary p. Useful to compare against Relion/pyem."""

    vol = grid_correct(vol, pfac = pfac, order = 1)

    pixel_size = star.calculate_apix(p) #* 64.0/66.0

    f3d, X, Y, Z = volume_fourier(np.fft.ifftshift(vol), pixel_size)

    mymask = create_mask(X,Y,Z, (0,0,0), np.max(X)+X[1,1,0])
    f3d = f3d * mymask

    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    # IMPORTANT: do not make this a Jax array
    x_grid = np.array([x_freq[1], len(x_freq)])
    y_grid = np.array([y_freq[1], len(y_freq)])
    z_grid = np.array([z_freq[1], len(z_freq)])

    angles = jnp.array([p[star.Relion.ANGLEPSI],
             p[star.Relion.ANGLETILT],
             p[star.Relion.ANGLEROT]]) / 180* jnp.pi # third angle is
                                        # rotation around the first z axis


    shifts = jnp.array([p[star.Relion.ORIGINX], p[star.Relion.ORIGINY]]) * pixel_size

#    ctf_params = {'def1'  : p[star.Relion.DEFOCUSU], 
#                  'def2'  : p[star.Relion.DEFOCUSV],
#                  'angast': p[star.Relion.DEFOCUSANGLE], 
#                  'phase' : p[star.Relion.PHASESHIFT],
#                  'kv'    : p[star.Relion.VOLTAGE],
#                  'ac'    : p[star.Relion.AC],
#                  'cs'    : p[star.Relion.CS],
#                  'bf'    : 0,
#                  'lp'    : 2 * pixel_size}

    # The project function requires the CTF parameters to be a list with
    # the elements ordered like the keys in the dict above.
    ctf_params = [p[star.Relion.DEFOCUSU], 
                  p[star.Relion.DEFOCUSV],
                  p[star.Relion.DEFOCUSANGLE], 
                  p[star.Relion.PHASESHIFT],
                  p[star.Relion.VOLTAGE],
                  p[star.Relion.AC],
                  p[star.Relion.CS],
                  0,
                  2 * pixel_size]

    f2d, coords_slice = project(f3d, x_grid, y_grid, z_grid, angles, shifts, 'tri',
            ctf_params)
    f2d = f2d.reshape(f3d.shape[0], f3d.shape[1]) * mymask[:,:,0]
    proj = np.real(np.fft.fftshift(np.fft.ifftn(f2d)))

    return proj


def create_grid(nx, px):
    """Create the (one dimensional) Fourier grid used for projections.
    
    <<<IMPORTANT!!!>>> 
    The grids must not be Tracer (aka Jax)  objects."""

    x_freq = np.fft.fftfreq(nx, px)
    x_grid = np.array([x_freq[1], len(x_freq)])
    
    return x_grid 


