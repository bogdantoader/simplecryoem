import numpy as np
import jax
import jax.numpy as jnp
from  matplotlib import pyplot as plt



def volume_comp(shape, dimensions, centres, radii, intensities, 
        apply_filter = False, sigma = 0.01):
    """Create a volume that is a sum of rand_volumes with given centres and
    radii.

    Parameters
    ----------
    shape : 3 x 1 array
        Dimensions of the volume, in number of elements
    dimensions: 3 x 1 array
        Dimensions of the volume, in units (e.g. Angst?)
    centres: N x 3 array
        Centre coordinates of each of the N components
    radii: N array 
        Radii of each of the N components
    """
    vol = sum(map(lambda cr : 
            spherical_volume(shape, dimensions, cr[0], cr[1], cr[2], False,
                apply_filter, sigma),
        zip(centres, radii, intensities)))
    vol = vol/jnp.max(vol)
    
    return vol

def spherical_volume(shape, dimensions, centre, radius, intensity, rand_or_not, 
        apply_filter = False, sigma = 0.01):
    """Generate a random smoothed spherical volume.

    Parameters
    ----------
    shape : 3 x 1 array
        Dimensions of the volume, in number of elements
    dimensions: 3 x 1 array
        Dimensions of the volume, in units (e.g. Angst?)
    radius: float
        Radius of spherical object
    sigma: float
        Sigma for the Gaussian window 
    rand_or_not: boolean
        If true generate the values using randn, otherwise ones.
    apply_filter : boolean
        If true, apply a Gaussian filter to the volume in the Fourier domain.
    sigma: float
        Sigma value for Gausian filter.
    Returns
    -------
    vol
        the volume
    """
    
    Nx, Ny, Nz = shape
    vol = intensity * np.ones([Nx, Ny, Nz])
    if rand_or_not:
        vol = vol + np.random.randn(Nx, Ny, Nz) 
        vol[vol < 0] = 0
    
    Lx, Ly, Lz = dimensions  
    dx, dy, dz = dimensions/shape # "pixel" size
    
    # By adjusting the interval by half a pixel on each side
    # we ensure the sampling locations are 
    # the centres of the "pixels"
    coords_x = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, Nx)
    coords_y = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, Ny)
    coords_z = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, Nz) 
    X, Y, Z = np.meshgrid(coords_x, coords_y, coords_z, indexing='xy')
    
    mask = create_mask(X, Y, Z, centre, radius) 
   
    if apply_filter:
        vol = low_pass_filter(mask*vol, X, Y, Z, sigma)
    else:
        vol =  mask * vol
    return vol

def create_mask(x_grid, centre, radius):
    x_freq = np.fft.fftfreq(x_grid[1].astype(np.int64), 1/(x_grid[1] * x_grid[0]))
    y_freq = x_freq
    z_freq = x_freq

    X, Y, Z = np.meshgrid(x_freq, z_freq, y_freq)

    mask = np.ones(X.shape)
    cx, cy, cz = centre
    r = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    mask[r > radius] = 0

    return np.array(mask)


def low_pass_filter(vol, X, Y, Z, sigma):
    gauss = np.exp(-(X**2 + Y**2 + Z**2)/(2*sigma))
    gauss = gauss/max(gauss.ravel())
    gauss = np.fft.fftshift(gauss)

    low_pass_vol = np.fft.ifftn(np.fft.fftn(vol) * gauss)
    return np.real(low_pass_vol)

def volume_fourier(vol, pixel_size, shape_f = None):
    """Calculate the FFT of the volume and return the frequency coordinates.

    Parameters
    ----------
    vol :
        Volume in spatial domain
    pixel_size: double 
        Pixel size, in units (e.g. Angst)
    shape_f: 3 x 1 array
        Shape of the Fourier volume

    Returns
    -------
    vol_f
        the Fourier volume
    X_f, Y_f, Z_f
        Fourier points
    """

    if shape_f == None:
        shape_f = vol.shape

    #vol_f = jnp.fft.fftn(vol, shape_f)
    vol_f = jnp.fft.fftn(vol)

    Nx, Ny, Nz = vol.shape
    Nx_f, Ny_f, Nz_f = shape_f

    x_freq = jnp.fft.fftfreq(Nx_f, pixel_size)
    y_freq = jnp.fft.fftfreq(Ny_f, pixel_size)
    z_freq = jnp.fft.fftfreq(Nz_f, pixel_size)

    X_f, Y_f, Z_f = jnp.meshgrid(x_freq, y_freq, z_freq, indexing='xy')

    return vol_f, X_f, Y_f, Z_f

def mip_z(img):
    plt.imshow(np.max(img, axis = 2))
    return

def mip_x(img):
    plt.imshow(np.max(img, axis = 0))
    return

# TODO: tests for the three functions below,
# and change the names of these functions, they're not great
def rescale_smaller_grid(v, x_grid, y_grid, z_grid, radius):
    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
    y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1/(y_grid[0]*y_grid[1]))
    z_freq = jnp.fft.fftfreq(int(z_grid[1]), 1/(z_grid[0]*z_grid[1]))
    X, Y, Z = jnp.meshgrid(x_freq, y_freq, z_freq)
    
    idx = (jnp.sqrt(X**2 + Y**2 + Z**2) <= radius)

    idx1 = idx[0,:,0]
    idx2 = idx[:,0,0]
    idx3 = idx[0,0,:]
    
    len_x = len(x_freq[idx1])
    len_y = len(y_freq[idx2])
    len_z = len(z_freq[idx3])

    v_c = v[idx1][:,idx2][:,:,idx3]
   
    x_grid_c = [x_grid[0], len_x]
    y_grid_c = [y_grid[0], len_y]
    z_grid_c = [z_grid[0], len_z]

    return v_c, x_grid_c, y_grid_c, z_grid_c
    

def get_pad_width(l0, l1):
    """Determine the padding width that a centred object of length l0 in the
    Fourier domain should be padded with so that the final length is l1."""
    
    if jnp.mod(l1-l0, 2) == 0: 
        w = (l1-l0)/2
        pad_width = [w, w]
    else:
        w = jnp.floor((l1-l0)/2)

        if jnp.mod(l0, 2) == 0:
            pad_width = [w, w+1]
        else:
            pad_width = [w+1, w]
    return jnp.array(pad_width).astype(jnp.int64)
    
# TODO: this should take a new radius probably,rather than new grid  length
def rescale_larger_grid(v, x_grid, y_grid, z_grid, new_grid_lengths):
    """Assume the new grid lengths are larger than the current ones."""
    
    x_grid_new = [x_grid[0], new_grid_lengths[0]]
    y_grid_new = [y_grid[0], new_grid_lengths[1]]
    z_grid_new = [z_grid[0], new_grid_lengths[2]]

    pad_width_x = get_pad_width(x_grid[1], new_grid_lengths[0])
    pad_width_y = get_pad_width(y_grid[1], new_grid_lengths[1])
    pad_width_z = get_pad_width(z_grid[1], new_grid_lengths[2])
    
    v_new = jnp.pad(jnp.fft.fftshift(v), (pad_width_x, pad_width_y, pad_width_z))
    v_new = jnp.fft.ifftshift(v_new)
    
    return v_new, x_grid_new, y_grid_new, z_grid_new  


def get_rotation_matrix(alpha, beta, gamma):
    """Given the Euler angles alpha, beta, gamma, return 
    the rotation matrix. As seen in the pyEM implementation."""

    ca = jnp.cos(alpha)
    cb = jnp.cos(beta)
    cg = jnp.cos(gamma)
    sa = jnp.sin(alpha)
    sb = jnp.sin(beta)
    sg = jnp.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    r = jnp.array([[cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
                  [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
                  [sc, ss, cb]])
    return r

@jax.jit
def l2sq(x, y = 0):
    return jnp.real(jnp.sum(jnp.conj(x-y)*(x-y)))
