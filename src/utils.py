import numpy as np
from  matplotlib import pyplot as plt


# TODO: maybe put all the coordinates X, Y, Z in one variable 
# if it makes more sense 

def volume_comp(shape, dimensions, centres, radii, intensities):
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
            rand_volume(shape, dimensions, cr[0], cr[1], cr[2]),
        zip(centres, radii, intensities)))
    
    return vol

# TODO: think properly about inputs-outputs.
# In the spatial domain, the function below makes sense.
def rand_volume(shape, dimensions, centre, radius, intensity, sigma = 0.1):
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

    Returns
    -------
    vol
        the volume
    """
    
    Nx, Ny, Nz = shape
    vol = np.random.randn(Nx, Ny, Nz) + intensity
    #vol = np.ones(shape) + 5
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
    
    #return low_pass_filter(mask*vol, X, Y, Z, sigma), X, Y, Z
    return mask * vol

def create_mask(X, Y, Z, centre, radius):
    mask = np.ones(X.shape)
    cx, cy, cz = centre
    r = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    mask[r > radius] = 0
    return mask

def low_pass_filter(vol, X, Y, Z, sigma):
    gauss = np.exp(-(X**2 + Y**2 + Z**2)/(2*sigma))
    gauss = gauss/max(gauss.ravel())
    gauss = np.fft.fftshift(gauss)

    low_pass_vol = np.fft.ifftn(np.fft.fftn(vol) * gauss)
    return np.real(low_pass_vol)

def volume_fourier(vol, dimensions, shape_f = None):
    """Calculate the FFT of the volume and return the frequency coordinates.

    Parameters
    ----------
    vol :
        Volume in spatial domain
    dimensions: 3 x 1 array
        Spatial dimensions of the volume, in units (e.g. Angst?)
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

    vol_f = np.fft.fftn(vol, shape_f)

    Nx, Ny, Nz = vol.shape
    Nx_f, Ny_f, Nz_f = shape_f
    dx, dy, dz = dimensions/vol.shape # "pixel" size

    x_freq = np.fft.fftfreq(Nx_f, dx)
    y_freq = np.fft.fftfreq(Ny_f, dy)
    z_freq = np.fft.fftfreq(Nz_f, dz)

    X_f, Y_f, Z_f = np.meshgrid(x_freq, y_freq, z_freq, indexing='xy')

    return vol_f, X_f, Y_f, Z_f, dx, dy, dz

def mip_z(img):
    plt.imshow(np.max(img, axis = 2))
    return



