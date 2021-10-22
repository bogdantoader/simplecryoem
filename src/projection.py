import numpy as np
import jax.numpy as jnp
from src.interpolate import interpolate
from src.utils import volume_fourier 



def project_spatial(v, angles, dimensions, method = "tri"):
    """Takes a centred object in the spatial domain and returns the centred
    projection in the spatial domain.""" 
    
    # First ifftshift in the spatial domain 
    v = jnp.fft.ifftshift(v)
    V, X, Y, Z, _, _, _ = volume_fourier(v, dimensions)
    V_slice, X_slice, Y_slice, Z_slice = project(V, X, Y, Z, angles, method)
    v_proj = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(V_slice)))

    return v_proj


def project(vol, X, Y, Z, angles, interpolation_method = "tri"):
    """Projection in the Fourier domain.
    Assumption: the frequencies are in the 'standard' order for vol and the
    coordinates X, Y, Z."""

    #TODO!! Any issue when the shape dimensions are not odd?

    # Only select the z=0 slice
    X_r, Y_r, Z_r = rotate(X[:,:,0], Y[:,:,0], Z[:,:,0], angles)
    slice_coords = jnp.array([X_r.ravel(), Y_r.ravel(),Z_r.ravel()])

    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    slice_interp = interpolate(slice_coords, x_freq, y_freq, z_freq, vol,
            interpolation_method)
    slice_interp_2d = slice_interp.reshape(vol.shape[0],vol.shape[1])

    return slice_interp_2d, X_r, Y_r, Z_r

# Rotation around the z axis to begin with
def rotate_z(X, Y, Z, alpha):
    rot_mat = np.array([[jnp.cos(alpha), -jnp.sin(alpha), 0],
                       [jnp.sin(alpha), jnp.cos(alpha), 0],
                       [0, 0, 1]])
    
    coords = jnp.array([X.ravel(), Y.ravel(), Z.ravel()])
    coords_r = jnp.matmul(rot_mat, coords)
    
    X_r = coords_r[0,:].reshape(X.shape)
    Y_r = coords_r[1,:].reshape(Y.shape)
    Z_r = coords_r[2,:].reshape(Z.shape)
    
    return X_r, Y_r, Z_r

def rotate(X, Y, Z, angles):
    """Rotate the coordinates given by X, Y, Z
    with Euler angles alpha, beta, gamma
    around axes x, y, z respectively.

    Parameters
    ----------
    X, Y, Z : Nx x Ny x Nz arrays
        The coordinates as given by meshgrid
    angles:  3 x 1 array
        [alpha, beta, gamma] Euler angles

    Returns
    -------
    X_r, Y_r, Z_r : Nx x Ny x Nz arrays
        The coordinates after rotaion.
    """

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

    coords = jnp.array([X.ravel(), Y.ravel(), Z.ravel()])
    coords_r = Rz @ Ry @ Rx @ coords

    X_r = coords_r[0,:].reshape(X.shape)
    Y_r = coords_r[1,:].reshape(Y.shape)
    Z_r = coords_r[2,:].reshape(Z.shape)

    return X_r, Y_r, Z_r
