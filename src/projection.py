import numpy as np
from src.interpolate import interpolate
from src.utils import volume_fourier 


def project_spatial(v, angles, dimensions, method = "tri"):
    """Takes a centred object in the spatial domain and returns the centred
    projection in the spatial domain.""" 
    
    # First ifftshift in the spatial domain 
    v = np.fft.ifftshift(v)
    V, X, Y, Z, _, _, _ = volume_fourier(v, dimensions)
    V_slice, X_slice, Y_slice, Z_slice = project(V, X, Y, Z, angles, method)
    v_proj = np.real(np.fft.fftshift(np.fft.ifftn(V_slice)))

    return v_proj


def project(vol, X, Y, Z, angles, interpolation_method = "tri"):
    """Projection in the Fourier domain.
    Assumption: the frequencies are in the 'standard' order for vol and the
    coordinates X, Y, Z."""

    # Any issue when the shape dimensions are not odd?
    # In that case, need to interpolate to find the plane z=0, 
    # rather than selecting the z-slice at index 0.
    X_r, Y_r, Z_r = rotate(X, Y, Z, angles)

    # Select the slice at z=0, assuming shape[2] is odd. 

    slice_coords = np.array([X_r[:,:,0].flatten(), Y_r[:,:,0].flatten(),
        Z_r[:,:,0].flatten()])

    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    slice_interp = interpolate(slice_coords, x_freq, y_freq, z_freq, vol,
            interpolation_method)
    slice_interp_2d = slice_interp.reshape(vol.shape[0],vol.shape[1])

    slice_X = X_r[:,:,0]
    slice_Y = Y_r[:,:,0]
    slice_Z = Z_r[:,:,0]

    return slice_interp_2d, slice_X, slice_Y, slice_Z

# Rotation around the z axis to begin with
def rotate_z(X, Y, Z, alpha):
    rot_mat = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])
    
    coords = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    coords_r = np.matmul(rot_mat, coords)
    
    X_r = coords_r[0,:].reshape(X.shape)
    Y_r = coords_r[1,:].reshape(Y.shape)
    Z_r = coords_r[2,:].reshape(Z.shape)
    
    return X_r, Y_r, Z_r

# Euler angles
# TODO: no need to apply the rotation to all coords if in Fourier
# we are only interested in the new coordinates of the pre-rotation
# plane z=0 
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

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])

    coords = np.array([X.flatten(), Y.flatten(), Z.flatten()])
    coords_r = Rz @ Ry @ Rx @ coords

    X_r = coords_r[0,:].reshape(X.shape)
    Y_r = coords_r[1,:].reshape(Y.shape)
    Z_r = coords_r[2,:].reshape(Z.shape)

    return X_r, Y_r, Z_r
