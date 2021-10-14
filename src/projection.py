import numpy as np
from src.interpolate import interpolate_nn, interpolate_tri 


def project(vol, X, Y, Z, angles):
    """Projection in the Fourier domain"""
    X_r, Y_r, Z_r = rotate(X, Y, Z, angles)

    slice_coords = np.array([X_r[:,:,0].flatten(), Y_r[:,:,0].flatten(),Z_r[:,:,0].flatten()])

    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    slice_interp = interpolate_tri(slice_coords , x_freq, y_freq, z_freq, vol)
    slice_interp_2d = slice_interp.reshape(X_r.shape[0], X_r.shape[1])

    slice_X = slice_coords[0,:].reshape(X_r.shape[0], X_r.shape[1])
    slice_Y = slice_coords[1,:].reshape(X_r.shape[0], X_r.shape[1])
    slice_Z = slice_coords[2,:].reshape(X_r.shape[0], X_r.shape[1])

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
# we are only interested in the plane z=0 after rotation
def rotate(X, Y, Z, angles):
    """Rotate the coordinates given by X, Y, Z
    with Euler angles alpha, betta, gamma
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
