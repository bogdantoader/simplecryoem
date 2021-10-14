import numpy as np


def interpolate(i_coords, x_freq, y_freq, z_freq, vol, method):
    """Given a volume vol sampled on meshgrid given
    by x_freq, y_freq, z_freq, return the interpolated values of vol
    at the coordinates i_coords of M points. 
    Nearest neighbour or trilinear interpolation.

    Parameters
    ----------
    i_coords: 3 x M array
        Interpolation points
    x_freq, y_freq, z_freq : Nx, Ny, Nz arrays
        The axes of the grid the volume is defined on
    vol : Nx x Ny x Nz array
        The volume
    method: string
        "nn" for nearest neighbour
        "tri" for trilinear 
    Returns
    -------
    i_slice : NxNy x 1 array
        The interpolated values of vol.
    """
    
    if method == "nn":
        interp_func = get_interpolate_nn_lambda(x_freq, y_freq, z_freq, vol)
    elif method == "tri":
        interp_func = get_interpolate_tri_lambda(x_freq, y_freq, z_freq, vol)
   
    i_slice = np.apply_along_axis(
        interp_func,
        axis = 0,
        arr = i_coords
    )

    return i_slice 

# Nearest neighbour interpolation
def get_interpolate_nn_lambda(x_freq, y_freq, z_freq, vol):

    # Obtain the closest grid point to the point and interpolate
    # i.e. take the value of the volume at the closest grid point.
    thelambda = lambda coords : vol[tuple(
        find_nearest_one_grid_point_idx(coords, x_freq, y_freq, z_freq)
    )]

    return thelambda


def get_interpolate_tri_lambda(x_freq, y_freq, z_freq, vol):

    # Obtain the eight grid points around each point and interpolate.
    thelambda = lambda coords : tri_interp_point(coords, vol,  
        find_nearest_eight_grid_points_idx(coords, x_freq, y_freq, z_freq)
    )

    return thelambda 

def tri_interp_point(i_coords, vol, xyz_and_idx):
    """Trilinear interpolation of the volume vol at the point given by coords
    on the grid points given by the first element of the tuple xyz_and_idx. 
    Using the methods described on the Wikipedia page
    https://en.wikipedia.org/wiki/Trilinear_interpolation

    Parameters
    ----------
    i_coords: array of length 3
        The coordinates to interpolate at.
    vol: Nx x Ny x Nz array
        The function to interpolate
    xyz_and_idx: Tuple (xyz, xyz_idx) where
        xyz = [[[x0, x1], [y0, y1], [z0, z1]]
        xyz_idx = same as xyz but their indices in the grid on which vol is
        defined. See the return of the function
        'find_nearest_eight_grid_point_idx'

    Returns
    -------
    i_val: Double
       The value of the interpolated function at i_coords. 
    """

    x, y, z = i_coords
    xyz, xyz_idx = xyz_and_idx 

    xd = (x - xyz[0,0])/(xyz[0,1] - xyz[0,0])
    yd = (y - xyz[1,0])/(xyz[1,1] - xyz[1,0])
    zd = (z - xyz[2,0])/(xyz[2,1] - xyz[2,0])

    c000 = vol[xyz_idx[0,0], xyz_idx[1,0], xyz_idx[2,0]] 
    c001 = vol[xyz_idx[0,0], xyz_idx[1,0], xyz_idx[2,1]] 
    c010 = vol[xyz_idx[0,0], xyz_idx[1,1], xyz_idx[2,0]] 
    c011 = vol[xyz_idx[0,0], xyz_idx[1,1], xyz_idx[2,1]] 
    c100 = vol[xyz_idx[0,1], xyz_idx[1,0], xyz_idx[2,0]] 
    c101 = vol[xyz_idx[0,1], xyz_idx[1,0], xyz_idx[2,1]] 
    c110 = vol[xyz_idx[0,1], xyz_idx[1,1], xyz_idx[2,0]] 
    c111 = vol[xyz_idx[0,1], xyz_idx[1,1], xyz_idx[2,1]] 

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    i_val = c0 * (1 - zd) + c1 * zd

    return i_val


# Can this be vectorized? 
# If using vmap and jax.numpy.apply_along_axis (which is implemented using jax
# vmap, maybe no need to vectorize?
def find_nearest_one_grid_point_idx(coords, x_freq, y_freq, z_freq):
    """For a point given by coords and a grid defined by
    x_freq, y_freq, z_freq, return the grid indices of the nearest grid point to coords.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Parameters
    ----------
    coords: array of length 3
        The coordinates of the input points
    x_freq, y_freq, z_freq: Nx, Ny, Nz arrays
        The Fourier grids on which we want to find the nearest neighbour.

    Returns
    -------
    index_in_volume: array of length 3
        The index in the volume (and grids) of the nearest grid point to
        coords.
    """
    
    xyz, xyz_idx = find_nearest_eight_grid_points_idx(coords, x_freq, y_freq, z_freq)

    pts = np.array([[xyz[0,0], xyz[1,0], xyz[2,0]],
                    [xyz[0,0], xyz[1,0], xyz[2,1]],
                    [xyz[0,0], xyz[1,1], xyz[2,0]],
                    [xyz[0,0], xyz[1,1], xyz[2,1]],
                    [xyz[0,1], xyz[1,0], xyz[2,0]],
                    [xyz[0,1], xyz[1,0], xyz[2,1]],
                    [xyz[0,1], xyz[1,1], xyz[2,0]],
                    [xyz[0,1], xyz[1,1], xyz[2,1]]])
    
    pts_idx = np.array([[xyz_idx[0,0], xyz_idx[1,0], xyz_idx[2,0]],
                    [xyz_idx[0,0], xyz_idx[1,0], xyz_idx[2,1]],
                    [xyz_idx[0,0], xyz_idx[1,1], xyz_idx[2,0]],
                    [xyz_idx[0,0], xyz_idx[1,1], xyz_idx[2,1]],
                    [xyz_idx[0,1], xyz_idx[1,0], xyz_idx[2,0]],
                    [xyz_idx[0,1], xyz_idx[1,0], xyz_idx[2,1]],
                    [xyz_idx[0,1], xyz_idx[1,1], xyz_idx[2,0]],
                    [xyz_idx[0,1], xyz_idx[1,1], xyz_idx[2,1]]])

    sq_dist = np.sum((pts - coords.T)**2, axis=1)
    min_idx = np.argmin(sq_dist)
    index_in_volume = pts_idx[min_idx]
    
    return index_in_volume

def find_nearest_eight_grid_points_idx(coords, x_freq, y_freq, z_freq):
    """For a point given by coords and a grid defined by
    x_freq, y_freq, z_freq, return the 8 grid points nearest to coords 
    and their indices on the grid.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Parameters
    ----------
    coords: array of length 3
        The coordinates of the input points
    x_freq, y_freq, z_freq: Nx, Ny, Nz arrays
        The Fourier grids on which we want to find the nearest neighbour.

    Returns
    -------
    xyz : 3 x 2 array 
        The array [[x0, x1], [y0, y1], [z0, z1]]
        where x0, x1 are the grid points to the left and right of coords[0],
        y0, y1 are the grid points to the left and right of coords[1] and
        z0, z1 are the grid points to the left and right of coords[2].
    xyz_idx: 3 x 2 array
        Array containing the indices of the points in xyz in the grid given by
        x_freq, y_freq, z_freq.
    """
    x, y, z = coords
    x0_idx, x1_idx = find_adjacent_grid_points_idx(x, x_freq)
    y0_idx, y1_idx = find_adjacent_grid_points_idx(y, y_freq)
    z0_idx, z1_idx = find_adjacent_grid_points_idx(z, z_freq)

    x0 = x_freq[x0_idx]
    x1 = x_freq[x1_idx]

    y0 = y_freq[y0_idx]
    y1 = y_freq[y1_idx]

    z0 = z_freq[z0_idx]
    z1 = z_freq[z1_idx]

    xyz = np.array([[x0, x1], [y0, y1], [z0, z1]])
    xyz_idx = np.array([[x0_idx, x1_idx], [y0_idx, y1_idx], [z0_idx, z1_idx]])
    return xyz, xyz_idx  

# Can this be vectorized for many coords/points?
# Only dx and grid length are needed, not full grid
def find_adjacent_grid_points_idx(p, grid):
    """For a one dimensional grid of Fourier samples
    and a point p, find the indices of the grid points
    on its left and its right.
    The Fourier samples must be in 'standard' order,
    i.e. [0, 1, ..., n/2-1, -n/2, ..., -2, -1])
    """

    dx = grid[1]
    pt = np.floor(p / dx)
    n = len(grid)
    idx_left = int(np.mod(pt, n))
    idx_right = np.mod(idx_left + 1,n)
    return idx_left, idx_right




