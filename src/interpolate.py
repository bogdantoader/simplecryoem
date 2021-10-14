import numpy as np

# Nearest neighbour interpolation
# We don't really need the full X, Y, Z here, 
# but only the actual grid points
def interpolate(i_coords, X, Y, Z, vol):
    """Given a volume vol sampled on meshgrid given
    by X, Y, Z, return the interpolated values of vol
    at the coordinates i_coords.

    Nearest neighbour interpolation.

    Parameters
    ----------
    i_coords: 3 x ... array
        Interpolation points
    X, Y, Z : Nx x Ny x Nz arrays
        Grid the volume is defined on
    vol : Nx x Ny x Nz array
        The volume

    Returns
    -------
    i_slice : NxNy x 1 array
        The interpolated values of vol.
    """

    x_freq = X[0,:,0]
    y_freq = Y[:,0,0]
    z_freq = Z[0,0,:]

    nearest_points_idxs = np.apply_along_axis(
        lambda c : find_nearest_grid_points_idx(c, x_freq, y_freq, z_freq),
        axis = 0,
        arr = i_coords)

    # 'Interpolation'
    #i_vol = vol.take(nearest_points_idxs,axis = 0)
    i_slice = np.apply_along_axis(lambda idx : vol[tuple(idx)],
                               axis = 0,
                               arr = nearest_points_idxs)

    #print(i_coords)
    #print(nearest_points_idxs)
    #print(i_vol)

    return i_slice

# Can this be vectorized?
def find_nearest_grid_points_idx(coords, x_freq, y_freq, z_freq):
    """For a point given by coords and a grid defined by
    X, Y, Z, return the grid indices of the nearest grid point to coords.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Parameters
    ---------
    coords: array of length 3
        The coordinates of the input points
    x_freq, y_freq, z_freq: Nx, Ny, Nz arrays
        The Fourier grids on which we want to find the nearest neighbour.
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

    pts = np.array([[x0, y0, z0],
                    [x0, y0, z1],
                    [x0, y1, z0],
                    [x0, y1, z1],
                    [x1, y0, z0],
                    [x1, y0, z1],
                    [x1, y1, z0],
                    [x1, y1, z1]])
    pts_idx = np.array([[x0_idx, y0_idx, z0_idx],
                    [x0_idx, y0_idx, z1_idx],
                    [x0_idx, y1_idx, z0_idx],
                    [x0_idx, y1_idx, z1_idx],
                    [x1_idx, y0_idx, z0_idx],
                    [x1_idx, y0_idx, z1_idx],
                    [x1_idx, y1_idx, z0_idx],
                    [x1_idx, y1_idx, z1_idx]])

    #print(pts)
    #print(pts_idx)
    #print(pts.shape)
    #print(coords.shape)


    sq_dist = np.sum((pts - coords.T)**2, axis=1)

    min_idx = np.argmin(sq_dist)

    return pts_idx[min_idx]

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
