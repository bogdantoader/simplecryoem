import numpy as np
import jax.numpy as jnp
from itertools import product
import jax

# Looks light it might be possible to only pass the grid step sizes and lengths
# rather than the full *_freq grids - if it has an impact on memory later.
def interpolate(i_coords, x_grid, y_grid, z_grid, vol, method):
    """Given a volume vol sampled on meshgrid given
    by x_grid, y_grid, z_grid, return the interpolated values of vol
    at the coordinates i_coords of M points. 
    Nearest neighbour or trilinear interpolation.

    Parameters
    ----------
    i_coords: 3 x M array
        Interpolation points
    x_grid, y_grid, z_grid: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which 
        the volume is defined. The full grids can be obtained by running
        x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).
    vol : Nx x Ny x Nz array
        The volume
    method: string
        "nn" for nearest neighbour
        "tri" for trilinear 
    Returns
    -------
    i_vals : NxNy x 1 array
        The interpolated values of vol.
    """
    
    if method == "nn":
        interp_func = get_interpolate_nn_lambda(x_grid, y_grid, z_grid, vol)
    elif method == "tri":
        interp_func = get_interpolate_tri_lambda(x_grid, y_grid, z_grid, vol)
   
    #i_vals = jax.vmap(interp_func, in_axes = 1)(i_coords)

    # apply_along_axis seems slightly faster than vmap (it is also implemented
    # using vmap)
    i_vals = jnp.apply_along_axis(
        interp_func,
        axis = 0,
        arr = i_coords
    )


    return i_vals

# Nearest neighbour interpolation
def get_interpolate_nn_lambda(x_grid, y_grid, z_grid, vol):

    # Obtain the closest grid point to the point and interpolate
    # i.e. take the value of the volume at the closest grid point.
    thelambda = lambda coords : vol[tuple(
        find_nearest_one_grid_point_idx(coords, x_grid, y_grid, z_grid)
    )]

    return thelambda


def get_interpolate_tri_lambda(x_grid, y_grid, z_grid, vol):

    # Obtain the eight grid points around each point and interpolate.
    thelambda = lambda coords : tri_interp_point(coords, vol,  
        find_nearest_eight_grid_points_idx(coords, x_grid, y_grid, z_grid)
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

    # The value of the function at each grid point.
    # Note that x and y indices are swapped in vol.
    # i.e. to obtain vol(x, y, z), call vol[y_idx, x_idx, z_idx]
    c000 = vol[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,0]] 
    c001 = vol[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,1]] 
    c010 = vol[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,0]] 
    c011 = vol[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,1]] 
    c100 = vol[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,0]] 
    c101 = vol[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,1]] 
    c110 = vol[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,0]] 
    c111 = vol[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,1]] 

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
def find_nearest_one_grid_point_idx(coords, x_grid, y_grid, z_grid):
    """For a point given by coords and a grid defined by
    x_grid, y_grid, z_grid, return the grid indices of the nearest grid point to coords.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Parameters
    ----------
    coords: array of length 3
        The coordinates of the input points
    x_grid, y_grid, z_grid: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which we want
        to find the nearest neighbour. The full grids can be obtained
        by running: x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).

    Returns
    -------
    index_in_volume: array of length 3
        The index in the volume (and grids) of the nearest grid point to
        coords. Note that x and y indices are swapped to match the indexing of
        the volume given by meshgrid(indices='xy')
    """

    x, y, z = coords
    xc_idx = find_nearest_grid_point_idx(x, x_grid[0], x_grid[1])
    yc_idx = find_nearest_grid_point_idx(y, y_grid[0], y_grid[1])
    zc_idx = find_nearest_grid_point_idx(z, z_grid[0], z_grid[1])
    
    # Note that x and y indices are swapped so the indexing is the same as in
    # volume
    xyz_idx = jnp.array([yc_idx, xc_idx, zc_idx])

    return xyz_idx.astype(jnp.int64)


def find_nearest_eight_grid_points_idx(coords, x_grid, y_grid, z_grid):
    """For a point given by coords and a grid defined by
    x_grid, y_grid, z_grid, return the 8 grid points nearest to coords 
    and their indices on the grid.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Parameters
    ----------
    coords: array of length 3
        The coordinates of the input points
    x_grid, y_grid, z_grid: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which we want
        to find the nearest eight grid points. The full grids can be obtained
        by running: x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).

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
    x0_idx, x1_idx = find_adjacent_grid_points_idx(x, x_grid[0], x_grid[1])
    y0_idx, y1_idx = find_adjacent_grid_points_idx(y, y_grid[0], y_grid[1])
    z0_idx, z1_idx = find_adjacent_grid_points_idx(z, z_grid[0], y_grid[1])

    x0 = get_fourier_grid_point(x0_idx, x_grid[0], x_grid[1])
    x1 = get_fourier_grid_point(x1_idx, x_grid[0], x_grid[1])

    y0 = get_fourier_grid_point(y0_idx, y_grid[0], y_grid[1])
    y1 = get_fourier_grid_point(y1_idx, y_grid[0], y_grid[1])

    z0 = get_fourier_grid_point(z0_idx, z_grid[0], z_grid[1])
    z1 = get_fourier_grid_point(z1_idx, z_grid[0], z_grid[1])

    xyz = jnp.array([[x0, x1], [y0, y1], [z0, z1]])
    xyz_idx = jnp.array([ [x0_idx, x1_idx], [y0_idx, y1_idx],[z0_idx, z1_idx]])

    return xyz, xyz_idx.astype(jnp.int64)  

def get_fourier_grid_point(idx, dx, N):
    """ Return the grid point at index idx from a grid of frequencies of 
    length N and spacing dx, assuming the standard order.
    """
    #return dx*idx if idx < N/2 else dx * (idx - N)

    # The jax-friendly rewriting of the above:
    return jax.lax.cond(idx < N/2, 
            true_fun = lambda _ : dx * idx,
            false_fun = lambda _ : dx * (idx - N), operand = None)


# Can this be vectorized for many coords/points?
# Would that be needed if we use jax.vmap anyway?
def find_adjacent_grid_points_idx(p, grid_spacing, grid_length):
    """For a one dimensional grid of Fourier samples
    and a point p, find the indices of the grid points
    on its left and its right.
    We assume that the Fourier grid, which here is specified by 
    the spancing and length,  follows the 'standard' order,
    i.e. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]), as generated by
    fftfreq(grid_length, 1/(grid_length * grid_spacing))
    """

    #dx = grid[1]

    # If, due to floating point errors, p/dx = 1.9999999, 
    # consider it to be on the grid at 2 and always make that the
    # left point, for consistency.
    eps = 1e-15
    
    #if p/dx - jnp.floor(p/dx) > 1-eps:
    #    pt = jnp.floor(p/dx) + 1 
    #else:
    #    pt = jnp.floor(p/dx)
    
    # Jaxify the if-else above
    pt = jnp.floor(p/grid_spacing) + (p/grid_spacing- jnp.floor(p/grid_spacing) > 1-eps).astype(jnp.float32)

    n = grid_length
    idx_left = jnp.mod(pt, n).astype(jnp.int32)
    idx_right = jnp.mod(idx_left + 1,n)

    return idx_left, idx_right


def find_nearest_grid_point_idx(p, grid_spacing, grid_length):
    """For a one dimensional grid of Fourier samples and a point p, 
    find the index of the grid point that is the closest to p.
    The grid is specified as grid_spacing and grid_length, and can be
    computed fully with:
    np.fft.fftfreq(grid_length, 1/(grid_length * grid_spacing))

    """
    
    eps = 1e-15 

    # For now, just generate the grid - hack it later if needed.
    grid = jnp.fft.fftfreq(int(grid_length), 1/(grid_length * grid_spacing))

    # In case we need to wrap around (maxium once on each side)
    increment = grid_length * grid_spacing

    #if p > max(grid):
    #    extended_grid = jnp.array([grid, grid+increment]).flatten()
    #elif p < min(grid):
    #    extended_grid = jnp.array([grid, grid-increment]).flatten()
    #else:
    #    extended_grid = grid

    # Avoiding if-else above due to jax. Replacing the above with cond
    # statements makes the code twice as slow.
    extended_grid = jnp.array([grid-increment, grid, grid+increment]).flatten()

    # Find the index of the closest grid point. 
    dists = abs(extended_grid - p)
    closest_idx1 = jnp.argmin(dists)
    dist1 = dists[closest_idx1]

    # We do the following to ensure that when the point is at the midpoint
    # between two grid points (or withing epsilon away from it due to e.g.
    # floating point error), we select the point on the left.
    
    # Find the index of the second closest grid point
    #dists[closest_idx1] = jnp.inf
    dists = dists.at[closest_idx1].set(jnp.inf)
    closest_idx2 = jnp.argmin(dists)
    dist2 = dists[closest_idx2]
   
    # If the distances are within eps of each other (i.e. if the point is
    # within epsilon from the midpoint between two grid point),
    # select the index on the left, unless the index on the right is at the
    # end, i.e. it is still the point on the left, wrapped around.

#    closest_idx1 = jnp.mod(closest_idx1, grid_length)
#    closest_idx2 = jnp.mod(closest_idx2, grid_length)
#    if abs(dist1 - dist2) > 2*eps:
#        # The first index found is clearly the closest one.
#        closest_idx = closest_idx1
#    elif closest_idx1 == 0 and closest_idx2 == grid_length - 1:
#        # Otherwise, if they are both close,
#        # the first index found is at zero and the second index found is at the
#        # end of the array, then the 'left' index is the second one.
#        closest_idx = closest_idx2
#    elif closest_idx2 == 0 and closest_idx1 == grid_length- 1:
#        # Same situation as above, but swapped indices.
#        closest_idx = closest_idx1
#    else:
#        # Otherwise, they are both close and we return the smaller index.
#        closest_idx = min(closest_idx1, closest_idx2)

    # Replace the above if-else logic with a jax-friendly alternative.
    closest_idx1 = jnp.mod(closest_idx1, grid_length)
    closest_idx2 = jnp.mod(closest_idx2, grid_length)
    
    closest_idx = jax.lax.cond(abs(dist1 - dist2) > 2*eps,
        # The first index found is clearly the closest one.
        true_fun = lambda _ : closest_idx1,
        false_fun = lambda _:
            jax.lax.cond((closest_idx1==0)/2 + (closest_idx2==grid_length-1)/2 >=1,
            # Otherwise, if they are both close,
            # the first index found is at zero and the second index found is at the
            # end of the array, then the 'left' index is the second one.
                true_fun = lambda _ : closest_idx2,
                false_fun = lambda _: 
                    jax.lax.cond((closest_idx2==0)/2 + (closest_idx1==grid_length-1)/2 >=1,
                    # Same situation as above, but swapped indices.
                        true_fun = lambda _ : closest_idx1,
                        false_fun = lambda _:
                            # Otherwise, they are both close and we return the smaller index.
                            jnp.min(jnp.array([closest_idx1, closest_idx2])),
                        operand = None
                    ),
                operand = None 
            ),
        operand = None
    )

    return closest_idx


