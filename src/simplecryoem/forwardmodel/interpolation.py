import jax.numpy as jnp
import jax
from jax import config
import warnings

config.update("jax_enable_x64", True)


def interpolate(i_coords, grid_vol, vol, method):
    """Given a volume vol sampled on meshgrid given
    by grid_vol, return the interpolated values of vol
    at the coordinates i_coords of M points.
    Nearest neighbour or trilinear interpolation.

    This function assumes the 3D grid has the same length
    and spacing in each dimension, and calls the more
    general function 'interpolate_diff_grids', which
    takes different grid objects for each dimension.

    Parameters
    ----------
    i_coords: 3 x M array
        Interpolation points
    grid_vol: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which
        the volume is defined. The full grids can be obtained by running
        x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).
    vol : Nx x Ny x Nz array
        The volume
    method: string
        "nn" for nearest neighbour interpolation
        "tri" for trilinear interpolation
    Returns
    -------
    i_vals : NxNy x 1 array
        The interpolated values of vol at grid coordinates i_coords.
    """
    return interpolate_diff_grids(i_coords, grid_vol, grid_vol, grid_vol, vol, method)


def interpolate_diff_grids(i_coords, x_grid, y_grid, z_grid, vol, method):
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
        interp_func = _get_interpolate_nn_func(x_grid, y_grid, z_grid, vol)
    elif method == "tri":
        interp_func = _get_interpolate_tri_func(x_grid, y_grid, z_grid, vol)

    i_vals = jnp.apply_along_axis(interp_func, axis=0, arr=i_coords)

    return i_vals


def _get_interpolate_nn_func(x_grid, y_grid, z_grid, vol):
    """Obtain the closest grid point to the point and interpolate
    i.e. take the value of the volume at the closest grid point."""

    def interpolate_nn_func(coords):
        return vol[
            tuple(_find_nearest_one_grid_point_idx(
                coords, x_grid, y_grid, z_grid))
        ]

    return interpolate_nn_func


def _get_interpolate_tri_func(x_grid, y_grid, z_grid, vol):
    """Obtain the eight grid points around each point and interpolate."""

    def interpolate_tri_func(coords):
        coords, nearest_pts = _find_nearest_eight_grid_points_idx(
            coords, x_grid, y_grid, z_grid
        )
        interp_pts = _tri_interp_point(coords, vol, nearest_pts)

        return interp_pts

    return interpolate_tri_func


def _tri_interp_point(i_coords, vol, xyz_and_idx):
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

    xd = (x - xyz[0, 0]) / (xyz[0, 1] - xyz[0, 0])
    yd = (y - xyz[1, 0]) / (xyz[1, 1] - xyz[1, 0])
    zd = (z - xyz[2, 0]) / (xyz[2, 1] - xyz[2, 0])

    # The value of the function at each grid point.
    # Note that x and y indices are swapped in vol.
    # i.e. to obtain vol(x, y, z), call vol[y_idx, x_idx, z_idx]
    c000 = vol[xyz_idx[1, 0], xyz_idx[0, 0], xyz_idx[2, 0]]
    c001 = vol[xyz_idx[1, 0], xyz_idx[0, 0], xyz_idx[2, 1]]
    c010 = vol[xyz_idx[1, 1], xyz_idx[0, 0], xyz_idx[2, 0]]
    c011 = vol[xyz_idx[1, 1], xyz_idx[0, 0], xyz_idx[2, 1]]
    c100 = vol[xyz_idx[1, 0], xyz_idx[0, 1], xyz_idx[2, 0]]
    c101 = vol[xyz_idx[1, 0], xyz_idx[0, 1], xyz_idx[2, 1]]
    c110 = vol[xyz_idx[1, 1], xyz_idx[0, 1], xyz_idx[2, 0]]
    c111 = vol[xyz_idx[1, 1], xyz_idx[0, 1], xyz_idx[2, 1]]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    i_val = c0 * (1 - zd) + c1 * zd

    return i_val


def _find_nearest_one_grid_point_idx(coords, x_grid, y_grid, z_grid):
    """For a point given by coords and a grid defined by
    x_grid, y_grid, z_grid, return the grid indices of the nearest grid point to coords.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    A more efficient version of _find_nearest_one_grid_point_idx_old,
    following the implementation of _find_nearest_eight_grid_points_idx, which is
    faster and more memory friendly.

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

    coords, (xyz, xyz_idx) = _find_nearest_eight_grid_points_idx(
        coords, x_grid, y_grid, z_grid
    )

    # Make the list of the actual grid points
    pts = jnp.array(
        [
            [xyz[0, 0], xyz[1, 0], xyz[2, 0]],
            [xyz[0, 0], xyz[1, 0], xyz[2, 1]],
            [xyz[0, 0], xyz[1, 1], xyz[2, 0]],
            [xyz[0, 0], xyz[1, 1], xyz[2, 1]],
            [xyz[0, 1], xyz[1, 0], xyz[2, 0]],
            [xyz[0, 1], xyz[1, 0], xyz[2, 1]],
            [xyz[0, 1], xyz[1, 1], xyz[2, 0]],
            [xyz[0, 1], xyz[1, 1], xyz[2, 1]],
        ]
    )

    # Note that x and y indices are swapped in vol.
    # i.e. to obtain vol(x, y, z), call vol[y_idx, x_idx, z_idx]
    pts_idx = jnp.array(
        [
            [xyz_idx[1, 0], xyz_idx[0, 0], xyz_idx[2, 0]],
            [xyz_idx[1, 0], xyz_idx[0, 0], xyz_idx[2, 1]],
            [xyz_idx[1, 1], xyz_idx[0, 0], xyz_idx[2, 0]],
            [xyz_idx[1, 1], xyz_idx[0, 0], xyz_idx[2, 1]],
            [xyz_idx[1, 0], xyz_idx[0, 1], xyz_idx[2, 0]],
            [xyz_idx[1, 0], xyz_idx[0, 1], xyz_idx[2, 1]],
            [xyz_idx[1, 1], xyz_idx[0, 1], xyz_idx[2, 0]],
            [xyz_idx[1, 1], xyz_idx[0, 1], xyz_idx[2, 1]],
        ]
    )

    # Compute the distances between the given point ('coords')
    # and its eight neighbour grid points.
    dists = jax.vmap(lambda p1, p2: jnp.linalg.norm(p1 - p2, 2), in_axes=(None, 0))(
        coords, pts
    )
    min_idx = jnp.argsort(dists)[0]

    return pts_idx[min_idx]


def _find_nearest_one_grid_point_idx_old(coords, x_grid, y_grid, z_grid):
    """WARNING: This function is deprecated and should be replaced with
    the more efficient version _find_nearest_one_grid_point_idx.

    For a point given by coords and a grid defined by
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

    warnings.warn(
        "Deprecated: Use _find_nearest_one_grid_point_idx instead!", DeprecationWarning
    )

    x, y, z = coords
    xc_idx = _find_nearest_grid_point_idx(x, x_grid[0], x_grid[1])
    yc_idx = _find_nearest_grid_point_idx(y, y_grid[0], y_grid[1])
    zc_idx = _find_nearest_grid_point_idx(z, z_grid[0], z_grid[1])

    # Note that x and y indices are swapped so the indexing is the same as in
    # volume
    xyz_idx = jnp.array([yc_idx, xc_idx, zc_idx])

    return xyz_idx.astype(jnp.int64)


def _find_nearest_eight_grid_points_idx(coords, x_grid, y_grid, z_grid, eps=1e-13):
    """For a point given by coords and a grid defined by
    x_grid, y_grid, z_grid, return the 8 grid points nearest to coords
    and their indices on the grid.
    It assumes the grid is a Fourier DFT sampling grid in
    'standard' order (e.g. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]).

    Note: If the coords 'overflow', in either direction', we add/subtract
    the length of the grid to the grid point, so that the coord is between
    them (note - the indices stay the same).

    Parameters
    ----------
    coords: array of length 3
        The coordinates of the input points
    x_grid, y_grid, z_grid: [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier grids on which we want
        to find the nearest eight grid points. The full grids can be obtained
        by running: x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).
    eps: Double
        How far we can be from the grid point (on either side)
        to still be at that point.

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

    # For eps < 1e-13, we start to see artefacts, so be careful.

    cx, cy, cz = coords
    x0_idx, x1_idx = _find_adjacent_grid_points_idx(
        cx, x_grid[0], x_grid[1], eps)
    y0_idx, y1_idx = _find_adjacent_grid_points_idx(
        cy, y_grid[0], y_grid[1], eps)
    z0_idx, z1_idx = _find_adjacent_grid_points_idx(
        cz, z_grid[0], y_grid[1], eps)

    x0 = _get_fourier_grid_point(x0_idx, x_grid[0], x_grid[1])
    x1 = _get_fourier_grid_point(x1_idx, x_grid[0], x_grid[1])
    x0x1 = jnp.array([x0, x1])
    cx, x0x1 = _adjust_grid_points(cx, x0x1, x_grid, eps)

    y0 = _get_fourier_grid_point(y0_idx, y_grid[0], y_grid[1])
    y1 = _get_fourier_grid_point(y1_idx, y_grid[0], y_grid[1])
    y0y1 = jnp.array([y0, y1])

    cy, y0y1 = _adjust_grid_points(cy, y0y1, y_grid, eps)

    z0 = _get_fourier_grid_point(z0_idx, z_grid[0], z_grid[1])
    z1 = _get_fourier_grid_point(z1_idx, z_grid[0], z_grid[1])
    z0z1 = jnp.array([z0, z1])
    cz, z0z1 = _adjust_grid_points(cz, z0z1, z_grid, eps)

    coords = jnp.array([cx, cy, cz])
    xyz = jnp.array([x0x1, y0y1, z0z1])
    xyz_idx = jnp.array([[x0_idx, x1_idx], [y0_idx, y1_idx], [z0_idx, z1_idx]])

    return coords, (xyz, xyz_idx.astype(jnp.int64))


def _get_fourier_grid_point(idx, dx, N):
    """Return the grid point at index idx from a grid of frequencies of
    length N and spacing dx, assuming the standard order in Fourier space.
    """
    # return dx*idx if idx < N/2 else dx * (idx - N)

    # The jax-friendly rewriting of the above:
    return jax.lax.cond(
        idx < N / 2,
        true_fun=lambda _: dx * idx,
        false_fun=lambda _: dx * (idx - N),
        operand=None,
    )


# TODO: write tests for this function and also add tests to the tri interpolate
# functions with overflowing coords and the eps stuff.
def _adjust_grid_points(p, x0x1, x_grid, eps=1e-13):
    """Since x0, x1 are grid points on a Fourier grid and the point p
    can overflow once on either side of the grid, we have to ensure that
    x0 <= p <= x1 so that the interpolation gives sensible results.
    We fix p by taking mod and the grid points by adding the grid length.
    Note this doesn't affect the indices (since they are the indices
    in the volume).

    x_grid = [grid_spacing, grid_length]
    """
    px = jnp.prod(x_grid)

    # First, bring the point p to the range [0, grid_length*grid_spacing)
    p = jnp.mod(p, px)

    # We use eps to avoid mod(-1e-16, 7) = 7
    p = jax.lax.cond(
        jnp.abs(p - px) < eps,
        true_fun=lambda _: jnp.float64(0),
        false_fun=lambda _: jnp.float64(p),
        operand=None,
    )

    # And then also bring the grid points to the same range)
    x0x1 = jnp.mod(x0x1 + px, px)

    # Now, the only issue can be when the 'right' grid point is 0.
    # We fix that by making it grid_length*grid_spacing.
    x0x1 = jax.lax.cond(
        x0x1[1] == 0,
        true_fun=lambda _: x0x1.at[1].set(px),
        false_fun=lambda _: x0x1,
        operand=None,
    )

    return p, x0x1


def _find_adjacent_grid_points_idx(p, grid_spacing, grid_length, eps=1e-13):
    """For a one dimensional grid of Fourier samples
    and a point p, find the indices of the grid points
    to its left and its right.
    We assume that the Fourier grid, which here is specified by
    the spancing and length,  follows the 'standard' order,
    i.e. [0, 1, ..., n/2-1, -n/2, ..., -2, -1]), as generated by
    fftfreq(grid_length, 1/(grid_length * grid_spacing))
    """

    # dx = grid[1]

    # If, due to floating point errors, p/dx = 1.9999999,
    # consider it to be on the grid at 2 and always make that the
    # left point, for consistency.

    # if p/dx - jnp.floor(p/dx) > 1-eps:
    #    pt = jnp.floor(p/dx) + 1
    # else:
    #    pt = jnp.floor(p/dx)

    # Jaxify the if-else above
    pt = jnp.floor(p / grid_spacing) + (
        p / grid_spacing - jnp.floor(p / grid_spacing) > 1 - eps
    ).astype(jnp.float64)

    idx_left = jnp.mod(pt, grid_length).astype(jnp.int32)
    idx_right = jnp.mod(idx_left + 1, grid_length)

    return idx_left, idx_right


def _find_nearest_grid_point_idx(p, grid_spacing, grid_length, eps=1e-13):
    """For a one dimensional grid of Fourier samples and a point p,
    find the index of the grid point that is the closest to p.
    The grid is specified as grid_spacing and grid_length, and can be
    computed fully with:
    np.fft.fftfreq(grid_length, 1/(grid_length * grid_spacing))

    """

    # For now, just generate the grid.
    grid = jnp.fft.fftfreq(int(grid_length), 1 / (grid_length * grid_spacing))

    # In case we need to wrap around (maxium once on each side)
    increment = grid_length * grid_spacing

    # if p > max(grid):
    #    extended_grid = jnp.array([grid, grid+increment]).flatten()
    # elif p < min(grid):
    #    extended_grid = jnp.array([grid, grid-increment]).flatten()
    # else:
    #    extended_grid = grid

    # Avoiding if-else above due to jax. Replacing the above with cond
    # statements makes the code twice as slow.
    extended_grid = jnp.array(
        [grid - increment, grid, grid + increment]).flatten()

    # Find the index of the closest grid point.
    dists = abs(extended_grid - p)
    closest_idx1 = jnp.argmin(dists)
    dist1 = dists[closest_idx1]

    # We do the following to ensure that when the point is at the midpoint
    # between two grid points (or withing epsilon away from it due to e.g.
    # floating point error), we select the point on the left.
    # For consistency.

    # Find the index of the second closest grid point
    # dists[closest_idx1] = jnp.inf
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

    closest_idx = jax.lax.cond(
        abs(dist1 - dist2) > 2 * eps,
        # The first index found is clearly the closest one.
        true_fun=lambda _: closest_idx1,
        false_fun=lambda _: jax.lax.cond(
            (closest_idx1 == 0) / 2 + (closest_idx2 == grid_length - 1) / 2 >= 1,
            # Otherwise, if they are both close,
            # the first index found is at zero and the second index found is at the
            # end of the array, then the 'left' index is the second one.
            true_fun=lambda _: closest_idx2,
            false_fun=lambda _: jax.lax.cond(
                (closest_idx2 == 0) / 2 + \
                (closest_idx1 == grid_length - 1) / 2 >= 1,
                # Same situation as above, but swapped indices.
                true_fun=lambda _: closest_idx1,
                false_fun=lambda _:
                # Otherwise, they are both close and we return the smaller index.
                jnp.min(jnp.array([closest_idx1, closest_idx2])),
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )

    return closest_idx
