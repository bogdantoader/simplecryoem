import numpy as np
import jax
import jax.numpy as jnp
from  matplotlib import pyplot as plt
from src.utils import get_rotation_matrix
from src.projection import rotate_z0
from src.interpolate import find_nearest_eight_grid_points_idx, find_nearest_one_grid_point_idx
from src.emfiles import load_data
import mrcfile

def calc_fsc(v1, v2, grid, dr = None):
    """Calculate the fourier shell correlation between the Fourier 
    volumes v1 and v2 on the Fourier grid given by grid and shell
    width given by dr.

    Parameters:
    ----------
    v1, v2: (N x N x N) arrays
        Fourier volumes to compute the FSC between.
    grid: [dx, N] 
        The Fourier grid the volumes are defined on.
        dx is the spacing and N is the number of points of the grid.
    dr: double 
        The width of each Fourier shell. 
    Returns:
    -------
    res: double array
        The resolutions defining each shell, the first being 0.
    fsc: double array
        The cross-correlation between the volumes at each shell.
    shell_points: int array    
        The number of points in each shell.
    
    Return the resolution and the FSC at that resolution."""

    # Calculate the radius in the Fourier domain.
    x_freq = jnp.fft.fftfreq(int(grid[1]), 1/(grid[0]*grid[1]))
    X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Max radius so that the shells are not outside the
    # rectangular domain.
    max_rad = jnp.max(r[:,0,0])

    # "Calculate" dr if not given:
    # The diagnoal of the smallest cube with a corner at the origin
    if dr is None:
        dr = r[1,1,1]

    # Calculate the shells.
    # Each shell has thickness = dr, with res storing the smallest radial coordinate 
    # of the shell (along the radius). The first shell has radius is manually set to 0.
    # Chose the lowest coordinate of the shell instead of the highest one so that the
    # estimates are conservative rather than optimisic
    s1 = []
    s2 = []
    res = []
    R = -dr/2
    while R + dr <= max_rad:
        cond = ((r >= R) & (r < R + dr))
        s1.append(v1[cond])
        s2.append(v2[cond])
        res.append(R)
        R += dr
    res[0] = 0

    # The correlations between corresponding shells.
    fsc = []
    for i in range(len(s1)):
        f = jnp.sum(s1[i] * jnp.conj(s2[i])) / (jnp.linalg.norm(s1[i],2) * jnp.linalg.norm(s2[i],2))
        fsc.append(f)

    res = jnp.array(res)
    fsc = jnp.real(jnp.array(fsc))
    shell_points = jnp.array([len(s) for s in s1])

    return jnp.array([res, fsc, shell_points])


def average_shells(v, grid, dr = None):
    """Calculate the average of the values in each shell, where the shells
    are computed in the same way as for the FSC function above.

    Parameters:
    ----------
    v :  (N x N x N) array
        Fourier volumes to compute the FSC between.
    grid: [dx, N] 
        The Fourier grid the volumes are defined on.
        dx is the spacing and N is the number of points of the grid.
    dr: double 
        The width of each Fourier shell. 
    Returns:
    -------
    res: double array
        The resolutions defining each shell, the first being 0.
    shell_means: double array
        The cross-correlation between the volumes at each shell.
    shell_points: int array    
        The number of points in each shell.
    
    Return the resolution and the FSC at that resolution."""

    # Calculate the radius in the Fourier domain.
    x_freq = jnp.fft.fftfreq(int(grid[1]), 1/(grid[0]*grid[1]))
    X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
    r = jnp.sqrt(X**2 + Y**2 + Z**2)

    # Max radius so that the shells are not outside the
    # rectangular domain.
    max_rad = jnp.max(r[:,0,0])

    # "Calculate" dr if not given
    if dr is None:
        dr = r[1,1,1]

    # Calculate the shells. See the explanation in the FSC function above.
    s = []
    res = []
    R = -dr/2
    while R + dr <= max_rad:
        cond = ((r >= R) & (r < R + dr))
        s.append(v[cond])
        res.append(R)
        R += dr
    res[0] = 0


    # The sums in each shell.
    sums = jnp.array([jnp.sum(si) for si in s])
    maxs = jnp.array([jnp.max(si) for si in s])
    medians = jnp.array([jnp.median(si) for si in s])
    mins = jnp.array([jnp.min(si) for si in s])

    shell_n_pts = jnp.array([len(si) for si in s])
    shell_means = sums/shell_n_pts

    res = jnp.array(res)

    return res, shell_means, shell_n_pts, maxs, medians#, mins


def average_shells_2D(img, grid, dr = None):
    "The same as the above function but only for an image"

    # Calculate the radius in the Fourier domain.
    x_freq = jnp.fft.fftfreq(int(grid[1]), 1/(grid[0]*grid[1]))
    X, Y = jnp.meshgrid(x_freq, x_freq)
    r = np.sqrt(X**2 + Y**2)

    # Max radius so that the shells are not outside the
    # rectangular domain.
    max_rad = jnp.max(r[:,0])
    
    # "Calculate" dr if not given
    if dr is None:
        dr = r[1,1]

    # Calculate the shells.
    s = []
    res = []
    R = -dr/2
    while R + dr <= max_rad:
        cond = ((r >= R) & (r < R + dr))
        s.append(img[cond])
        res.append(R)
        R += dr
    res[0] = 0

        
    # The sums in each shell.
    sums = jnp.array([jnp.sum(si) for si in s])
    shell_n_pts = jnp.array([len(si) for si in s])
    shell_means = sums/shell_n_pts

    res = jnp.array(res)

    return shell_means


def rotate_list(x_grid, angles):
    """Apply the rotate function to a list of angles."""

    rc = jax.vmap(rotate_z0, in_axes = (None, 0))(x_grid, angles)
    return jnp.swapaxes(rc, 1, 2).reshape(-1,3).T


def points_orientations_tri(angles, nx, number_of_batches = 100):
    """Given a list of orientations as angles, return a volume that
    contains, at each entry, the number of times that volume entry is 
    used by the interpolation function for the given orientations.

    We assume the volume has equal size and grid spacing in all dimensions.
    """
    # We set the grid spacing to one as the exact number (determined by
    # pixel size) is irrelevant.
    x_grid = jnp.array([1, nx])

    print("Rotating coordinates")
    rc = rotate_list(x_grid, angles)
    print("Finding point indices")
    _,(_,xyz_idxs) = jax.vmap(find_nearest_eight_grid_points_idx, 
            in_axes = (1,None, None, None))(rc, x_grid, x_grid, x_grid)

    shape = np.array([nx, nx, nx]).astype(np.int64)
    points_v = np.zeros(shape)

    print("Splitting in batches.")
    # If number_of_batches is too high, this
    #        can take a surprisingly long time
    # BECAUSE jnp.array_split loads the images into memory.
    # Use regular np instead.
    xyz_idx_batches = np.array_split(xyz_idxs, number_of_batches)

    print("Adding up number of points from batches.")
    # This needs to be balanced
    # carefully with the amount of GPU memory available.
    # We want the number of batches to be as small as possible, but not so  
    # small that we cannot allocate enough memory for one batch on the GPU. 
    po_func = get_points_orientations_to_vol_tri_one(nx)
    for xyz_idx_batch in xyz_idx_batches:
        rr = jnp.sum(jax.vmap(po_func, in_axes=0)(xyz_idx_batch), axis=0)

        # The same as the line above, without jax magic for debugging
        #rr = jnp.sum(
        #    jnp.array([points_orientations_to_vol_tri_one(xyz_idx) for xyz_idx in xyz_idx_batch]),
        #axis=0)

        points_v += rr

    return jnp.array(points_v)


# Closure to return the below function without jit issues if 
# nx was an argument
def get_points_orientations_to_vol_tri_one(nx):
    po_func = lambda xyz_idx : points_orientations_to_vol_tri_one(xyz_idx, nx)
    return jax.jit(po_func)

def points_orientations_to_vol_tri_one(xyz_idx, nx):
    # Note that x and y indices are swapped in vol, 
    # similar to the tri_interp_point funtion.
    # i.e. to obtain vol(x, y, z), call vol[y_idx, x_idx, z_idx]

    points_v = jnp.zeros([nx, nx, nx])
    points_v = points_v.at[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,0]].set(1)
    points_v = points_v.at[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,1]].set(1) 
    points_v = points_v.at[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,0]].set(1)
    points_v = points_v.at[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,1]].set(1)
    points_v = points_v.at[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,0]].set(1)
    points_v = points_v.at[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,1]].set(1)
    points_v = points_v.at[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,0]].set(1)
    points_v = points_v.at[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,1]].set(1)

    return points_v

def points_orientations_tri_iter(angles, nx):
    """As above, without vmap."""

    # We set the grid spacing to one as the exact number (determined by
    # pixel size) is irrelevant.
    x_grid = jnp.array([1, nx])

    shape = np.array([nx, nx, nx]).astype(np.int64)
    points_v = np.zeros(shape)

    for ang in angles:
        rc = rotate_z0(x_grid, ang).T

        for c in rc:
            _,(_,xyz_idx) = find_nearest_eight_grid_points_idx(c, x_grid, x_grid, x_grid) 

            # Note that x and y indices are swapped in vol, 
            # similar to the tri_interp_point funtion.
            # i.e. to obtain vol(x, y, z), call vol[y_idx, x_idx, z_idx]

            points_v[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,0]] += 1
            points_v[xyz_idx[1,0], xyz_idx[0,0], xyz_idx[2,1]] += 1 
            points_v[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,0]] += 1
            points_v[xyz_idx[1,1], xyz_idx[0,0], xyz_idx[2,1]] += 1
            points_v[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,0]] += 1
            points_v[xyz_idx[1,0], xyz_idx[0,1], xyz_idx[2,1]] += 1
            points_v[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,0]] += 1
            points_v[xyz_idx[1,1], xyz_idx[0,1], xyz_idx[2,1]] += 1

    return jnp.array(points_v)



def points_orientations_nn(angles, nx):
    """Same as points_orientations_tri but for nearest neighbour
    interpolation."""

    x_grid = jnp.array([1, nx])
    
    rc = rotate_list(x_grid, angles)
    xyz_idxs = jax.vmap(find_nearest_one_grid_point_idx, 
            in_axes = (1,None, None, None))(rc, x_grid, x_grid, x_grid)

    shape = np.array([nx, nx, nx]).astype(np.int64)
    points_v = np.zeros(shape)

    for xyz_idx in xyz_idxs:
        points_v[tuple(xyz_idx)] += 1

    return jnp.array(points_v)


def points_orientations_star(data_dir, star_file, nx = -1, method = "tri", out_file = None):
    """Call points_orientations_* with the orientations taken 
    from a starfile. If nx is not given, it loads the images form the mrcs file
    to obtain nx.""" 

    if nx == -1:
        params, imgs_f = load_data(data_dir, star_file, load_imgs = True)
        nx = imgs_f.shape[1]
    else:
        params, _ = load_data(data_dir, star_file, load_imgs = False)

    angles = params["angles"]

    # It works with 1000 angles for 256 x 256 imgs, it crashes at more
    # angles, at least on pi_lederman.
    if method == "tri":
        pts = points_orientations_tri(angles, nx)
    elif method == "nn":
        pts = points_orientations_nn(angles, nx)

    if out_file is not None:
        with mrcfile.new('out_file', overwrite=True) as mrc:
            mrc.set_data(pts.astype(np.float32))

    return pts 



def shell_points_used(points, grid, dr = None):
    """Given a volume containing the number of orientations that uses each 
    point (as output by the points_orientations_tri and points_orientations_nn 
    functions), using the standard Fourier ordering, 
    calculate the total number of used points 
    in each spherical Fourier shell, normalised by the total number of points
    in each shell respectively."""

    # Calculate the radius in the Fourier domain.
    x_freq = jnp.fft.fftfreq(int(grid[1]), 1/(grid[0]*grid[1]))
    X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Max radius so that the shells are not outside the
    # rectangular domain.
    max_rad = jnp.max(r[:,0,0])

    # "Calculate" dr if not given
    if dr is None:
        dr = r[1,1,1]

    # Calculate the shells.
    shells = []
    res = []
    R = -dr
    while R + dr <= max_rad:
        cond = ((r >= R) & (r < R + dr))
        shells.append(points[cond])
        res.append(R+dr)
        R += dr

    res = jnp.array(res)
    shell_points_used = jnp.array([len(s[s > 0]) for s in shells])
    shell_points_total = jnp.array([len(s) for s in shells])

    return res, jnp.divide(shell_points_used, shell_points_total) 







