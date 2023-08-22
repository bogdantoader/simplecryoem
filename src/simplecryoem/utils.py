import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from matplotlib import pyplot as plt


def volume_comp(
    shape, dimensions, centres, radii, intensities, apply_filter=False, sigma=0.01
):
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
    vol = sum(
        map(
            lambda cr: spherical_volume(
                shape, dimensions, cr[0], cr[1], cr[2], False, apply_filter, sigma
            ),
            zip(centres, radii, intensities),
        )
    )
    vol = vol / jnp.max(vol)

    return vol


def spherical_volume(
    shape,
    dimensions,
    centre,
    radius,
    intensity,
    rand_or_not,
    apply_filter=False,
    sigma=0.01,
):
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
    dx, dy, dz = dimensions / shape  # "pixel" size

    # By adjusting the interval by half a pixel on each side
    # we ensure the sampling locations are
    # the centres of the "pixels"
    coords_x = np.linspace(-Lx / 2 + dx / 2, Lx / 2 - dx / 2, Nx)
    coords_y = np.linspace(-Ly / 2 + dy / 2, Ly / 2 - dy / 2, Ny)
    coords_z = np.linspace(-Lz / 2 + dz / 2, Lz / 2 - dz / 2, Nz)
    x_grid = np.array([coords_x[1] - coords_x[0], len(coords_x)])

    vol = vol * create_3d_mask(x_grid, centre, radius)

    if apply_filter:
        vol = low_pass_filter(vol, X, Y, Z, sigma)

    vol = np.fft.fftshift(vol)

    return vol


def plot_angles(angs):
    """Display a list of Euler angles as points on a sphere."""

    # A sphere
    phi = jnp.linspace(0, jnp.pi)
    theta = jnp.linspace(0, 2 * jnp.pi)
    Phi, Theta = jnp.meshgrid(phi, theta)
    x = jnp.cos(Phi) * jnp.sin(Theta)
    y = jnp.sin(Phi) * jnp.sin(Theta)
    z = jnp.cos(Theta)

    # Get point coordinates
    coords = jnp.array(
        [get_rotation_matrix(a[0], a[1], a[2]) @ jnp.array([0, 0, 1.1]) for a in angs]
    )
    xx, yy, zz = jnp.hsplit(coords, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.6, linewidth=0)
    ax.scatter(xx, yy, zz, color="k", s=20)
    return


def get_preconditioner(x_grid):
    x_freq = np.fft.fftfreq(x_grid[1].astype(np.int64), 1 / (x_grid[1] * x_grid[0]))
    y_freq = x_freq
    z_freq = x_freq

    X, Y, Z = np.meshgrid(x_freq, z_freq, y_freq)

    return X**2 + Y**2 + Z**2


def get_sinc(x_grid):
    x_freq = np.fft.fftfreq(x_grid[1].astype(np.int64), 1 / (x_grid[1] * x_grid[0]))
    y_freq = x_freq
    z_freq = x_freq

    X, Y, Z = np.meshgrid(x_freq, z_freq, y_freq)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = (
            np.sin(np.pi * X)
            * np.sin(np.pi * Y)
            * np.sin(np.pi * Z)
            / (X * Y * Z * np.pi**3)
        )
    res[0, 0, 0] = 1

    return jnp.array(res)


def create_3d_mask(x_grid, centre, radius):
    """Works in the Fourier domain with the standard ordering, but obviously
    it can be applied in the spatial domain too."""

    x_freq = np.fft.fftfreq(x_grid[1].astype(np.int64), 1 / (x_grid[1] * x_grid[0]))
    y_freq = x_freq
    z_freq = x_freq

    X, Y, Z = np.meshgrid(x_freq, z_freq, y_freq)

    mask = np.ones(X.shape)
    cx, cy, cz = centre
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
    mask[r > radius] = 0

    return np.array(mask)


def create_2d_mask(x_grid, centre, radius):
    """Works in the Fourier domain with the standard ordering, but obviously
    it can be applied in the spatial domain too.
    Always centered at zero."""

    x_freq = np.fft.fftfreq(x_grid[1].astype(np.int64), 1 / (x_grid[1] * x_grid[0]))
    y_freq = x_freq

    X, Y = np.meshgrid(x_freq, y_freq)

    mask = np.ones(X.shape)
    cx, cy = centre
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask[r > radius] = 0

    return np.array(mask)


def low_pass_filter(vol, X, Y, Z, sigma):
    gauss = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma))
    gauss = gauss / max(gauss.ravel())
    gauss = np.fft.fftshift(gauss)

    low_pass_vol = np.fft.ifftn(np.fft.fftn(vol) * gauss)
    return np.real(low_pass_vol)


def volume_fourier(vol, pixel_size, pfac=1):
    """Calculate the FFT of the volume and return the frequency coordinates
    Assume the volume has equal dimensions and is centred.
    If pfac > 1, the input is zero padded before taking the FFT.
    The resulting Fourier volume is in the standard Fourier ordering.

    Parameters
    ----------
    vol : n x n x n array
        Volume in spatial domain.
    pixel_size: double
        Pixel size, in units (e.g. Angst)
    pfac : int
        Padding factor.
    Returns
    -------
    vol_f: n x n x n array
        The FFT of the volume vol.
    grid_vol: [grid_spacing, grid_length]   2 x 1 array
        Fourier grid that vol_f is defined on.
    grid_vol_nopad: [grid_spacing, grid_length]   2 x 1 array
        Fourier grid vol_f was defined on if pfac = 1.
    """

    n = vol.shape[0]

    # We do the padding explicitly as opposed to calling fft with larger size
    # because funky things happen when padding a non-centred image.
    pad = n * (pfac - 1) / 2
    if jnp.mod(pad, 2) == 0:
        # In this case, add the same number of zeros to each side.
        vol = jnp.pad(vol, int(pad))
    else:
        # If the number of total zeros to add in each dimension is odd,
        # add the zeros so that when ifftshifted, the center of the image
        # is at the zero index.
        pad = int(jnp.floor(pad))
        vol = jnp.pad(vol, (pad + 1, pad))

    vol_f = jnp.fft.fftn(jnp.fft.ifftshift(vol))
    grid_vol = create_grid(vol_f.shape[0], pixel_size)
    grid_vol_nopad = create_grid(n, pixel_size)

    return vol_f, grid_vol, grid_vol_nopad


def mip_x(img):
    plt.imshow(np.max(img, axis=0))
    return


def mip_y(img):
    plt.imshow(np.max(img, axis=1))
    return


def mip_z(img):
    plt.imshow(np.max(img, axis=2))
    return


# TODO: tests for the three functions below,
# and change the names of these functions, they're not great
def rescale_smaller_grid(v, x_grid, y_grid, z_grid, radius):
    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1 / (x_grid[0] * x_grid[1]))
    y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1 / (y_grid[0] * y_grid[1]))
    z_freq = jnp.fft.fftfreq(int(z_grid[1]), 1 / (z_grid[0] * z_grid[1]))
    X, Y, Z = jnp.meshgrid(x_freq, y_freq, z_freq)

    idx = jnp.sqrt(X**2 + Y**2 + Z**2) <= radius

    idx1 = idx[0, :, 0]
    idx2 = idx[:, 0, 0]
    idx3 = idx[0, 0, :]

    len_x = len(x_freq[idx1])
    len_y = len(y_freq[idx2])
    len_z = len(z_freq[idx3])

    v_c = v[idx1][:, idx2][:, :, idx3]

    x_grid_c = [x_grid[0], len_x]
    y_grid_c = [y_grid[0], len_y]
    z_grid_c = [z_grid[0], len_z]

    return v_c, x_grid_c, y_grid_c, z_grid_c


def get_pad_width(l0, l1):
    """Determine the padding width that a centred object of length l0 in the
    Fourier domain should be padded with so that the final length is l1."""

    if jnp.mod(l1 - l0, 2) == 0:
        w = (l1 - l0) / 2
        pad_width = [w, w]
    else:
        w = jnp.floor((l1 - l0) / 2)

        if jnp.mod(l0, 2) == 0:
            pad_width = [w, w + 1]
        else:
            pad_width = [w + 1, w]
    return jnp.array(pad_width).astype(jnp.int64)


def rescale_larger_grid(v, x_grid, nx_new):
    """Zero pad Fourier volume v to obtain a larger volume
    of dimension nx_new."""

    pad_width_x = get_pad_width(x_grid[1], nx_new)
    v_new = jnp.pad(jnp.fft.fftshift(v), pad_width_x)
    v_new = jnp.fft.ifftshift(v_new)

    x_grid_new = [x_grid[0], nx_new]

    return v_new, x_grid_new


# TODO : write tests for this function and make with work with odd dimensions too.
def crop_fourier_images(imgs, x_grid, nx_new):
    """Given an N x nx0 x nx0 array of N images of dimension nx0 x nx0 in the
    frequency space with the standard ordering, crop the high-frequency entries
    to reduce the image to the dimensions nx_new x nx_new.
    Also adjust the grid arrays accordingly.

    Parameters:
    ----------
    imgs : N x nx0 x nx0 array
        N stacked images of dimensions nx0 x nx0 in the Fourier domain
        and standard ordering.
    x_grid: 2 x 1 array
        Spacing and length of the Fourier grid in each dimension (we assume
        they are the same in all dimensions), in the format:
        [grid_spacing, grid_length].
    nx_new : integer
        The target length each dimension of the images after cropping.

    Returns:
    -------
        imgs_cropped: N x nx_new x nx_new)
            N stacked cropped images.
        x_grid_cropped: 2 x 1 array
            The new Fourier grid corresponding to the cropped images.
    """

    N = imgs.shape[0]
    mid = imgs.shape[-1] / 2

    idx = np.concatenate([np.arange(nx_new / 2), np.arange(-nx_new / 2, 0)]).astype(
        np.int64
    )
    imgs_cropped = imgs[np.ix_(np.arange(N), idx, idx)]

    # <<< IMPORTANT!!!>>>
    # The grid must not be a Jax object.
    x_grid_cropped = np.array([x_grid[0], nx_new])

    return imgs_cropped, x_grid_cropped


def crop_fourier_images_batch(imgs, x_grid, nx_new):
    """Here the images are in batches so
    imgs has dimension N1 x N2 x nx0 x nx0."""

    imgs_c = []
    for i in np.arange(imgs.shape[0]):
        imgs_c_i, x_grid_cropped = crop_fourier_images(imgs[i], x_grid, nx_new)
        imgs_c.append(imgs_c_i)

    return np.array(imgs_c), x_grid_cropped


def crop_fourier_volume(vol, x_grid, nx_new):
    """Same as above, but a volume."""

    vol = np.fft.fftshift(vol)
    mid = vol.shape[-1] / 2

    vol_cropped = np.fft.ifftshift(
        vol[
            int(mid - nx_new / 2) : int(mid + nx_new / 2),
            int(mid - nx_new / 2) : int(mid + nx_new / 2),
            int(mid - nx_new / 2) : int(mid + nx_new / 2),
        ]
    )

    # <<< IMPORTANT!!!>>>
    # The grid must not be a Jax object.
    x_grid_cropped = np.array([x_grid[0], nx_new])

    return vol_cropped, x_grid_cropped


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
    r = jnp.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ]
    )
    return r


def generate_uniform_orientations(N):
    alpha = np.random.rand(N, 1) * 2 * np.pi
    gamma = np.random.rand(N, 1) * 2 * np.pi
    z = np.random.rand(N, 1) * 2 - 1
    beta = np.arccos(z)
    angles = np.concatenate([alpha, beta, gamma], axis=1)
    return jnp.array(angles)


def generate_uniform_orientations_jax(key, a0):
    key1, key2, key3 = random.split(key, 3)
    N = a0.shape[0]

    alpha = random.uniform(key1, (N,)) * 2 * jnp.pi
    gamma = random.uniform(key2, (N,)) * 2 * jnp.pi
    z = random.uniform(key3, (N,)) * 2 - 1
    beta = jnp.arccos(z)
    return jnp.array([alpha, beta, gamma]).transpose()


def generate_perturbed_orientations(key, a0, sig):
    return a0 + sig * random.normal(key, a0.shape)


def generate_uniform_orientations_jax_batch(key, N1, N2):
    keys = random.split(key, N1)
    angles = []
    a0 = jnp.zeros(N2)

    for i in jnp.arange(N1):
        angles.append(generate_uniform_orientations_jax(keys[i], a0))

    return jnp.array(angles)


def generate_uniform_shifts(key, N, B):
    """Generate uniformly sampled shifts in [-B,B] x [-B,B]."""

    return random.uniform(key, (N, 2)) * 2 * B - B


def generate_gaussian_shifts(key, N, B):
    """Generate normally sampled shifts around (0,0)
    with standard deviation B."""

    return random.normal(key, (N, 2)) * B


def generate_gaussian_shifts_batch(key, N1, N2, B):
    """Generate normally sampled shifts around (0,0)
    with standard deviation B."""

    return random.normal(key, (N1, N2, 2)) * B


@jax.jit
def l2sq(x, y=0):
    return jnp.real(jnp.sum(jnp.conj(x - y) * (x - y)))


@jax.jit
def wl2sq(x, y=0, w=1):
    """Weighted l2 squared norm/error."""
    return jnp.real(jnp.sum(w * jnp.conj(x - y) * (x - y)))


def create_grid(nx, px):
    """Create the (one dimensional) Fourier grid used for projections.

    <<<IMPORTANT!!!>>>
    The grids must not be Tracer (aka Jax)  objects."""

    x_freq = np.fft.fftfreq(nx, px)
    x_grid = np.array([x_freq[1], len(x_freq)])

    return x_grid


def err_orientations(angles1, angles2):
    """Calculate the angular distances between the orientations
    a1_i and a2_i from the N x 3 arrays angles1 and angles2."""

    # Making get_rotation_matrix function vmap friendly
    get_rot_mats = jax.vmap(lambda a: get_rotation_matrix(a[0], a[1], a[2]), in_axes=0)

    # Get the matrix for each orientation
    M1 = get_rot_mats(angles1)
    M2 = get_rot_mats(angles2)

    # The chordal distance (Frobenium norm error between matrices)
    err = jnp.sqrt(jnp.sum((M1 - M2) ** 2, axis=(1, 2)))

    # The actual angular distance
    theta = 2 * jnp.arcsin(err / (2 * jnp.sqrt(2)))

    return theta, err


@jax.jit
def wrap_around_distance_2d(x1, x2, B):
    """The wrap around distance between x1 and x2 on the square [-B, B]^2."""
    d = jnp.abs(x1 - x2)
    d = jnp.abs(jnp.minimum(d, 2 * B - d))
    return jnp.sqrt(jnp.sum(d**2))


@jax.jit
def wrap_arround_distance_2d_array(x1, x2, B):
    return jax.vmap(wrap_around_distance_2d, in_axes=(0, 0, None))(x1, x2, B)
