import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from pyem.vop import grid_correct
from pyem import star

from simplecryoem.ctf import eval_ctf
from . import interpolate
from simplecryoem.utils import (
    volume_fourier,
    create_2d_mask,
    create_3d_mask,
    get_rotation_matrix,
)

config.update("jax_enable_x64", True)


def project_spatial(
    v, angles, pixel_size, shifts=[0, 0], method="tri", ctf_params=jnp.array([jnp.nan]), pfac=1
):
    """Takes a centred object in the spatial domain and returns the centred
    projection in the spatial domain.
    If N is the number of pixels in one dimension, then the origin is
    assumed to be in the pixel with index (N-1)/2 if N is odd
    and N/2 is N is even, similar to an fftshifted Fourier grid.

    Parameters:
    -----------
    v:
        Volume in spatial domain.

    angles: [psi, tilt, rot]
        Proper Euler angles.

    pixel_size: double
        Pixel size in Angstrom.

    shifts : [originx, originy]

    method: "tri" or "nn"
        Interpolation method.

    ctf_params : 9 x 1 array or None
        As in the ctf file.

    pfac : int
        Factor to multiply the interpolation grid in Fourier domain
        for finer interpolation. Does not affect the dimensions of
        the output projection.

    Returns:
    --------
    v_proj:
         The projection of the volume after rotation, interpolation,
         shifts and CTF.
    """

    V, grid_vol, grid_proj = volume_fourier(v, pixel_size, pfac)

    V_slice = project(V, angles, shifts, ctf_params,
                      grid_vol, grid_proj, method)

    # Make it 2D
    V_slice = V_slice.reshape(int(grid_proj[1]), int(grid_proj[1]))

    # Back to spatial domain
    v_proj = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(V_slice)))

    return v_proj

# TODO: New parameters order: vol, grid_vol, grid_proj, angles, shifts, ctf_params.
# Then I can give default values for shifts and CTF params.


def project(
    vol, angles, shifts, ctf_params, grid_vol, grid_proj, interpolation_method="tri"
):
    """Projection in the Fourier domain. Defining separate grid_vol (the grid
    on which the given 3D volume is defined) and grid_proj (the grid on which
    the 2D projection is defined) allows us to implement Relion-style padding
    (i.e. Oversample the Fourier transform to interpolate on, but the dimensions
    of the projection are equal to those of the initial volume).

    Assumption 1: the volume has equal size in all dimensions.
    Assumption 2: the frequencies are in the 'standard' order for vol.

    Parameters:
    -----------
    vol:
        Volume in Fourier domain, in standard order.

    angles: [psi, tilt, rot]
        Proper Euler angles.

    shifts : [originx, originy]

    ctf_params : 9 x 1 array or None
        As in the ctf file.

    grid_vol : [grid_spacing, grid_length]
        Fourier grid that vol is defined on.

    grid_proj: [grid_spacing, grid_length]
        Fourier grid that the projection is defined on.
        It is distinct from grid_vol when pfac > 1.

    interpolation_method : "tri" or "nn"

    Returns:
    --------
    proj:
         The projection of the volume after rotation, interpolation,
         shifts and CTF.
    """

    # Get the rotated coordinates of the projection in the z=0 plane.
    proj = rotate_and_interpolate(
        vol, angles, grid_vol, grid_proj, interpolation_method=interpolation_method
    )

    # Apply the shift and the CTF.
    proj = apply_shifts_and_ctf(proj, grid_proj, shifts, ctf_params)

    return proj


def rotate_and_interpolate(
    vol, angles, grid_vol, grid_proj, interpolation_method="tri"
):
    """Apply the rotation and interpolation of the z=0 Fourier plane."""

    # Get the rotated coordinates of the projection in the z=0 plane.
    proj_coords = rotate_z0(grid_proj, angles)
    return interpolate(proj_coords, grid_vol, vol, interpolation_method)


def apply_shifts_and_ctf(proj, grid_proj, shifts=jnp.array([0.0, 0.0]), ctf_params=jnp.array([jnp.nan])):
    """Apply the shifts and CTF to an image in the Fourier domain.

    To apply no CTF, make ctf_params jnp.array([jnp.nan]) or np.array([None])."""

    shift = get_shift_term(grid_proj, grid_proj, shifts)
    proj *= shift

    def compute_ctf():
        x_freq = jnp.fft.fftfreq(
            int(grid_proj[1]), 1 / (grid_proj[0] * grid_proj[1]))
        X, Y = jnp.meshgrid(x_freq, x_freq)
        r = jnp.sqrt(X**2 + Y**2)
        theta = jnp.arctan2(Y, X)

        return eval_ctf(r, theta, ctf_params).astype(jnp.complex128).ravel()

    ctf = jax.lax.cond(
        jnp.isnan(ctf_params[0]),
        true_fun=lambda _: jnp.ones_like(proj),
        false_fun=lambda _: compute_ctf(),
        operand=None
    )

    proj *= ctf

    # return proj, proj_coords
    return proj


def rotate_z0(grid_proj, angles):
    """Rotate the coordinates X, Y, Z=0
    obtained using grid_proj
    with Euler angles alpha, beta, gamma
    around axes x, y, z respectively.

    Parameters
    ----------
    grid_proj : [grid_spacing, grid_length]
        The grid spacing and grid size of the Fourier 1D grids on which
        the we project. The full grids can be obtained by running
        x_freq = np.fft.fftfreq(grid_length, 1/(grid_length*grid_spacing)).
    angles:  3 x 1 array
        [alpha, beta, gamma] Euler angles

    Returns
    -------
    rotated_coords : N x 2
        Array of the rotated coorinates.
    """

    # Change later if too inefficient. Do not use itertools, it triggers
    # an XLA 'too slow' bug when calling with jit.

    # Generate the x and y grids.
    x_freq = jnp.fft.fftfreq(
        int(grid_proj[1]), 1 / (grid_proj[0] * grid_proj[1]))

    X, Y = jnp.meshgrid(x_freq, x_freq)
    coords = jnp.array(
        [X.ravel(), Y.ravel(), jnp.zeros(len(x_freq) * len(x_freq))])

    angles = -jnp.array(angles)

    # And finally rotate
    rotated_coords = get_rotation_matrix(*angles) @ coords

    return rotated_coords


def rotate_full_grid(grid_proj, angles):
    """The same as rotate_z0, but rotate the full grid instead of
    only the z0 plane."""

    x_freq = jnp.fft.fftfreq(
        int(grid_proj[1]), 1 / (grid_proj[0] * grid_proj[1]))

    X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
    coords = jnp.array([X.ravel(), Y.ravel(), Z.ravel()])

    angles = -jnp.array(angles)

    # And finally rotate
    rotated_coords = get_rotation_matrix(*angles) @ coords

    return rotated_coords


def get_shift_term(x_grid, y_grid, shifts):
    """Generate the phase term corresponding to the shifts in
    units (e.g. Angstroms)."""

    # Generate the x and y grids.
    x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1 / (x_grid[0] * x_grid[1]))
    y_freq = jnp.fft.fftfreq(int(y_grid[1]), 1 / (y_grid[0] * y_grid[1]))

    X, Y = jnp.meshgrid(x_freq, y_freq)
    shift = jnp.exp(2 * jnp.pi * 1j * (X * shifts[0] + Y * shifts[1]))

    return shift.ravel()


def project_star_params(vol, p, pfac=1):
    """Spatial domain projection of vol with parameters from one row of
    a star file given by the dictionary p.

    Useful to compare against Relion and pyem.
    We assume vol is the same size in all dimensions."""

    vol = grid_correct(vol, pfac=pfac, order=1)

    # TODO: uncomment the 64/66 and check comparison with Pyem
    pixel_size = star.calculate_apix(p)  # * 64.0/66.0

    f3d, grid_vol, grid_proj = volume_fourier(vol, pixel_size, pfac)

    mask_radius = jnp.prod(grid_vol) / 2
    mask_vol = create_3d_mask(grid_vol, (0, 0, 0), mask_radius)
    f3d = f3d * mask_vol

    angles = (
        jnp.array(
            [p[star.Relion.ANGLEPSI], p[star.Relion.ANGLETILT], p[star.Relion.ANGLEROT]]
        )
        / 180
        * jnp.pi
    )  # third angle is
    # rotation around the first z axis

    shifts = jnp.array(
        [p[star.Relion.ORIGINX], p[star.Relion.ORIGINY]]) * pixel_size

    #    ctf_params = {'def1'  : p[star.Relion.DEFOCUSU],
    #                  'def2'  : p[star.Relion.DEFOCUSV],
    #                  'angast': p[star.Relion.DEFOCUSANGLE],
    #                  'phase' : p[star.Relion.PHASESHIFT],
    #                  'kv'    : p[star.Relion.VOLTAGE],
    #                  'ac'    : p[star.Relion.AC],
    #                  'cs'    : p[star.Relion.CS],
    #                  'bf'    : 0,
    #                  'lp'    : 2 * pixel_size}

    # The project function requires the CTF parameters to be a list with
    # the elements ordered like the keys in the dict above.
    ctf_params = [
        p[star.Relion.DEFOCUSU],
        p[star.Relion.DEFOCUSV],
        p[star.Relion.DEFOCUSANGLE],
        p[star.Relion.PHASESHIFT],
        p[star.Relion.VOLTAGE],
        p[star.Relion.AC],
        p[star.Relion.CS],
        0,
        2 * pixel_size,
    ]

    f2d = project(f3d, angles, shifts, ctf_params, grid_vol, grid_proj, "tri")

    mask_radius = jnp.prod(grid_proj) / 2
    mask_proj = create_2d_mask(grid_proj, (0, 0), mask_radius)

    f2d = f2d.reshape(int(grid_proj[1]), int(grid_proj[1])) * mask_proj
    proj = np.real(np.fft.fftshift(np.fft.ifftn(f2d)))

    return proj
