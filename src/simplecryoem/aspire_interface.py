import numpy as np
import jax.numpy as jnp
from simplecryoem.utils import create_2d_mask
from aspire.source import Simulation


def get_params_from_aspire(src: Simulation, pixel_size):
    """Function that takes an ASPIRE ImageSource object and pixel size
    and extracts the images, angles, shifts, ctf_params in the right format
    for JaxEM, plus a few other useful things for further JaxEM processing."""
    n = src.n
    imgs = np.array(src.images(0, n).data)
    imgs_f = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs])
    nx = imgs.shape[1]

    angles = (
        src.angles
    )  # "_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi" in radians -> I store psi, tilt,rot
    shifts = src.offsets

    myshifts = jnp.flip(shifts, axis=1) * pixel_size
    myangles = -jnp.flip(angles, axis=1)

    cf = src.unique_filters
    myctf_params = jnp.array(
        [
            jnp.array(
                [
                    cf[ci].defocus_u,
                    cf[ci].defocus_v,
                    cf[ci].defocus_ang,
                    0,
                    cf[ci].voltage,
                    cf[ci].alpha,
                    cf[ci].Cs,
                    0,
                    2 * pixel_size,
                ]
            )
            for ci in src.filter_indices
        ]
    )
    x_grid = np.array([1 / (nx * pixel_size), nx])
    radius = x_grid[0] * (x_grid[1] / 2 - 1)
    mask = create_2d_mask(x_grid, [0, 0], radius)

    return imgs, imgs_f, myangles, myshifts, myctf_params, x_grid, mask
