import jax
import jax.numpy as jnp
from functools import partial

from . import (
    project,
    apply_shifts_and_ctf,
    rotate_and_interpolate,
)


class Slice:
    """Class encapsulating useful projection-related operations
    in the Fourier domain, as jit-compiled JAX functions.

    Instantiate this class with a given x_grid, mask, and
    interpolation and projection-related functions to benefit
    from JAX compilation and vectorization.

    Attributes:
    -----------
    x_grid: [grid_spacing, grid_length]
        Fourier grid that vol is defined on.

    mask: nx x nx array
        Image-shaped mask to apply to the error when computing
        the loss.

    project:
        Projection function, defaults to forwardmodel.project.

    rotate_and_interpolate:
        Rotate and interpolate function, defaults to
        forwardmodel.rotate_and_interpolate.

    apply_shifts_and_ctf:
        Apply shifts and CTF function, defaults to
        forwardmodel.apply_shifts_and_ctf.

    interp_method: string
        Interpolation method, "tri" or "nn", defaults to "tri".
    """

    def __init__(
        self,
        x_grid,
        mask=None,
        project=project,
        rotate_and_interpolate=rotate_and_interpolate,
        apply_shifts_and_ctf=apply_shifts_and_ctf,
        interp_method="tri",
    ):
        self.project = project
        self.x_grid = x_grid
        self.interp_method = interp_method

        if mask is None:
            nx = int(x_grid[1])
            self.mask = jnp.ones([nx, nx])
        else:
            self.mask = mask

        self.project = project
        self.rotate_and_interpolate = rotate_and_interpolate
        self.apply_shifts_and_ctf_func = apply_shifts_and_ctf

    @partial(jax.jit, static_argnums=(0,))
    def slice(self, v, angles, shifts, ctf_params):
        """Take a slice of the volume in the Fourier domain
        to obtain the FFT of the corresponding projection.

        A jit-compiled version of the project function with
        additional mask applied.
        For more details, see the documentation of project.
        """

        return self.project(
            v * self.mask,
            angles,
            shifts,
            ctf_params,
            self.x_grid,
            self.x_grid,
            self.interp_method,
        )

    @partial(jax.jit, static_argnums=(0,))
    def slice_array(self, v, angles, shifts, ctf_params):
        """A vmap-ed version of the slice function.

        Vectorization along angles, shifts, ctf_params.
        """

        return jax.vmap(self.slice, in_axes=(None, 0, 0, 0))(
            v, angles, shifts, ctf_params
        )

    @partial(jax.jit, static_argnums=(0,))
    def slice_array_angles(self, v, angles, shifts, ctf_params):
        """Same as slice_array, except that the shifts and ctf_params
        are fixed and we don't vectorize them."""

        return jax.vmap(self.slice, in_axes=(None, 0, None, None))(
            v, angles, shifts, ctf_params
        )

    @partial(jax.jit, static_argnums=(0,))
    def apply_shifts_and_ctf(self, projection, shifts, ctf_params):
        """Jit-compiled version of the apply_shifts_and_ctf function."""

        return self.apply_shifts_and_ctf_func(
            projection, self.x_grid, shifts, ctf_params,
        )

    @partial(jax.jit, static_argnums=(0,))
    def rotate_and_interpolate_vmap(self, v, angles):
        """Jit-compiled and vectorized version of the rotate_and_interpolate
        function, with the vectorization based on the angles."""

        return jax.vmap(self.rotate_and_interpolate, in_axes=(None, 0, None, None))(
            v * self.mask, angles, self.x_grid, self.x_grid
        )
