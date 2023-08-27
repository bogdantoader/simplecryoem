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
    in the Fourier domain, as jit-compiled JAX functions."""

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
        return jax.vmap(self.slice, in_axes=(None, 0, 0, 0))(
            v, angles, shifts, ctf_params
        )

    @partial(jax.jit, static_argnums=(0,))
    def slice_array_angles(self, v, angles, shifts, ctf_params):
        """Same as above, except the shifts and ctf_params are
        fixed and we don't vectorize them."""
        return jax.vmap(self.slice, in_axes=(None, 0, None, None))(
            v, angles, shifts, ctf_params
        )

    @partial(jax.jit, static_argnums=(0,))
    def apply_shifts_and_ctf(self, projection, shifts, ctf_params):
        return self.apply_shifts_and_ctf_func(
            projection, shifts, ctf_params, self.x_grid
        )

    @partial(jax.jit, static_argnums=(0,))
    def rotate_and_interpolate_vmap(self, v, angles):
        return jax.vmap(self.rotate_and_interpolate, in_axes=(None, 0, None, None))(
            v * self.mask, angles, self.x_grid, self.x_grid
        )
