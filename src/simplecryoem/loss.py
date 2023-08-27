import jax
import jax.numpy as jnp
from functools import partial
from simplecryoem.utils import l2sq, wl2sq
from simplecryoem.forwardmodel import Slice


class Loss:
    """The loss function, batched and sum.

    Attributes:
    -----------
    slice_obj:
        An instance of the Slice class defined above.

    err_func :
        Function to calculate the error between two images,
        defaults to l2 squared.

    alpha :
        Regularisation parameter for l2 regularisation.

    Methods:
    -------
    loss:

    loss_batched:

    loss_sum:
    """

    def __init__(self, slice: Slice, err_func=wl2sq, alpha=0):
        self.slice = slice
        self.err_func = err_func
        self.alpha = alpha

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, v, angles, shifts, ctf_params, img, sigma=1):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard
        deviation of the noise."""
        return (
            1
            / 2
            * (
                self.alpha * l2sq(v)
                + self.err_func(
                    self.slice.slice(v, angles, shifts, ctf_params), img, 1 / sigma**2
                )
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss0(self, v, angles, shifts, ctf_params, img, sigma=1):
        """Similar to loss function, but with alpha=0 (no regularisation)."""
        return (
            1
            / 2
            * self.err_func(
                self.slice.slice(v, angles, shifts, ctf_params), img, 1 / sigma**2
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_batched(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_batched0(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss0, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jnp.mean(self.loss_batched(v, angles, shifts, ctf_params, imgs, sigma))

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj(self, v, projection, shifts, ctf_params, img, sigma):
        """Loss when the rotation is already done.
        L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard
        deviation of the noise."""

        proj = self.slice.apply_shifts_and_ctf(projection, shifts, ctf_params)
        return 1 / 2 * (self.alpha * l2sq(v) + self.err_func(proj, img, 1 / sigma**2))

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj0(self, v, projection, shifts, ctf_params, img, sigma):
        "The alpha=0 version of loss_proj." ""

        proj = self.slice.apply_shifts_and_ctf(projection, shifts, ctf_params)
        return 1 / 2 * self.err_func(proj, img, 1 / sigma**2)

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj_batched(self, v, projection, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss_proj, in_axes=(None, 0, 0, 0, 0, None))(
            v, projection, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj_batched0(self, v, projection, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss_proj0, in_axes=(None, 0, 0, 0, 0, None))(
            v, projection, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_px(self, v, angles, shifts, ctf_params, img, sigma=1):
        """Return the pixel-wise loss for one image,
        with no sigma and regularization."""
        err = self.slice.slice(v, angles, shifts, ctf_params) - img
        return 1 / 2 * jnp.real(jnp.conj(err) * err)

    @partial(jax.jit, static_argnums=(0,))
    def loss_px_batched(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss_px, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_px_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jnp.mean(
            self.loss_px_batched(v, angles, shifts, ctf_params, imgs, sigma), axis=0
        )


class GradV:
    """Gradient of the loss function with respect to the volume."""

    def __init__(self, loss: Loss):
        self.loss = loss

    @partial(jax.jit, static_argnums=(0,))
    def grad_loss_volume(self, v, angles, shifts, ctf_params, img, sigma):
        return jax.grad(self.loss.loss)(v, angles, shifts, ctf_params, img, sigma)

    @partial(jax.jit, static_argnums=(0,))
    def grad_loss_volume_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.grad(self.loss.loss_sum)(v, angles, shifts, ctf_params, imgs, sigma)
