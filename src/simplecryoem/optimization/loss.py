import jax
import jax.numpy as jnp
from functools import partial
from simplecryoem.utils import l2sq, wl2sq
from simplecryoem.forwardmodel import Slice


class Loss:
    """The loss function, batched and sum.

    Attributes:
    -----------
    slice:
        An instance of the Slice class from forwardmodel.

    err_func :
        Function to calculate the error between two images,
        defaults to utils.wl2sq.

    alpha :
        Regularisation parameter for l2 regularisation,
        defaults to 0.
    """

    def __init__(self, slice: Slice, err_func=wl2sq, alpha=0):
        self.slice = slice
        self.err_func = err_func
        self.alpha = alpha

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, v, angles, shifts, ctf_params, img, sigma=1):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard
        deviation of the noise.

        More specifically, given volume v, image x, and forward
        operator P (that incorporates rotation, projection, shifts
        and CTF), compute

        1/(2*sigma^2) \| P(v) - x \|^2

        Parameters:
        -----------
        v : nx x nx x nx array
            Volume to apply the loss to
        angles : 1 x 3 array
            Euler angles for the rotation before the projection.
        shifts : 1 x 2 array
            Shifts to apply after projection.
        ctf_params : 1 x 9 array
            CTF parameters
        img : nx x nx array
            The particle image to compute the loss against.
        sigma : double
            Noise standard deviation.

        Returns:
        --------
        The value of the loss function with the given parameters.
        """

        return (
            1
            / 2
            * (
                self.alpha * l2sq(v)
                + self.err_func(
                    self.slice.slice(v, angles, shifts,
                                     ctf_params), img, 1 / sigma**2
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
                self.slice.slice(v, angles, shifts,
                                 ctf_params), img, 1 / sigma**2
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_batched(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Vectorized loss function along angles, shifts, ctf_params, imgs."""
        return jax.vmap(self.loss, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_batched0(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Vectorized loss0 function along angles, shifts, ctf_params, imgs."""
        return jax.vmap(self.loss0, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Compute the mean of the vectorized loss function loss_batched,
        over the vectorized axis."""
        return jnp.mean(self.loss_batched(v, angles, shifts, ctf_params, imgs, sigma))

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj(self, v, projection, shifts, ctf_params, img, sigma):
        """Loss when the rotation is already applied.
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
        """Vectorized version of loss_proj,
        along projection, shifts, ctf_params, imgs."""

        return jax.vmap(self.loss_proj, in_axes=(None, 0, 0, 0, 0, None))(
            v, projection, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_proj_batched0(self, v, projection, shifts, ctf_params, imgs, sigma):
        """Vectorized version of loss_proj0,
        along projection, shifts, ctf_params, imgs."""

        return jax.vmap(self.loss_proj0, in_axes=(None, 0, 0, 0, 0, None))(
            v, projection, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_px(self, v, angles, shifts, ctf_params, img, sigma=1):
        """Pixel-wise loss for one image, with no sigma and regularization."""

        err = self.slice.slice(v, angles, shifts, ctf_params) - img
        return 1 / 2 * jnp.real(jnp.conj(err) * err)

    @partial(jax.jit, static_argnums=(0,))
    def loss_px_batched(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Vectorized loss_px function, along angles, shifts, ctf_params, imgs."""

        return jax.vmap(self.loss_px, in_axes=(None, 0, 0, 0, 0, None))(
            v, angles, shifts, ctf_params, imgs, sigma
        )

    @partial(jax.jit, static_argnums=(0,))
    def loss_px_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Compute the mean of the loss_px_batched over the vectorized axis."""

        return jnp.mean(
            self.loss_px_batched(v, angles, shifts, ctf_params, imgs, sigma), axis=0
        )


class GradV:
    """Gradient of the loss function with respect to the volume.

    Attributes:
    -----------
    loss:
        An instance of the Loss class defined above.
    """

    def __init__(self, loss: Loss):
        self.loss = loss

    @partial(jax.jit, static_argnums=(0,))
    def grad_loss_volume(self, v, angles, shifts, ctf_params, img, sigma):
        """Gradient of the loss function with respect to the volume."""

        return jax.grad(self.loss.loss)(v, angles, shifts, ctf_params, img, sigma)

    @partial(jax.jit, static_argnums=(0,))
    def grad_loss_volume_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        """Gradient of the loss_sum function with respect to the volume."""

        return jax.grad(self.loss.loss_sum)(v, angles, shifts, ctf_params, imgs, sigma)
