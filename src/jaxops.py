import jax
import jax.numpy as jnp
from src.utils import l2sq, wl2sq
from src.projection import project, apply_shifts_and_ctf, rotate_and_interpolate

class Slice:
    """Class to represent the slice operator in the Fourier domain.
    The methods are useful functions for reconstruction, 
    as jit-compiled JAX functions."""

    def __init__(self, x_grid, mask = None, project = project,  
            apply_shifts_and_ctf = apply_shifts_and_ctf, 
            interp_method = "tri"):

        self.project = project
        self.x_grid = x_grid
        self.interp_method = interp_method

        if mask == None:
            nx = x_grid[1]
            self.mask = jnp.ones([nx, nx])
        else:
            self.mask = mask

    @jax.jit
    def slice(self, v, angles, shifts, ctf_params):
        return self.project(v * self.mask, angles, shifts, ctf_params, 
                self.x_grid, self.x_grid, self.interp_method)

    @jax.jit
    def slice_array(self, v, angles, shifts, ctf_params):    
        return jax.vmap(self.slice, in_axes = (None, 0, 0, 0))(v, angles, shifts, ctf_params)

    @jax.jit
    def slice_array_angles(self, v, angles, shifts, ctf_params):    
        """Same as above, except the shifts and ctf_params are 
        fixed and we don't vectorize them."""
        return jax.vmap(self.slice, in_axes = (None, 0, None, None))(v, angles, shifts, ctf_params)

    @jax.jit
    def apply_shifts_and_ctf(projection, shifts, ctf_params):
        return self.apply_shifts_and_ctf(projection, shifts, ctf_params, self.x_grid)

    @jax.jit
    def rotate_and_interpolate_vmap(v, angles):
        return jax.vmap(rotate_and_interpolate, in_axes=(None,0,None,None))(v*mask, angles, x_grid, x_grid)

class Loss:
    """A class to represent the loss function, batched and sum.

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

    def __init__(self, slice: Slice, err_func = wl2sq, alpha = 0):
        self.slice = slice
        self.err_func = err_func
        self.alpha = alpha 

    @jax.jit
    def loss(self, v, angles, shifts, ctf_params, img, sigma = 1):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard 
        deviation of the noise."""
        return 1/2 * (self.alpha * l2sq(v) + self.err_func(self.slice.slice(v, angles, shifts, ctf_params), img, 1/sigma**2))

    @jax.jit 
    def loss_batched(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss, in_axes = (None, 0, 0, 0, 0,  None))(v, angles, shifts, ctf_params, imgs, sigma)

    @jax.jit
    def loss_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jnp.mean(self.loss_batched(v, angles, shifts, ctf_params, imgs, sigma))

    @jax.jit
    def loss_proj(v, projection, shifts, ctf_params, img, sigma):
        """Loss when the rotation is already done.
        L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard 
        deviation of the noise."""

        proj = self.slice.apply_shifts_and_ctf_jit(projection, shifts, ctf_params)
        return 1/2 * (self.alpha * l2sq(v) + self.err_func(projection, img, 1/sigma**2))

    @jax.jit 
    def loss_proj_batched(v, projection, shifts, ctf_params, imgs, sigma):
        return jax.vmap(self.loss_proj, in_axes = (None, 0, 0, 0, 0, None))(v, projection, shifts, ctf_params, imgs, sigma)


class GradV:
    def __init__(self, loss: Loss):
        self.loss = loss

    @jax.jit
    def grad_loss_volume(self, v, angles, shifts, ctf_params, img, sigma):
        return jax.grad(self.loss.loss)(v, angles, shifts, ctf_params, img, sigma)

    @jax.jit
    def grad_loss_volume_sum(self, v, angles, shifts, ctf_params, imgs, sigma):
        return jax.grad(self.loss.loss_sum)(v, angles, shifts, ctf_params, imgs, sigma)



##TODO: Include the below functions in the above classes
def get_loss_grad_funcs_classes(loss_func):
    """Wrappers of the loss and grad functions above that handle the 
    class assignment variable z."""


    @jax.jit
    def loss_func_z(v, angles, shifts, ctf_params, img, z, sigma = 1):
        """Here, v is an array of volumes of length K (number of classes)."""

        return loss_func(v[z], angles, shifts, ctf_params, img, sigma)

    @jax.jit
    def loss_func_z_batched(v, angles, shifts, ctf_params, imgs, z, sigma):
        return jax.vmap(loss_func_z, in_axes = (None, 0, 0, 0, 0, 0, None))(v, angles, shifts, ctf_params, imgs, z, sigma)

    @jax.jit
    def loss_func_z_sum(v, angles, shifts, ctf_params, imgs, z, sigma):
        return jnp.mean(loss_func_z_batched(v, angles, shifts, ctf_params, imgs, z, sigma))

    @jax.jit
    def grad_loss_volume_z(v, angles, shifts, ctf_params, img, z, sigma):
        return jax.grad(loss_func_z)(v, angles, shifts, ctf_params, img, z, sigma)

    @jax.jit
    def grad_loss_volume_sum_z(v, angles, shifts, ctf_params, imgs, z, sigma):
        return jax.grad(loss_func_z_sum)(v, angles, shifts, ctf_params, imgs, z, sigma)

    return loss_func_z, loss_func_z_batched, loss_func_z_sum, grad_loss_volume_z, grad_loss_volume_sum_z

