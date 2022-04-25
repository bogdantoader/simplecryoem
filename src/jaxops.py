import jax
import jax.numpy as jnp
from src.utils import l2sq, wl2sq


#TODO: maybe not all the functions in this file need to be jit-ed

# Slice functions
def get_slice_funcs(project, x_grid,  mask, interp_method = "tri"):

    @jax.jit
    def slice_func(v, angles, shifts, ctf_params):
        return project(v * mask, angles, shifts, ctf_params, x_grid, x_grid, interp_method)

    @jax.jit
    def slice_func_array(v, angles, shifts, ctf_params):    
        return jax.vmap(slice_func, in_axes = (None, 0, 0, 0))(v, angles, shifts, ctf_params)

    @jax.jit
    def slice_func_array_angles(v, angles, shifts, ctf_params):    
        """Same as above, except the shifts and ctf_params are fixed and we don't vectorize them."""
        return jax.vmap(slice_func, in_axes = (None, 0, None, None))(v, angles, shifts, ctf_params)

    return slice_func, slice_func_array, slice_func_array_angles
   
# Loss functions
def get_loss_funcs(slice_func, err_func = wl2sq, alpha = 0):
    """Get the loss function, batched and sum.

    Parameters:
    -----------
    slice_func : 
        Slice function for one image

    err_func : 
        Function to calculate the error between two images,
        defaults to l2 squared.

    alpha : 
        Regularisation parameter for l2 regularisation.
    """


    @jax.jit
    def loss_func(v, angles, shifts, ctf_params, img, sigma = 1):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard 
        deviation of the noise."""
        return 1/2 * (alpha * l2sq(v) + err_func(slice_func(v, angles, shifts, ctf_params), img, 1/sigma**2))

    @jax.jit 
    def loss_func_batched(v, angles, shifts, ctf_params, imgs, sigma):
        return jax.vmap(loss_func, in_axes = (None, 0, 0, 0, 0, 0, None))(v, angles, shifts, ctf_params, imgs, sigma)

    @jax.jit
    def loss_func_sum(v, angles, shifts, ctf_params, imgs, sigma):
        return jnp.mean(loss_func_batched(v, angles, shifts, ctf_params, imgs, sigma))

    return loss_func, loss_func_batched, loss_func_sum


# Grads
def get_grad_v_funcs(loss_func, loss_func_sum):

    @jax.jit
    def grad_loss_volume(v, angles, shifts, ctf_params, img, sigma):
        return jax.grad(loss_func)(v, angles, shifts, ctf_params, img, sigma)

    @jax.jit
    def grad_loss_volume_sum(v, angles, shifts, ctf_params, imgs, sigma):
        return jax.grad(loss_func_sum)(v, angles, shifts, ctf_params, imgs, sigma)

    return grad_loss_volume, grad_loss_volume_sum

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

# Loss from proj functions - to be used for MCMC for shifts
def get_loss_proj_funcs(apply_shifts_and_ctf, x_grid, err_func = wl2sq, alpha = 0):

    @jax.jit
    def apply_shifts_and_ctf_jit(proj, shifts, ctf_params):
        return apply_shifts_and_ctf(proj, shifts, ctf_params, x_grid)

    @jax.jit
    def loss_proj_func(v, proj, shifts, ctf_params, img, sigma):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter and sigma is the pixel-wise standard 
        deviation of the noise."""
        proj = apply_shifts_and_ctf_jit(proj, shifts, ctf_params)

        return 1/2 * (alpha * l2sq(v) + err_func(proj, img, 1/sigma**2))

    @jax.jit 
    def loss_proj_func_batched(v, proj, shifts, ctf_params, imgs, sigma):
        return jax.vmap(loss_proj_func, in_axes = (None, 0, 0, 0, 0, None))(v, proj, shifts, ctf_params, imgs, sigma)

    return loss_proj_func_batched
