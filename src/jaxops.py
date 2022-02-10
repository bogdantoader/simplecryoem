import jax
import jax.numpy as jnp
from src.utils import l2sq


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
def get_loss_funcs(slice_func, err_func = l2sq, alpha = 0):


    #TODO: should the regularisation term here be normalise to account for
    # the fact that v is 3D while the arguments of err_func in the fidelity
    # term are 2D, and also the fact that we have one regularisation term
    # included for each image? So in SGD, when varying the batch size, 
    # the implicit regularisation parameter will change too.
    # Yeah, this definitely needs to be fixed.
    @jax.jit
    def loss_func(v, angles, shifts, ctf_params, img):
        """L2 squared error with L2 regularization, where alpha is the
        regularization parameter."""
        
        return 1/(2* v.shape[-1]**2) * (alpha * l2sq(v) + err_func(slice_func(v, angles, shifts, ctf_params), img))
        #return 1/2 * (alpha * l2sq(v) + err_func(slice_func(v, angles, shifts, ctf_params), img))

    @jax.jit 
    def loss_func_batched(v, angles, shifts, ctf_params, imgs):
        return jax.vmap(loss_func, in_axes = (None, 0, 0, 0, 0))(v, angles, shifts, ctf_params, imgs)

    # TODO: and is it ok to take the mean instead of the sum here?
    # Does it break the Hastings ratio?
    @jax.jit
    def loss_func_sum(v, angles, shifts, ctf_params, imgs):
        return jnp.mean(loss_func_batched(v, angles, shifts, ctf_params, imgs))
        #return jnp.sum(loss_func_batched(v, angles, shifts, ctf_params, imgs))

    #@jax.jit 
    def loss_func_sum_iter(v, angles, shifts, ctf_params, imgs):
        loss = jnp.zeros(v.shape)

        for i in range(angles.shape[0]):
            loss += loss_func(v,angles[i], shifts[i], ctf_params[i], imgs[i])

        return loss/v.shape[0]
    
    return loss_func, loss_func_batched, loss_func_sum, loss_func_sum_iter

# Grads
def get_grad_v_funcs(loss_func, loss_func_sum):

    @jax.jit
    def grad_loss_volume(v, angles, shifts, ctf_params, img):
        return jax.grad(loss_func)(v, angles, shifts, ctf_params, img)

    @jax.jit
    def grad_loss_volume_batched(v, angles, shifts, ctf_params, imgs):
        #return 1/imgs.shape[0] * jnp.sum(jax.vmap(grad_loss_volume, in_axes = (None, 0, 0, 0, 0))(v, angles, shifts, ctf_params, imgs), axis=0)
        return jnp.sum(jax.vmap(grad_loss_volume, in_axes = (None, 0, 0, 0, 0))(v, angles, shifts, ctf_params, imgs), axis=0)

    @jax.jit
    def grad_loss_volume_sum(v, angles, shifts, ctf_params, imgs):
        return jax.grad(loss_func_sum)(v, angles, shifts, ctf_params, imgs)

    return grad_loss_volume, grad_loss_volume_batched, grad_loss_volume_sum


