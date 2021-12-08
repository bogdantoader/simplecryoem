import jax
import jax.numpy as jnp

# Forward operator

#TODO: maybe not all the functions in this file need to be jit-ed
#TODO 2: do I need to have all these functions? maybe one function returning whats necessary would be better?
#TODO 3: as part of 2, make the get_loss_func, slice_func, l2sq play together more nicely, see in Roy's code - should take a generic project function and loss function, then return all the good stuff I need (slice, loss, grad)
def get_slice_funcs(project, x_grid, y_grid, z_grid, interp_method = "tri"):

    @jax.jit
    def slice_func(v, angles, shifts, ctf_params):
        projection, _ = project(v, angles, shifts, ctf_params, x_grid, y_grid, z_grid, interp_method)
        return projection

    @jax.jit
    def slice_func_array(v, angles, shifts, ctf_params):    
        return jax.vmap(slice_func, in_axes = (None, 0, 0, 0))(v, angles)

    return slice_func, slice_func_array
   
# Loss functions

def get_loss_funcs(slice_func, alpha):
    """L2 squared error with L2 regularization, where alpha is the
    regularization parameter."""

    @jax.jit
    def loss_func(v, angles, shifts, ctf_params, img):
        nx = v.shape[-1]

        #return 1/(2* nx*nx) * l2sq(slice_func(v, angles) - img)
        # With l2 regularization
        return 1/(2* nx*nx) * (alpha * l2sq(v) + l2sq(slice_func(v, angles, shifts, ctf_params) - img))

    @jax.jit 
    def loss_func_batched(v, angles, shifts, ctf_params, imgs):
        return jax.vmap(loss_func, in_axes = (None, 0, 0, 0, 0))(v, angles, shifts, ctf_params, imgs)

    @jax.jit
    def loss_func_sum(v, angles, shifts, ctf_params, imgs):
        return jnp.mean(loss_func_batched(v, angles, shifts, ctf_params, imgs))

    return jax.jit(loss_func), jax.jit(loss_func_sum)


@jax.jit
def l2sq(x):
    return jnp.real(jnp.sum(jnp.conj(x)*x))
