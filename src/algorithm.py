import jax.numpy as jnp
from tqdm import tqdm




def conjugate_gradient(op, b, x0, iterations, eps = 1e-16, verbose = False):
    """ Apply the conjugate gradient method where op(x) performs Ax for
    Hermitian positive-definite matrix A."""
    r = b - op(x0)

    x = x0
    p = r
    #for k in tqdm(range(iterations)):
    for k in range(iterations):
        rkTrk = jnp.sum(jnp.conj(r) * r)
        Ap = op(p)
        
        alpha = rkTrk / jnp.sum(jnp.conj(p) * Ap)

        x = x + alpha * p
        r = r - alpha * Ap 

        norm_r = jnp.linalg.norm(r.ravel(),2)
        if norm_r < eps:
            return x, k

        beta = jnp.sum(jnp.conj(r) * r) / rkTrk
        p = r + beta * p

        if verbose and jnp.mod(k,50) == 0:
            print("Iter", k, "||r|| =", norm_r)
                    
    return x, k


def get_cg_vol_ops(grad_loss_volume_sum, angles, shifts, ctf_params, imgs_f, vol_shape):
    """Get the AA and Ab required to apply CG to find
    the volume for known angles and shifts.
    
    Parameters:
    -----------
    grad_loss_volume_sum: Jax function
        Function that returns the gradient of the loss function
        with respect to the volume.

    angles: N x 3 array
        N stacked vectors representing the orientation angles [psi, tilt, rot]. 

    shifts: N x 2 array
        N stacked vectors representing the shifts [originx, originy].

    ctf_params: N x 9 array
        N stacked vectors representing the CTF parameters, in the order given 
        in the ctf.py file.

    imgs_f : N x n array
        N images (in the Fourier domain), vectorized, each of dimension n.

    vol_shape: [int, int, int]
        Shape of the volume.

    Returns:
    --------
    AA, Ab: the operator and the vector to give CG to solve A*Ax = A*b.
    """

    zero = jnp.zeros(vol_shape).astype(jnp.complex64)
    Abfun = grad_loss_volume_sum(zero, angles, shifts, ctf_params, imgs_f)

    Ab = - jnp.conj(Abfun)
    AA = lambda vv : jnp.conj(grad_loss_volume_sum(vv, angles, shifts, ctf_params, imgs_f)) + Ab

    return AA, Ab
