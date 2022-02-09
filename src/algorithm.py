import jax.numpy as jnp
import numpy as np
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
            print("  cg iter", k, "||r|| =", norm_r)
                    
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


def sgd(grad_func, N, x0, alpha = 1, N_epoch = 10, batch_size = -1, P = None, verbose = False, loss_func = None):
    """SGD
   
   Parameters:
   grad_func : (x, idx) -> sum_i grad(f_i(x)), i=1,...N
        A function that takes a volume x and an array of indices idx
        and returns the sum of the gradients of the loss functions at 
        volume x and images indexed by idx.

    N : int 
        The total number of images/samples.

    x0 : nx x nx x nx
        Starting volume

    alpha : float
        Learning rate

    N_epochs : int
        Number of passes through the full dataset.

    batch_size : int
        Batch size

    P : nx x nx x nx
        Diagonal preconditioner (entry-wise multiplication of the gradient).
    
    """

    rng = np.random.default_rng()

    if batch_size == -1 or batch_size == N:
        number_of_batches = 1
    else:
        number_of_batches = N/batch_size

    if P is None:
        P = jnp.ones(x0.shape)

    x = x0
    for epoch in range(N_epoch):
        idx_batches = np.array_split(rng.permutation(N), number_of_batches)

        for i, idx in enumerate(idx_batches):
            x = x - alpha * P * jnp.conj(grad_func(x, idx))

            if verbose and jnp.mod(epoch,50) == 0 and i == len(idx_batches)-1:
                #print("  sgd epoch " + str(epoch) + ": mean sampled gradient = " + str(jnp.abs(jnp.mean(grad_func(x, idx)))))
                #Print the full gradient for now
                if loss_func is not None:
                    print("  sgd epoch " + str(epoch) + ": mean loss func= " + str(loss_func(x, jnp.arange(N))))
                else:
                    print("  sgd epoch " + str(epoch) + ": mean full gradient = " + str(jnp.abs(jnp.mean(grad_func(x, jnp.arange(N))))))

    return x


def get_sgd_vol_ops(grad_loss_volume_batched, loss_func_sum, angles, shifts, ctf_params, imgs):

    grad_func = lambda v, idx : grad_loss_volume_batched(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx]) 
    loss_func = lambda v, idx : loss_func_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx])
    return grad_func, loss_func


