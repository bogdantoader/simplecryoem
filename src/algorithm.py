import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from  matplotlib import pyplot as plt
import time

from src.utils import l2sq, generate_uniform_orientations_jax, generate_uniform_shifts,generate_gaussian_shifts
from src.jaxops import GradV, Loss



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
            #print("  cg iter", k, "||r|| =", norm_r)
            return x, k

        beta = jnp.sum(jnp.conj(r) * r) / rkTrk
        p = r + beta * p

        if verbose and jnp.mod(k,10) == 0:
            print("  cg iter", k, "||r|| =", norm_r)
                    
    return x, k


def get_cg_vol_ops(grad_loss_volume_sum, angles, shifts, ctf_params, imgs_f, vol_shape, sigma = 1):
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
    Abfun = grad_loss_volume_sum(zero, angles, shifts, ctf_params, imgs_f, sigma)

    Ab = - jnp.conj(Abfun)
    AA = lambda vv : jnp.conj(grad_loss_volume_sum(vv, angles, shifts, ctf_params, imgs_f, sigma)) + Ab

    return AA, Ab


def sgd(grad_func, loss_func, N, x0, alpha = 1, N_epoch = 10, batch_size = -1, P = None, eps = 1e-15, verbose = False, iter_display = 1):
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
        N_batch = 1
    else:
        N_batch = N/batch_size

    if P is None:
        P = jnp.ones(x0.shape)

    x = x0
    loss_list = []
    grad_list = []
    for idx_epoch, epoch in enumerate(range(N_epoch)):
        print(f"Epoch {idx_epoch+1}/{N_epoch} ", end="")
        idx_batches = np.array_split(rng.permutation(N), N_batch)

        grad_epoch = []
        loss_epoch = []
        pbar = tqdm(idx_batches)
        #aa = 0
        for idx in pbar:
            gradx = grad_func(x, idx)
            x = x - alpha * P * jnp.conj(gradx)
            
            loss_iter = loss_func(x, idx)

            gradmax = jnp.max(jnp.abs(gradx))
            grad_epoch.append(gradmax)
            loss_epoch.append(loss_iter)
            
            pbar.set_postfix(grad = f"{gradmax :.3e}",
                    loss = f"{loss_iter :.2f}")
                
            #time.sleep(2)
            #aa +=1 
            #if aa == 100:
            #    break

        grad_epoch = jnp.mean(jnp.array(grad_epoch))
        loss_epoch = jnp.mean(jnp.array(loss_epoch)) 
        print(f"  |Grad| = {grad_epoch :.3e}")
        print(f"  |Loss| = {loss_epoch :.3f}")

        grad_list.append(grad_epoch)
        loss_list.append(loss_epoch)

        if grad_epoch < eps:
            break


    return x, jnp.array(loss_list), jnp.array(grad_list)


def get_sgd_vol_ops(gradv: GradV, loss: Loss, angles, shifts, ctf_params, imgs, sigma = 1):
    loss_func = lambda v, idx : loss.loss_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma) 
    grad_func = lambda v, idx : gradv.grad_loss_volume_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx],  sigma) 

    return grad_func, loss_func



def kaczmarz(key, data, angles, fwd_model_vmap, loss_func, grad_loss_func, x0, N_epoch, N_batches, N_iter_cg = 2, eps_cg = 1e-7):
    """Implementation of the randomized block Kaczmarz method
    introduced in [Needell & Tropp 2014]. Convenient for processing
    a batch of particle images at one time, where for each batch
    we solve a least squares problem using the CG algorithm above.

    Rough around the edges but working implementation. It might require
    a few adaptations to work with the cryoEM operators.l"""


    key, subkey = random.split(key)

    N = data.shape[0]
    index_permutations = random.permutation(subkey, N)
    block_indices = np.array(np.array_split(index_permutations, N_batches))
    print(f"{block_indices.shape[0]} iterations/epoch")

    x = x0 
    zero = jnp.zeros(x0.shape)

    for ep in range(N_epoch):   
        if ep % 1 == 0:
            print(f"Epoch {ep}")
            verbose_cg = True

        for i, idx in tqdm(enumerate(block_indices)):
            #if verbose_cg:
            #    print(i)

            # Solve the least squares problem to apply the pseudoinverse   
            data_block = -fwd_model_vmap(x, angles[idx]) + data[idx]

            Ab = -grad_loss_func(zero, angles[idx], data_block)
            AA = lambda v : grad_loss_func(v, angles[idx], data_block) + Ab

            x_ls, k = conjugate_gradient(AA, Ab, zero, N_iter_cg, eps = eps_cg, verbose = verbose_cg)

            x = x + x_ls

            verbose_cg = False    

    return x







