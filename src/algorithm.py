import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from  matplotlib import pyplot as plt

from src.utils import l2sq, generate_uniform_orientations_jax



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
            print("  cg iter", k, "||r|| =", norm_r)
            return x, k

        beta = jnp.sum(jnp.conj(r) * r) / rkTrk
        p = r + beta * p

        if verbose and jnp.mod(k,50) == 0:
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


def sgd(grad_func, N, x0, alpha = 1, N_epoch = 10, batch_size = -1, P = None, eps = 1e-15, verbose = False, loss_func = None):
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

            if jnp.mod(epoch, 10) == 0 and i == len(idx_batches)-1:
                full_grad = jnp.abs(jnp.mean(grad_func(x, jnp.arange(N))))

                if verbose:
                    print("  sgd epoch " + str(epoch) + ": mean gradient = " + str(full_grad))

        if full_grad < eps:
            break

    return x


def get_sgd_vol_ops(grad_loss_volume_batched, angles, shifts, ctf_params, imgs, sigma = 1):
    #loss_func = lambda v, idx : loss_func_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx]) 
    grad_func = lambda v, idx : grad_loss_volume_batched(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma) 

    return grad_func


def mala_vol_proposal(loss_func, grad_func, N, v0, tau):
    """Generate a new proposal for the volume with a Langevin step.
    Returns the proposal and the Hastings ratio."""

    rng = np.random.default_rng()
    noise = jnp.array(rng.standard_normal(v0.shape))

    # Full gradient for now
    idx = jnp.arange(N)

    # Note the minus sign since the loss function implicitly has a minus too.
    g = grad_func(v0, idx)
    det_term = v0 - tau**2/2 * jnp.conj(g)
    v1 =  det_term + tau * noise
    #v1 = det_term
    #print(jnp.abs(jnp.mean(g)))

    # How and where to compute the Hastings radio (or some of its components)
    # for maximum efficiency.
    qv0v1_exparg = tau*noise #v1-det_term
    qv1v0_exparg = v0 - v1 + tau**2/2 * jnp.conj(grad_func(v1, idx) )

    q_ratio = jnp.exp(-1/(2*tau**2) * (l2sq(qv1v0_exparg) - l2sq(qv0v1_exparg)))
    h_ratio = jnp.exp(-loss_func(v1, idx) + loss_func(v0, idx))
    ratio = q_ratio*h_ratio

    #print("MALA")
    #print(-loss_func(v1,idx))
    #print(-loss_func(v0,idx))
    #print(q_ratio)
    #print(h_ratio)

    return v1, ratio


def proposal_mala(key, logPi, x0, gradLogPi, tau):
    noise = jnp.array(random.normal(key, x0.shape))
    x1 = x0 + tau**2/2 * gradLogPi(x0) + tau * noise

    qx0x1_exparg = tau*noise
    qx1x0_exparg = x0 - x1 - tau**2/2 * gradLogPi(x1)
    q_ratio = jnp.exp(-1/(2*tau**2) * (l2sq(qx1x0_exparg) - l2sq(qx0x1_exparg)))
    h_ratio = jnp.exp(logPi(x1)-logPi(x0))
    r = q_ratio * h_ratio

    return x1, r


def proposal_hmc(key, logPi, x0, gradLogPi, dt, L = 1, M = 1):
    """ Hamiltonian Monte Carlo proposal function.
    For simplicity, the mass matrix M is an array of 
    entry-wise scalings (i.e. a diagonal matrix).
    This should be scaled roughly similarly to gradLogPi, e.g.
    for the Relion tutorial data, I take
    M = 1/max(sigma_noise)**2 * ones.
    """

    p0 = random.normal(key, x0.shape) * M
    r0exponent = logPi(x0) - jnp.sum(jnp.real(jnp.conj(p0) * p0))/2

    for i in range(L):
        p01 = p0 + dt/2 * gradLogPi(x0)
        x1 = x0 + dt * p01 / M
        p1 = p01 + dt/2 * gradLogPi(x1)

        p0 = p1
        x0 = x1

    r1exponent = logPi(x1) - jnp.sum(jnp.real(jnp.conj(p1) * p1))/2
    r = jnp.exp(r1exponent - r0exponent)

    return x1, r


#TODO: Remove gradLogPi when the mcmc function doesn't require it.
def proposal_uniform_orientations(key, logPi, x0):
    """Uniform orientations proposal function, for N
    (independent) images at once.
    
    Parameters:
    -----------
    key : jax.random.PRNGKey
    
    logPi :
        Function that computes the log of the target
        density function Pi. If working on multiple images, 
        logPi returns a vector.
        
    x0 : array 
        Current state, with N = x0.shape[0]. It is important
        that x0 is an array even when N=1.
    
    Returns:
    --------
    Proposed sample x1 and the Metropolis-Hastings ratio r.
    """

    N = x0.shape[0]
    x1 = generate_uniform_orientations_jax(key, N)
    r = jnp.exp(logPi(x1) - logPi(x0))

    return x1, r



def mcmc(key, N_samples, proposal_func, logPi, x0, proposal_params = {}, N_batch = 1, save_samples = -1, verbose = True):
    """Generic code for MCMC sampling.

    Parameters:
    ----------
    key : jnp.random.PRNGKey
        Key for jax random functions

    N_samples : int
        Number of MCMC samples

    proposal_func : 
        Function that gives a proposal sample and its 
        Metropolis-Hastings ratio r.

    logPi : 
        Function that evaluates the log of the target 
        distribution Pi.

    gradLogPi :
        Function that evaluates the gradient of the log of the target
        distribution Pi.

    x0:
        Starting point.

    proposal_params: dict
        A dict of params corresponding to the specific parameters of 
        the proposal_func. This includes gradLogPi if required by
        the proposal function.

    N_batch : int
        The number of independent variables that are being sampled
        in parallel (e.g. for volume it would be one, for orientations
        it would be more).

    save_samples: int
        Save and return all the samples with index i such that
        mod(i, save_samples) = 0. If save_samples = -1, return an
        empty array instead.

    Returns:
    -------
    x_mean : 
        The mean of the samples.
  
    r : array
        Array containing all the Metropolis Hastings ratios.

    samples : 
        The array of saved samples. Empty if save_samples = -1.
    """ 


    key, *keys = random.split(key, 2*N_samples+1)
    
    r_samples = []
    samples = []
    x_mean = jnp.zeros(x0.shape)

    x1 = x0
    for i in range(1, N_samples):
        x0 = x1
        x1, r = proposal_func(keys[2*i], logPi, x0, **proposal_params)
        a = jnp.minimum(1, r)
        r_samples.append(a)
  
        if N_batch > 1:
            unif_var = random.uniform(keys[2*i+1], (N_batch,)) 
            x1 = accept_reject_vmap(unif_var, a, x0, x1)
        else:
            unif_var = random.uniform(keys[2*i+1]) 
            x1 = accept_reject_scalar(unif_var, a, x0, x1)

        x_mean = (x_mean * (i-1) + x1) / i
        
        if save_samples and jnp.mod(i, save_samples) == 0:
            samples.append(x1)

        if verbose and jnp.mod(i, 50) == 0:
            if N_batch > 1:
                print("Iter", i, ", a[0] = ", a[0])
            else:
                print("Iter", i, ", a = ", a)
            #plt.imshow(jnp.fft.fftshift(jnp.abs(x_mean[0]))); plt.colorbar()
            #plt.show()

    r_samples = jnp.array(r_samples)
    samples = jnp.array(samples)

    return x_mean, r_samples, samples 


def accept_reject_scalar(unif_var, a, x0, x1):
    return jax.lax.cond(unif_var < a, 
        true_fun = lambda _ : x1,
        false_fun = lambda _ : x0,
        operand = None)

accept_reject_vmap = jax.vmap(accept_reject_scalar, in_axes = (0, 0, 0, 0))

# Not used anywhere, to delete
def mcmc_batch_step(key, N_samples, logPi_batch, x0):
    """MCMC sampling of N independent variables, batched 
    (e.g. orientations of different images)."""
    key1, key2 = random.split(key, 2)
    
    N = x0.shape[0]             
    x1 = generate_uniform_orientations_jax(key1, N)       
    r0exponent = logPi_batch(x0)
    r1exponent = logPi_batch(x1)
       
    r = jnp.exp(r1exponent-r0exponent)
    a = jnp.minimum(1, r)
    
    unif_var = random.uniform(key2, (N,))
    return accept_reject_vmap(unif_var, a, x0, x1)
