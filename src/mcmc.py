import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def mcmc(key, proposal_func, x0, N_samples, proposal_params, N_batch = 1, save_samples = -1, verbose = True, iter_display = 50):
    """Generic code for MCMC sampling.

    Parameters:
    ----------
    key : jnp.random.PRNGKey
        Key for jax random functions

    proposal_func : 
        Function that gives a proposal sample and its 
        Metropolis-Hastings ratio r.
    x0:
        Starting point.

    N_samples : int
        Number of MCMC samples

    logPi : 
        Function that evaluates the log of the target 
        distribution Pi. It is only used for displaying progress
        or debugging, the actual computation is done inside proposal_func.

    N_batch : int
        The number of independent variables that are being sampled
        in parallel (e.g. for volume it would be one, for orientations
        it would be more).

    save_samples: int
        Save and return all the samples with index i such that
        mod(i, save_samples) = 0. If save_samples = -1, only return 
        the last sample.

    iter_display : int
        Show the value of the distribution at the current sample 
        every iter_display iterations.

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

    if N_batch > 1:
        logPiX1 = jnp.inf*jnp.ones(N_batch)
    else:
        logPiX1 = jnp.inf 
    for i in range(1, N_samples):
        x0 = x1
        logPiX0 = logPiX1
        x1, r, logPiX1, logPiX0 = proposal_func(keys[2*i], x0, logPiX0, **proposal_params)

        a = jnp.minimum(1, r)
        r_samples.append(a)

        if N_batch > 1:
            unif_var = random.uniform(keys[2*i+1], (N_batch,)) 
            x1, logPiX1 = accept_reject_vmap(unif_var, a, x0, x1, logPiX0, logPiX1)
        else:
            unif_var = random.uniform(keys[2*i+1]) 
            x1, logPiX1 = accept_reject_scalar(unif_var, a, x0, x1, logPiX0, logPiX1)

        x_mean = (x_mean * (i-1) + x1) / i

        if save_samples > 0 and jnp.mod(i, save_samples) == 0:
            samples.append(x1)

        if verbose and jnp.mod(i, iter_display) == 0:
            if isinstance(N_batch, jnp.ndarray):
                loss_i = jnp.mean(logPiX1)
                print(f"  MCMC sample {i}, posterior val = {loss_i}")
            elif N_batch > 1:
                loss_i = jnp.mean(logPiX1)
                #print("  Iter", i, ", a_mean = ", jnp.mean(a))
                print(f"  MCMC sample {i}, posterior val = {loss_i}")
            else:
                loss_i = logPiX1
                print(f"  MCMC sample {i}, posterior val = {loss_i}, a = {a}")

    if save_samples == -1:
        samples.append(x1)

    r_samples = jnp.array(r_samples)
    samples = jnp.array(samples)

    return x_mean, r_samples, samples 


@jax.jit
def accept_reject_scalar(unif_var, a, x0, x1, logPiX0, logPiX1):
    x = jax.lax.cond(unif_var <= a, 
        true_fun = lambda _ : x1,
        false_fun = lambda _ : x0,
        operand = None)

    logPiX = jax.lax.cond(unif_var <= a, 
        true_fun = lambda _ : logPiX1,
        false_fun = lambda _ : logPiX0,
        operand = None)

    return x, logPiX

accept_reject_vmap = jax.jit(jax.vmap(accept_reject_scalar, in_axes = (0, 0, 0, 0, 0, 0)))


def proposal_hmc(key, x0, logPiX0, logPi, gradLogPi, dt_list, L = 1, M = 1):
    """ Hamiltonian Monte Carlo proposal function.
    For simplicity, the mass matrix M is an array of 
    entry-wise scalings (i.e. a diagonal matrix).
    This should be scaled roughly similarly to gradLogPi, e.g.
    M = 1/max(sigma_noise)**2 * ones.
    """

    key, subkey = random.split(key)
    dt = random.permutation(subkey, dt_list)[0]
    
    logPiX0 = logPi(x0)

    p0 = random.normal(key, x0.shape) * M
    r0exponent = logPiX0 - jnp.sum(jnp.real(jnp.conj(p0) * p0))/2

    # Doing this so that we don't compute gradLogPi(x1) twice.
    gradLogPiX0 = gradLogPi(x0)

    body_func = lambda i, xpg0: leapfrog_step(i, xpg0, dt, gradLogPi, M) 

    x1 = x0
    p1 = p0
    gradLogPiX1 = gradLogPiX0
    for i in jnp.arange(0, L):
        x1, p1, gradLogPiX1 = body_func(i, jnp.array([x1, p1, gradLogPiX1]))

    logPiX1 = logPi(x1)
    r1exponent = logPiX1 - jnp.sum(jnp.real(jnp.conj(p1) * p1))/2
    r = jnp.exp(r1exponent - r0exponent)
    
    return x1, r, logPiX1, logPiX0


def leapfrog_step(i, xpg0, dt, gradLogPi, M):
    x0, p0, gradLogPiX0 = xpg0 
    
    # note the + instead of in the p updates since we take U(x)=-log(pi(x))
    p01 = p0 + dt/2 * gradLogPiX0
    x1 = x0 + dt * p01 / M

    gradLogPiX1 = gradLogPi(x1)
    p1 = p01 + dt/2 * gradLogPiX1
    
    return jnp.array([x1, p1, gradLogPiX1]) 


def proposal_mala(key, x0, logPi, gradLogPi, tau):
    noise = jnp.array(random.normal(key, x0.shape))
    x1 = x0 + tau**2/2 * gradLogPi(x0) + tau * noise

    qx0x1_exparg = tau*noise
    qx1x0_exparg = x0 - x1 - tau**2/2 * gradLogPi(x1)
    q_ratio = jnp.exp(-1/(2*tau**2) * (l2sq(qx1x0_exparg) - l2sq(qx0x1_exparg)))
    h_ratio = jnp.exp(logPi(x1)-logPi(x0))
    r = q_ratio * h_ratio

    return x1, r
