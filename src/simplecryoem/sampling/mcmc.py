import jax
from jax import random
import jax.numpy as jnp


def mcmc_sampling(
    key,
    proposal_func,
    x0,
    N_samples,
    proposal_params,
    N_batch=1,
    save_samples=-1,
    verbose=True,
    iter_display=50,
):
    """Generic code for MCMC sampling.

    Parameters:
    ----------
    key : jnp.random.PRNGKey
        Key for jax random functions

    proposal_func :
        Function that gives a proposal sample and its
        Metropolis-Hastings ratio r.

    x0 :
        Starting point.

    N_samples : int
        Number of MCMC samples

    proposal_params: dict
        Dictionary containing the parameters of proposal_func.

    N_batch : int
        The number of independent variables that are being sampled
        in parallel, which determines whether the accept_reject function
        is applied over the whole vector or independently for each variable
        being sampled. In particular, for volume sampling,set N_batch=1
        and for orientations/shifts sampling, set N_batch = N_images.

    save_samples: int
        Save and return all the samples with index i such that
        mod(i, save_samples) = 0. If save_samples = -1, only return
        the last sample.

    verbose : boolean

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
        The array of saved samples. If save_samples = -1,
        it only contains the last sample.
    """

    key, *keys = random.split(key, 2 * N_samples + 1)

    r_samples = []
    samples = []
    x_mean = jnp.zeros(x0.shape)

    x1 = x0

    if N_batch > 1:
        logPiX1 = jnp.inf * jnp.ones(N_batch)
    else:
        logPiX1 = jnp.inf
    for i in range(1, N_samples):
        x0 = x1
        logPiX0 = logPiX1
        x1, r, logPiX1, logPiX0 = proposal_func(
            keys[2 * i], x0, logPiX0, **proposal_params
        )

        a = jnp.minimum(1, r)
        r_samples.append(a)

        if N_batch > 1:
            unif_var = random.uniform(keys[2 * i + 1], (N_batch,))
            x1, logPiX1 = accept_reject_vmap(
                unif_var, a, x0, x1, logPiX0, logPiX1)
        else:
            unif_var = random.uniform(keys[2 * i + 1])
            x1, logPiX1 = accept_reject_scalar(
                unif_var, a, x0, x1, logPiX0, logPiX1)

        x_mean = (x_mean * (i - 1) + x1) / i

        if save_samples > 0 and jnp.mod(i, save_samples) == 0:
            samples.append(x1)

        if verbose and jnp.mod(i, iter_display) == 0:
            if isinstance(N_batch, jnp.ndarray):
                loss_i = jnp.mean(logPiX1)
                print(f"  MCMC sample {i}, posterior val = {loss_i}")
            elif N_batch > 1:
                loss_i = jnp.mean(logPiX1)
                # print("  Iter", i, ", a_mean = ", jnp.mean(a))
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
    """Function to accept/reject proposed samples.
    This version of the function works with scalars.
    For vectors, use `accept_reject_vmap`."""

    x = jax.lax.cond(
        unif_var <= a, true_fun=lambda _: x1, false_fun=lambda _: x0, operand=None
    )

    logPiX = jax.lax.cond(
        unif_var <= a,
        true_fun=lambda _: logPiX1,
        false_fun=lambda _: logPiX0,
        operand=None,
    )

    return x, logPiX


accept_reject_vmap = jax.jit(
    jax.vmap(accept_reject_scalar, in_axes=(0, 0, 0, 0, 0, 0)))
