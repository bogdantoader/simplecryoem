from jax import random
import jax.numpy as jnp


def proposal_hmc(key, x0, logPiX0, logPi, gradLogPi, dt_list, L=1, M=1):
    """Hamiltonian Monte Carlo proposal function.
    For simplicity, the mass matrix M is an array of
    entry-wise scalings (i.e. a diagonal matrix).
    This should be scaled roughly similarly to gradLogPi, e.g.
    M = 1/max(sigma_noise)**2 * ones.
    """

    key, subkey = random.split(key)
    dt = random.permutation(subkey, dt_list)[0]

    logPiX0 = logPi(x0)

    p0 = random.normal(key, x0.shape) * M
    r0exponent = logPiX0 - jnp.sum(jnp.real(jnp.conj(p0) * p0)) / 2

    # Doing this so that we don't compute gradLogPi(x1) twice.
    gradLogPiX0 = gradLogPi(x0)

    body_func = lambda i, xpg0: leapfrog_step(i, xpg0, dt, gradLogPi, M)

    x1 = x0
    p1 = p0
    gradLogPiX1 = gradLogPiX0
    for i in jnp.arange(0, L):
        x1, p1, gradLogPiX1 = body_func(i, jnp.array([x1, p1, gradLogPiX1]))

    logPiX1 = logPi(x1)
    r1exponent = logPiX1 - jnp.sum(jnp.real(jnp.conj(p1) * p1)) / 2
    r = jnp.exp(r1exponent - r0exponent)

    return x1, r, logPiX1, logPiX0


def leapfrog_step(i, xpg0, dt, gradLogPi, M):
    """
    Leapfrog integrator.

    Arguments:
    ----------
    i : int
        step index (not actually used?)
    xpg0 : jnp.array([x0, p0, grad(log(Pi(x0)))])
    dt : double
    gradLogPi : function x -> grad(log(Pi(x))
    M : mass matrix (see above)

    Returns:
    --------
    xpg0 : jnp.array([x1, p1, grad(log(Pi(x1)))])

    """
    x0, p0, gradLogPiX0 = xpg0

    # note the + instead of in the p updates since we take U(x)=-log(pi(x))
    p01 = p0 + dt / 2 * gradLogPiX0
    x1 = x0 + dt * p01 / M

    gradLogPiX1 = gradLogPi(x1)
    p1 = p01 + dt / 2 * gradLogPiX1

    return jnp.array([x1, p1, gradLogPiX1])
