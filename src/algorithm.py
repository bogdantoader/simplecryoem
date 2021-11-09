import jax.numpy as jnp



def conjugate_gradient(op, b, x0, iterations, eps = 1e-16, verbose = False):
    """ Apply the conjugate gradient method where op(x) performs Ax for
    Hermitian positive-definite matrix A."""
    r = b - op(x0)

    x = x0
    p = r
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

        if verbose:
            print("Iter", k, "||r|| =", norm_r)
                    
    return x, k



