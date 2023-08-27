from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def oasis_adaptive(
    key,
    F,
    gradF,
    hvpF,
    w0,
    eta0,
    D0,
    beta2,
    alpha,
    N_epoch=20,
    batch_size=None,
    N=1,
    iter_display=1,
):
    """Original OASIS implementation with adaptive learning rate,
    deterministic and stochastic.

    As introduced in Jahani et al., 2021 https://arxiv.org/pdf/2109.05198.pdf
    See paper for parameter details.

    Minor adaptation to leverage that the Hessian does not change
    in our case so far.
    """

    n = jnp.array(w0.shape)

    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N / batch_size

    key, subkey0, subkey1 = random.split(key, 3)
    gradFw0 = gradF(w0, random.permutation(subkey0, N)[:batch_size])
    theta0 = jnp.inf
    Dhat0 = jnp.maximum(jnp.abs(D0), alpha)

    invDhat0 = 1 / Dhat0
    w1 = w0 - eta0 * jnp.conj(invDhat0 * gradFw0)

    gradFw1 = gradF(w1, random.permutation(subkey1, N)[:batch_size])

    nsamp = 0
    Davg = jnp.zeros(D0.shape)

    loss_list = []
    for idx_epoch in range(1, N_epoch + 1):
        if idx_epoch % iter_display == 0:
            print(f"Epoch {idx_epoch}/{N_epoch}")

        key, subkey1, subkey2 = random.split(key, 3)

        idx_batches_grad = np.array_split(random.permutation(subkey1, N), N_batch)

        zkeys = random.split(key, len(idx_batches_grad))

        if idx_epoch % iter_display == 0:
            pbar = tqdm(range(len(idx_batches_grad)))
        else:
            pbar = range(len(idx_batches_grad))
        for k in pbar:
            h_steps = 1

            z = random.rademacher(
                zkeys[k - 1], jnp.flip(jnp.append(n, h_steps))
            ).astype(w0.dtype)

            hvp_step = [zi * hvpF(w0, zi, idx_batches_grad[k - 1]) for zi in z]
            hvp_step = jnp.mean(jnp.array(hvp_step), axis=0)

            nsamp0 = nsamp
            nsamp = nsamp + 1
            Davg0 = Davg

            Davg = Davg0 * nsamp0 / nsamp + hvp_step / nsamp

            # Exponential average between the 'guess' and the latest running average.
            D1 = beta2 * D0 + (1 - beta2) * Davg
            # D1 = beta2 * D0 + (1-beta2) * (z * hvpF(w1, z, idx_batches_grad[k-1]))

            Dhat1 = jnp.maximum(jnp.abs(D1), alpha)
            invDhat1 = 1 / Dhat1

            tl = jnp.sqrt(1 + theta0) * eta0

            gradFw1 = gradF(w1, idx_batches_grad[k - 1])
            gradFw0 = gradF(w0, idx_batches_grad[k - 1])

            wd = w1 - w0
            gfd = gradFw1 - gradFw0
            tr = (
                1
                / 2
                * jnp.sqrt(
                    jnp.real(jnp.sum(jnp.conj(wd) * Dhat1 * wd))
                    / jnp.real(jnp.sum(jnp.conj(gfd) * invDhat1 * gfd))
                )
            )

            eta1 = jnp.minimum(tl, tr)

            w2 = w1 - eta1 * jnp.conj(invDhat1 * gradFw1)

            theta1 = eta1 / eta0

            w0 = w1
            w1 = w2
            D0 = D1

            eta0 = eta1
            theta0 = theta1

            loss_iter = F(w1, idx_batches_grad[k - 1])

            if idx_epoch % iter_display == 0:
                pbar.set_postfix(loss=f"{loss_iter : .3e}")

        loss_epoch = []
        for k in pbar:
            loss_iter = F(w1, idx_batches_grad[k - 1])
            loss_epoch.append(loss_iter)
        loss_epoch = jnp.mean(jnp.array(loss_epoch))

        loss_list.append(loss_epoch)

        if idx_epoch % iter_display == 0:
            print(f"  Loss = {loss_epoch : .3e}")

    return w1, jnp.array(loss_list)
