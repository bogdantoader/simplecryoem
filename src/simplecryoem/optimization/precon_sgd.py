from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


# TODO: need a better and less confusing name for this
def precon_sgd(
    key,
    F,
    gradF,
    hvpF,
    w0,
    eta,
    D0,
    beta2,
    alpha,
    N_epoch=20,
    batch_size=None,
    N=1,
    adaptive_step_size=False,
    c=0.5,
    iter_display=1,
    adaptive_threshold=False,
):
    """Preconditioned SGD, where the preconditioner is estimated
    and improved using minibatches and Hutchinson's diagonal estimator.

    The adaptive step size uses line search with preconditioned
    Armijo condition. When the adaptive_step_size=False, the algorithm
    is the same as OASIS with fixed learning rate
    (see Jahani et al., 2021).
    """

    n = jnp.array(w0.shape)

    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N / batch_size

    key, subkey = random.split(key)

    # Since we only work with the diagonal of the Hessian, we
    # can simply write it as a matrix of whatever shape the input
    # is and element-wise multiply with it (instead of forming a
    # diagonal matrix and do matrix-vector multiplication).

    # This can be placed before the epoch loop starts or before each epoch
    # (or even between iterations within an epoch)
    # depending on when the Hessian changes
    nsamp = 0
    # D1sum = jnp.zeros(D0.shape)
    Davg = jnp.zeros(D0.shape)

    if adaptive_step_size:
        eta_max = eta

    # beta0 = beta2
    loss_list = []
    step_sizes = []
    iterates = [w0]
    for idx_epoch in range(1, N_epoch + 1):
        try:
            if idx_epoch % iter_display == 0:
                print(f"Epoch {idx_epoch}/{N_epoch}")

            key, subkey1, subkey2 = random.split(key, 3)

            # if idx_epoch == 1:
            #    beta2 = 1
            # else:
            #    beta2 = beta0

            idx_batches_grad = np.array_split(random.permutation(subkey1, N), N_batch)

            zkeys = random.split(key, len(idx_batches_grad))

            if adaptive_step_size:
                eta = eta_max
            if idx_epoch % iter_display == 0:
                pbar = tqdm(range(len(idx_batches_grad)))
            else:
                pbar = range(len(idx_batches_grad))
            for k in pbar:
                h_steps = 1

                z = random.rademacher(
                    zkeys[k - 1], jnp.flip(jnp.append(n, h_steps))
                ).astype(w0.dtype)

                # D1 = beta2 * D0 + (1-beta2) * (z * hvpF(w1, z, idx_batches_grad[k-1]))

                # D1sum = D1sum + (z * hvpF(w1, z, idx_batches_grad[k-1]))

                hvp_step = [zi * hvpF(w0, zi, idx_batches_grad[k - 1]) for zi in z]
                hvp_step = jnp.mean(jnp.array(hvp_step), axis=0)
                # D1sum += hvp_step
                # nsamp += 1
                # Davg = D1sum/nsamp

                nsamp0 = nsamp
                nsamp = nsamp + 1
                Davg0 = Davg

                Davg = Davg0 * nsamp0 / nsamp + hvp_step / nsamp

                # Exponential average between the 'guess' and the latest running avg.
                D1 = beta2 * D0 + (1 - beta2) * Davg

                Dhat = jnp.maximum(jnp.abs(D1), alpha)
                invDhat = 1 / Dhat

                Fw0 = F(w0, idx_batches_grad[k - 1])
                gradFw0 = gradF(w0, idx_batches_grad[k - 1])

                if adaptive_step_size:
                    eta = eta * 1.2
                    # eta = eta_max
                    # print("hello")

                w1 = w0 - eta * invDhat * jnp.conj(gradFw0)
                Fw1 = F(w1, idx_batches_grad[k - 1])

                if adaptive_step_size:
                    while Fw1 > Fw0 - c * eta * jnp.real(
                        jnp.sum(jnp.conj(gradFw0) * invDhat * gradFw0)
                    ):
                        eta = eta / 2
                        w1 = w0 - eta * invDhat * jnp.conj(gradFw0)
                        Fw1 = F(w1, idx_batches_grad[k - 1])

                w0 = w1
                D0 = D1
                loss_iter = Fw1
                step_sizes.append(eta)

                if idx_epoch % iter_display == 0:
                    pbar.set_postfix(loss=f"{loss_iter : .3e}", eta=f"{eta :.3e}")

            loss_epoch = []
            for k in pbar:
                loss_iter = F(w1, idx_batches_grad[k - 1])
                loss_epoch.append(loss_iter)
            loss_epoch = jnp.mean(jnp.array(loss_epoch))

            loss_list.append(loss_epoch)
            iterates.append(w1)

            if idx_epoch % iter_display == 0:
                print(f"  Loss = {loss_epoch : .8e}")
                print(f"  eta = {eta}")
                print(f"  alpha= {alpha}")

            if adaptive_threshold:
                alpha = alpha / 2
        except KeyboardInterrupt:
            break

    return w1, loss_list, iterates, step_sizes
