from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def precon_sgd(
    key,
    F,
    gradF,
    hvpF,
    w0,
    eta,
    D0,
    beta,
    alpha,
    N_epoch=20,
    batch_size=None,
    N=1,
    adaptive_step_size=False,
    c=0.5,
    iter_display=1,
):
    """Preconditioned SGD, where the diagonal of the Hessian of the
    loss function is estimated and improved using minibatches and
    Hutchinson's diagonal estimator. This estimate is then used
    as a diagonal preconditioner.

    The adaptive step size uses line search with preconditioned Armijo condition.

    When the adaptive_step_size=False, the algorithm is the same as OASIS
    with fixed learning rate (see Jahani et al., 2021).

    Parameters:
    -----------
    key: jax.random.PRNGKey

    F : (x, idx) -> sum_i f_i(x), i=1,...N
         A function that takes a volume x and an array of indices idx
         and returns the sum of the loss functions at
         volume x and images indexed by idx.

    gradF : (x, idx) -> sum_i grad(f_i(x)), i=1,...N
         A function that takes a volume x and an array of indices idx
         and returns the sum of the gradients of the loss functions at
         volume x and images indexed by idx.

    hvpF: (v, x, idx) -> (sum_i Hessian(f_i(v)))^T x, i=1,...N
         A function that takes a volume v, a vector (i.e. another volume) x
         and an array of indices idx and returns
         Hessian-vector product of the loss function over the minibatch idx
         evaluated at v and applied to x.

    w0 : nx x nx x nx
         Initial volume

    eta: float
         (Initial) Step size.

    D0 : nx x nx x nx
         Initialization of the diagonal of the Hessian to be estimated
         during the run.

    beta : double
         Weight used in the exponential average between D0 and the
         current estimate of the Hessian diagonal.

    alpha : double
        Threshold the current estimate of the Hessian diagonal D from
        below if its entries are very small so that the preconditioner
        P = 1/D does not blow up.

    N_epoch : int
         Number of passes through the full dataset.

    batch_size : int
         Batch size. Set to None for deterministic gradient descent.

    N : int
         The total number of images/particles.

    adaptive_step_size: boolean
         Step size adaptation based on (precondition) Armijo condition.

    c : double
         Constant that determines the strength of the Armijo condition.

    eps :
        Stop when max(abs(gradient_epoch)) < eps.

    iter_display : int
        Print output every iter_display epochs. Default value 1.
        Set higher when running many epochs for deterministic gradient descent.

    Returns:
    --------
    w1: nx x nx x nx
        Final volume reconstruction

    loss_list : jnp.array(loss_list)
        Loss function values at the end of each epoch.

    iterates: N_epoch x nx x nx x nx
        Iterates from all epochs.

    step_sizes:
        All step sizes from all iterations (not epochs).

    """

    w_shape = jnp.array(w0.shape)

    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N / batch_size

    # Since we only work with the diagonal of the Hessian, we
    # can simply write it as a matrix of whatever shape the input
    # is and element-wise multiply with it (instead of forming a
    # diagonal matrix and do matrix-vector multiplication).

    # This can be placed before the epoch loop starts or before each epoch
    # (or even between iterations within an epoch)
    # depending on when the Hessian changes
    n_samp = 0
    Davg = jnp.zeros(D0.shape)

    if adaptive_step_size:
        eta_max = eta

    loss_list = []
    step_sizes = []
    iterates = [w0]
    for idx_epoch in range(1, N_epoch + 1):
        try:
            if idx_epoch % iter_display == 0:
                print(f"Epoch {idx_epoch}/{N_epoch}")

            key, subkey = random.split(key, 2)

            idx_batches = np.array_split(
                random.permutation(subkey, N), N_batch)

            key, *zkeys = random.split(key, 1 + len(idx_batches))

            if adaptive_step_size:
                eta = eta_max
            if idx_epoch % iter_display == 0:
                pbar = tqdm(range(len(idx_batches)))
            else:
                pbar = range(len(idx_batches))
            for k in pbar:
                # Draw h_steps Rademacher vectors
                h_steps = 1

                z = random.rademacher(
                    zkeys[k], jnp.insert(w_shape, 0, h_steps), dtype=w0.dtype
                )

                # Apply the Hutchinson step for each Rademacher vector
                hvp_step = [zi * hvpF(w0, zi, idx_batches[k]) for zi in z]
                hvp_step = jnp.mean(jnp.array(hvp_step), axis=0)

                # Update the running average of the Hutchinson steps
                n_samp0 = n_samp
                n_samp = n_samp + 1
                Davg0 = Davg
                Davg = Davg0 * n_samp0 / n_samp + hvp_step / n_samp

                # Exponential average between D0 and the latest running avg.
                D1 = beta * D0 + (1 - beta) * Davg

                # Thresholding
                Dhat = jnp.maximum(jnp.abs(D1), alpha)
                invDhat = 1 / Dhat

                # And finally the adaptive step size rule
                Fw0 = F(w0, idx_batches[k])
                gradFw0 = gradF(w0, idx_batches[k])

                if adaptive_step_size:
                    eta = eta * 1.2

                w1 = w0 - eta * invDhat * jnp.conj(gradFw0)
                Fw1 = F(w1, idx_batches[k])

                if adaptive_step_size:
                    while Fw1 > Fw0 - c * eta * jnp.real(
                        jnp.sum(jnp.conj(gradFw0) * invDhat * gradFw0)
                    ):
                        eta = eta / 2
                        w1 = w0 - eta * invDhat * jnp.conj(gradFw0)
                        Fw1 = F(w1, idx_batches[k])

                w0 = w1
                D0 = D1
                loss_iter = Fw1
                step_sizes.append(eta)

                if idx_epoch % iter_display == 0:
                    pbar.set_postfix(
                        loss=f"{loss_iter: .3e}", eta=f"{eta:.3e}")

            loss_epoch = []
            for k in pbar:
                loss_iter = F(w1, idx_batches[k])
                loss_epoch.append(loss_iter)
            loss_epoch = jnp.mean(jnp.array(loss_epoch))

            loss_list.append(loss_epoch)
            iterates.append(w1)

            if idx_epoch % iter_display == 0:
                print(f"  Loss = {loss_epoch: .8e}")
                print(f"  eta = {eta}")
                print(f"  alpha= {alpha}")

        except KeyboardInterrupt:
            break

    return w1, loss_list, iterates, step_sizes
