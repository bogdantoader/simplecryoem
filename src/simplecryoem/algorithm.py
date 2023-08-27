import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from simplecryoem.loss import Loss, GradV 


def conjugate_gradient(op, b, x0, iterations, eps=1e-16, verbose=False):
    """Apply the conjugate gradient method where op(x) performs Ax for
    Hermitian positive-definite matrix A."""
    r = b - op(x0)

    x = x0
    p = r
    x_all = []
    # for k in tqdm(range(iterations)):
    for k in range(iterations):
        rkTrk = jnp.sum(jnp.conj(r) * r)
        Ap = op(p)

        alpha = rkTrk / jnp.sum(jnp.conj(p) * Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        norm_r = jnp.linalg.norm(r.ravel(), 2)
        if norm_r < eps:
            # print("  cg iter", k, "||r|| =", norm_r)
            return x, k

        beta = jnp.sum(jnp.conj(r) * r) / rkTrk
        p = r + beta * p

        x_all.append(x)
        if verbose and jnp.mod(k, 10) == 0:
            print("  cg iter", k, "||r|| =", norm_r)

    return x, k, x_all


def get_cg_vol_ops(
    grad_loss_volume_sum, angles, shifts, ctf_params, imgs_f, vol_shape, sigma=1
):
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

    zero = jnp.zeros(vol_shape).astype(jnp.complex128)
    Abfun = grad_loss_volume_sum(zero, angles, shifts, ctf_params, imgs_f, sigma)

    Ab = -jnp.conj(Abfun)
    AA = (
        lambda vv: jnp.conj(
            grad_loss_volume_sum(vv, angles, shifts, ctf_params, imgs_f, sigma)
        )
        + Ab
    )

    return AA, Ab


# TODO:
# 1. use jax.value_and_grad to speed things up (need to modify the jax operator classes)
def sgd(
    key,
    grad_func,
    loss_func,
    N,
    x0,
    eta=1,
    N_epoch=10,
    batch_size=None,
    D0=None,
    adaptive_step_size=False,
    c=0.5,
    eps=1e-15,
    verbose=False,
    iter_display=1,
    adaptive_threshold=False,
    alpha=1e-10,
):
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

     eta: float
         Learning rate

     N_epochs : int
         Number of passes through the full dataset.

     batch_size : int
         Batch size

     P : nx x nx x nx
         Diagonal preconditioner (entry-wise multiplication of the gradient).

    """

    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N / batch_size

    if D0 is None:
        D0 = jnp.ones(x0.shape)

    D0hat = jnp.maximum(jnp.abs(D0), alpha)
    P = 1 / D0hat

    x = x0
    loss_list = []
    grad_list = []

    iterates = [x0]

    if adaptive_step_size:
        eta_max = eta

    step_sizes = []
    for idx_epoch in range(N_epoch):
        try:
            # This is mostly useful when running a lot of epochs as deterministic GD
            if idx_epoch % iter_display == 0:
                print(f"Epoch {idx_epoch+1}/{N_epoch} ", end="")

            key, subkey = random.split(key)
            idx_batches = np.array_split(random.permutation(subkey, N), N_batch)

            grad_epoch = []

            if idx_epoch % iter_display == 0:
                pbar = tqdm(idx_batches)
            else:
                pbar = idx_batches

            # Trying this: reset the step size at each epoch in case it goes
            # very bad (i.e. very small) during the previous epoch.
            if adaptive_step_size:
                eta = eta_max

            for idx in pbar:
                # TODO: adapt the grad functions to return the function value too
                # (since JAX can return it for free)
                gradx = grad_func(x, idx)
                fx = loss_func(x, idx)

                if adaptive_step_size:
                    eta = eta * 2  # 1.2
                    # eta = eta_max

                x1 = x - eta * P * jnp.conj(gradx)
                # x1 = x1 * mask # TEMPORARY
                # x1 = x1.at[jnp.abs(x1) > 1e4].set(0)

                fx1 = loss_func(x1, idx)

                if adaptive_step_size:
                    while fx1 > fx - c * eta * jnp.real(
                        jnp.sum(jnp.conj(gradx) * P * gradx)
                    ):
                        # print("AAA")
                        # print(fx1)
                        # print(fx - 1/2*eta*jnp.real(jnp.sum(jnp.conj(gradx)*gradx)))

                        eta = eta / 2
                        # print(f"Halving step size. New eta = {eta}")

                        x1 = x - eta * P * jnp.conj(gradx)
                        # x1 = x1 * mask # TEMPORARY
                        # x1 = x1.at[jnp.abs(x1) > 1e4].set(0)

                        fx1 = loss_func(x1, idx)

                step_sizes.append(eta)

                x = x1
                loss_iter = fx1

                gradmax = jnp.max(jnp.abs(gradx))
                grad_epoch.append(gradmax)

                if idx_epoch % iter_display == 0:
                    pbar.set_postfix(
                        grad=f"{gradmax :.3e}",
                        loss=f"{loss_iter :.3e}",
                        eta=f"{eta :.3e}",
                    )

            grad_epoch = jnp.mean(jnp.array(grad_epoch))

            loss_epoch = []
            for idx in pbar:
                loss_iter = loss_func(x, idx)
                loss_epoch.append(loss_iter)
            loss_epoch = jnp.mean(jnp.array(loss_epoch))

            grad_list.append(grad_epoch)
            loss_list.append(loss_epoch)

            iterates.append(x)

            if idx_epoch % iter_display == 0:
                print(f"  |Grad| = {grad_epoch :.3e}")
                print(f"  Loss = {loss_epoch :.8e}")

                print(f"  eta = {eta}")
                print(f"  alpha = {alpha}")

            if grad_epoch < eps:
                break

            if adaptive_threshold:
                alpha = alpha / 2
                D0hat = jnp.maximum(jnp.abs(D0), alpha)
                P = 1 / D0hat

        except KeyboardInterrupt:
            break

    return x, jnp.array(loss_list), jnp.array(grad_list), iterates, step_sizes


def get_sgd_vol_ops(
    gradv: GradV, loss: Loss, angles, shifts, ctf_params, imgs, sigma=1
):
    """Return the loss function, its gradient function and a its hessian-vector
    product function in a way that allows subsampling of the gradient
    (and Hessian) for SGD or higher order stochastic methods."""

    @jax.jit
    def hvp_loss_func(v, x, angles, shifts, ctf_params, imgs, sigma_noise):
        return jax.jvp(
            lambda u: gradv.grad_loss_volume_sum(
                u, angles, shifts, ctf_params, imgs, sigma_noise
            ),
            (v,),
            (x,),
        )[1]

    loss_func = lambda v, idx: loss.loss_sum(
        v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
    )

    grad_func = lambda v, idx: gradv.grad_loss_volume_sum(
        v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
    )
    hvp_func = lambda v, x, idx: hvp_loss_func(
        v, x, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
    )

    loss_px_func = lambda v, idx: loss.loss_px_sum(
        v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
    )

    return grad_func, loss_func, hvp_func, loss_px_func


def oasis(
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
    """OASIS with fixed learning rate, deterministic or stochastic."""

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
    """Original OASIS implementation with adaptive learning rate, deterministic
    and stochastic.
    As introduced in Jahani et al., 2021
    https://arxiv.org/pdf/2109.05198.pdf
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
    # D1sum = jnp.zeros(D0.shape)
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
