import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from simplecryoem.loss import Loss, GradV


# TODO:
# Use jax.value_and_grad to speed things up (need to modify the jax operator classes)
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
