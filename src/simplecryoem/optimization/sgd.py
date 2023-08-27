import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .loss import Loss, GradV


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
    """Stochastic Gradient Descent

    Parameters:
    -----------
    key: jax.random.PRNGKey

    grad_func : (x, idx) -> sum_i grad(f_i(x)), i=1,...N
         A function that takes a volume x and an array of indices idx
         and returns the sum of the gradients of the loss functions at
         volume x and images indexed by idx.

    loss_func: (x, idx) -> sum_i f_i(x), i=1,...N
         A function that takes a volume x and an array of indices idx
         and returns the sum of the loss functions at
         volume x and images indexed by idx.

    N : int
         The total number of images/particles.

    x0 : nx x nx x nx
         Starting volume

    eta: float
         (Initial) Step size.

    N_epoch : int
         Number of passes through the full dataset.

    batch_size : int
         Batch size. Set to None for deterministic gradient descent.

    D0 : nx x nx x nx
         Apply the diagonal preconditioner P = 1/max(abs(D0), alpha)
         Diagonal preconditioner (entry-wise multiplication of the gradient).
         See alpha below.

    adaptive_step_size: boolean
         Step size adaptation based on (precondition) Armijo condition.

    c : double
         Constant that determines the strength of the Armijo condition.

    eps :
        Stop when max(abs(gradient_epoch)) < eps.

    verbose :
        Not actually used.

    iter_display : int
        Print output every iter_display epochs. Default value 1.
        Set higher when running many epochs for deterministic gradient descent.

    adaptive_threshold : boolean
        Adpative rule for adjusting the preconditioner threshold alpha.

    alpha : double
        Threshold D0 from below if its entries are very small so that the
        preconditioner P = 1/D0 does not blow up.

    Returns:
    --------
    x: nx x nx x nx
        Final volume reconstruction

    loss_list : jnp.array(loss_list)
        Loss function values at the end of each epoch.

    grad_list : jnp.array(grad_list)
        Mean values of the loss function gradients at the end of each epoch.
        (See the code for the exact details)

    iterates: N_epoch x nx x nx x nx
        Iterates from all epochs.

    step_sizes:
        All step sizes from all iterations (not epochs).
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

            # Reset the step size at each epoch in case it goes
            # very bad (i.e. very small) during the previous epoch.
            if adaptive_step_size:
                eta = eta_max

            for idx in pbar:
                # TODO: adapt the grad functions to return the function value too
                # (since JAX can return it for free)
                gradx = grad_func(x, idx)
                fx = loss_func(x, idx)

                if adaptive_step_size:
                    eta = eta * 2

                x1 = x - eta * P * jnp.conj(gradx)
                fx1 = loss_func(x1, idx)

                if adaptive_step_size:
                    while fx1 > fx - c * eta * jnp.real(
                        jnp.sum(jnp.conj(gradx) * P * gradx)
                    ):
                        eta = eta / 2
                        x1 = x - eta * P * jnp.conj(gradx)
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
    """Return the loss function, its gradient function and its hessian-vector
    product function in a way that allows subsampling of the gradient
    (and Hessian) for SGD or higher order stochastic methods.

    Parameters:
    -----------
    gradv: GradV object
    loss: Loss object
    angles: N x 3 array of Euler angles
    shifts: N x 2 array of shifts
    ctf_params: array of CTF parameters
    imgs: array of images
    sigma: noise standard deviation

    Returns:
    --------
    grad_func: function computing the gradient of the loss function
        over a minibatch of images, given
        (v, idx): volume and array of image indices
    loss_func: the loss function at volume v over a minibatch given by idx
    hvp_func: Hessian-vector product function
        Hessian of the loss function at volume v and minibatch idx
        applied to vector x
    loss_px_punc: Pixel-wise loss function at volume v and minibatch idx

    """

    @jax.jit
    def hvp_loss_func(v, x, angles, shifts, ctf_params, imgs, sigma_noise):
        return jax.jvp(
            lambda u: gradv.grad_loss_volume_sum(
                u, angles, shifts, ctf_params, imgs, sigma_noise
            ),
            (v,),
            (x,),
        )[1]

    def loss_func(v, idx):
        return loss.loss_sum(
            v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
        )

    def grad_func(v, idx):
        return gradv.grad_loss_volume_sum(
            v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
        )

    def hvp_func(v, x, idx):
        return hvp_loss_func(
            v, x, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
        )

    def loss_px_func(v, idx):
        return loss.loss_px_sum(
            v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma
        )

    return grad_func, loss_func, hvp_func, loss_px_func
