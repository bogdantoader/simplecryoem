from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from simplecryoem.algorithm import conjugate_gradient


def kaczmarz(
    key,
    data,
    angles,
    fwd_model_vmap,
    loss_func,
    grad_loss_func,
    x0,
    N_epoch,
    N_batches,
    N_iter_cg=2,
    eps_cg=1e-7,
):
    """Implementation of the randomized block Kaczmarz method
    introduced in [Needell & Tropp 2014]. Convenient for processing
    a batch of particle images at one time, where for each batch
    we solve a least squares problem using the CG algorithm above.

    Rough around the edges but working implementation. It might require
    a few adaptations to work with the cryoEM operators.l"""

    key, subkey = random.split(key)

    N = data.shape[0]
    index_permutations = random.permutation(subkey, N)
    block_indices = np.array(np.array_split(index_permutations, N_batches))
    print(f"{block_indices.shape[0]} iterations/epoch")

    x = x0
    zero = jnp.zeros(x0.shape)

    for ep in range(N_epoch):
        if ep % 1 == 0:
            print(f"Epoch {ep}")
            verbose_cg = True

        for i, idx in tqdm(enumerate(block_indices)):
            # if verbose_cg:
            #    print(i)

            # Solve the least squares problem to apply the pseudoinverse
            data_block = -fwd_model_vmap(x, angles[idx]) + data[idx]

            Ab = -grad_loss_func(zero, angles[idx], data_block)
            AA = lambda v: grad_loss_func(v, angles[idx], data_block) + Ab

            x_ls, k = conjugate_gradient(
                AA, Ab, zero, N_iter_cg, eps=eps_cg, verbose=verbose_cg
            )

            x = x + x_ls

            verbose_cg = False

    return x
