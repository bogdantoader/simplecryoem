import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from  matplotlib import pyplot as plt
import time

from src.utils import l2sq, generate_uniform_orientations_jax, generate_uniform_shifts,generate_gaussian_shifts
from src.jaxops import GradV, Loss



def conjugate_gradient(op, b, x0, iterations, eps = 1e-16, verbose = False):
    """ Apply the conjugate gradient method where op(x) performs Ax for
    Hermitian positive-definite matrix A."""
    r = b - op(x0)

    x = x0
    p = r
    x_all = []
    #for k in tqdm(range(iterations)):
    for k in range(iterations):
        rkTrk = jnp.sum(jnp.conj(r) * r)
        Ap = op(p)
        
        alpha = rkTrk / jnp.sum(jnp.conj(p) * Ap)

        x = x + alpha * p
        r = r - alpha * Ap 

        norm_r = jnp.linalg.norm(r.ravel(),2)
        if norm_r < eps:
            #print("  cg iter", k, "||r|| =", norm_r)
            return x, k

        beta = jnp.sum(jnp.conj(r) * r) / rkTrk
        p = r + beta * p

        x_all.append(x)
        if verbose and jnp.mod(k,10) == 0:
            print("  cg iter", k, "||r|| =", norm_r)
                    
    return x, k, x_all


def get_cg_vol_ops(grad_loss_volume_sum, angles, shifts, ctf_params, imgs_f, vol_shape, sigma = 1):
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

    Ab = - jnp.conj(Abfun)
    AA = lambda vv : jnp.conj(grad_loss_volume_sum(vv, angles, shifts, ctf_params, imgs_f, sigma)) + Ab

    return AA, Ab

# TODO: 
# 1. use jax.value_and_grad to speed things up (need to modify the jax operator classes)
def sgd(key, grad_func, loss_func, N, x0, alpha = 1, N_epoch = 10, batch_size = None, P = None, adaptive_step_size = False, c = 0.5, eps = 1e-15, verbose = False, iter_display = 1, mask = None):
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

    alpha : float
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
        N_batch = N/batch_size

    if P is None:
        P = jnp.ones(x0.shape)
        
    if mask is None:
        print("mask is None")
        mask = jnp.ones(x0.shape)

    x = x0
    loss_list = []
    grad_list = []

    iterates = [x0]

    if adaptive_step_size:
        alpha_max = alpha

    step_sizes = []
    for idx_epoch in range(N_epoch):
        try:
            # This is mostly useful when running a lot of epochs as deterministic gradient descent
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
                alpha = alpha_max

            for idx in pbar:

                #TODO: adapt the grad functions to return the function value too 
                #(since JAX can return it for free)
                gradx = grad_func(x, idx)
                fx = loss_func(x, idx)

                if adaptive_step_size:  
                    alpha = alpha * 1.2
                    #alpha = alpha_max

                x1 = x - alpha * P * jnp.conj(gradx)
                #x1 = x1 * mask # TEMPORARY
                #x1 = x1.at[jnp.abs(x1) > 1e4].set(0)
                
                fx1 = loss_func(x1, idx)

                if adaptive_step_size:

                    while fx1 > fx - c * alpha * jnp.real(jnp.sum(jnp.conj(gradx)* P * gradx)):
                        #print("AAA")
                        #print(fx1)
                        #print(fx - 1/2*alpha*jnp.real(jnp.sum(jnp.conj(gradx)*gradx)))

                        alpha = alpha / 2
                        #print(f"Halving step size. New alpha = {alpha}")

                        x1 = x - alpha * P * jnp.conj(gradx)
                        #x1 = x1 * mask # TEMPORARY
                        #x1 = x1.at[jnp.abs(x1) > 1e4].set(0)

                        fx1 = loss_func(x1, idx)

                step_sizes.append(alpha)

                x = x1
                loss_iter = fx1

                gradmax = jnp.max(jnp.abs(gradx))
                grad_epoch.append(gradmax)

                if idx_epoch % iter_display == 0:
                    pbar.set_postfix(grad=f"{gradmax :.3e}",
                            loss=f"{loss_iter :.3e}", 
                            alpha=f"{alpha :.3e}")

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
                print(f"  Loss = {loss_epoch :.3e}")

                print(f"  alpha = {alpha}")

            if grad_epoch < eps:
                break
        except KeyboardInterrupt:
            break

    return x, jnp.array(loss_list), jnp.array(grad_list), iterates, step_sizes


def get_sgd_vol_ops(gradv: GradV, loss: Loss, angles, shifts, ctf_params, imgs, sigma = 1):
    """Return the loss function, its gradient function and a its hessian-vector 
    product function in a way that allows subsampling of the gradient 
    (and Hessian) for SGD or higher order stochastic methods."""

    @jax.jit
    def hvp_loss_func(v, x, angles, shifts, ctf_params, imgs, sigma_noise):
        return jax.jvp(lambda u : gradv.grad_loss_volume_sum(u, angles, shifts, ctf_params, imgs, sigma_noise), (v,), (x,))[1]

    loss_func = lambda v, idx : loss.loss_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma) 

    grad_func = lambda v, idx : gradv.grad_loss_volume_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx],  sigma) 
    hvp_func = lambda v, x, idx : hvp_loss_func(v, x, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma)

    loss_px_func = lambda v, idx : loss.loss_px_sum(v, angles[idx], shifts[idx], ctf_params[idx], imgs[idx], sigma) 

    return grad_func, loss_func, hvp_func, loss_px_func


def kaczmarz(key, data, angles, fwd_model_vmap, loss_func, grad_loss_func, x0, N_epoch, N_batches, N_iter_cg = 2, eps_cg = 1e-7):
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
            #if verbose_cg:
            #    print(i)

            # Solve the least squares problem to apply the pseudoinverse   
            data_block = -fwd_model_vmap(x, angles[idx]) + data[idx]

            Ab = -grad_loss_func(zero, angles[idx], data_block)
            AA = lambda v : grad_loss_func(v, angles[idx], data_block) + Ab

            x_ls, k = conjugate_gradient(AA, Ab, zero, N_iter_cg, eps = eps_cg, verbose = verbose_cg)

            x = x + x_ls

            verbose_cg = False    

    return x



def oasis(key, F, gradF, hvpF, w0, eta, D0, beta2, alpha, N_epoch = 20, batch_size = None, N = 1, adaptive_step_size = False, c = 0.5, iter_display = 1):
    """OASIS with fixed learning rate, deterministic or stochastic."""

    n = jnp.array(w0.shape )

    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N/batch_size

    key, subkey = random.split(key)

    gradFw0 = gradF(w0, random.permutation(subkey, N)[:batch_size])
    Dhat0 = jnp.maximum(jnp.abs(D0), alpha)

    # Since we only work with the diagonal of the Hessian, we
    # can simply write it as a matrix of whatever shape the input 
    # is and element-wise multiply with it (instead of forming a
    # diagonal matrix and do matrix-vector multiplication).
    invDhat0 = 1/Dhat0
    w1 = w0 - eta * invDhat0 * jnp.conj(gradFw0)

    # This can be placed before the epoch loop starts or before each epoch
    # (or even between iterations within an epoch)
    # depending on when the Hessian changes
    nsamp = 0
    D1sum = jnp.zeros(D0.shape)
    Davg = jnp.zeros(D0.shape) 

    if adaptive_step_size:
        eta_max = eta

    beta0 = beta2
    loss_list = []
    for idx_epoch in range(1, N_epoch+1):
        if idx_epoch % iter_display == 0:
            print(f"Epoch {idx_epoch}/{N_epoch}")

        key, subkey1, subkey2 = random.split(key, 3)

        if idx_epoch == 1:
            beta2 = 1
        else:
            beta2 = beta0

        idx_batches_grad = np.array_split(random.permutation(subkey1, N), N_batch)

        zkeys = random.split(key, len(idx_batches_grad))

        if idx_epoch % iter_display == 0:
            pbar = tqdm(range(len(idx_batches_grad)))
        else:
            pbar = range(len(idx_batches_grad))
        for k in pbar:

            h_steps = 4

            z = random.rademacher(zkeys[k-1], jnp.flip(jnp.append(n, h_steps))).astype(w0.dtype)

            #D1 = beta2 * D0 + (1-beta2) * (z * hvpF(w1, z, idx_batches_grad[k-1]))

            #D1sum = D1sum + (z * hvpF(w1, z, idx_batches_grad[k-1]))


            hvp_step = [zi * hvpF(w1, zi, idx_batches_grad[k-1]) for zi in z]
            hvp_step = jnp.mean(jnp.array(hvp_step), axis=0)
            #D1sum += hvp_step 
            #nsamp += 1
            #Davg = D1sum/nsamp

            nsamp0 = nsamp
            nsamp = nsamp + 1
            Davg0 = Davg

            Davg = Davg0 * nsamp0/nsamp + hvp_step/nsamp


            # Exponential average between the 'guess' and the latest running average.
            D1 = beta2*D0 + (1-beta2)*Davg

            Dhat1 = jnp.maximum(jnp.abs(D1), alpha)       
            invDhat1 = 1/Dhat1

            Fw1 = F(w1, idx_batches_grad[k-1])
            gradFw1 = gradF(w1, idx_batches_grad[k-1])

            if adaptive_step_size:
                eta = eta * 1.2 
                #eta = eta_max
                #print("hello")

            w2 = w1 - eta * invDhat1 * jnp.conj(gradFw1)
            Fw2 = F(w2, idx_batches_grad[k-1])

            if adaptive_step_size:
                while Fw2 > Fw1 - c * eta * jnp.real(jnp.sum(jnp.conj(gradFw1)* invDhat1 * gradFw1)):
                    eta = eta / 2
                    w2 = w1 - eta * invDhat1 * jnp.conj(gradFw1)
                    Fw2 = F(w2, idx_batches_grad[k-1])

            w0 = w1
            w1 = w2
            D0 = D1

            #loss_iter = F(w1, idx_batches_grad[k-1])
            loss_iter = Fw2

            #loss_epoch.append(loss_iter)
            #print(loss_iter)     
            if idx_epoch % iter_display == 0:
                pbar.set_postfix(loss = f"{loss_iter : .3e}")

        loss_epoch = []
        for k in pbar:
            loss_iter = F(w1, idx_batches_grad[k-1])
            loss_epoch.append(loss_iter)
        loss_epoch = jnp.mean(jnp.array(loss_epoch)) 

        loss_list.append(loss_epoch)

        if idx_epoch % iter_display == 0:
            print(f"  Loss = {loss_epoch : .3e}")
            print(f"  eta = {eta}")

    return w1, jnp.array(loss_list)


def oasis_adaptive(key, F, gradF, hvpF, w0, eta0, D0, beta2, alpha, N_epoch = 20, batch_size = None, N = 1, iter_display = 1):
    """OASIS with adaptive learning rate, deterministic and stochastic."""

    n = jnp.array(w0.shape)
    
    if batch_size is None or batch_size == N:
        N_batch = 1
    else:
        N_batch = N/batch_size
   
    key, subkey0, subkey1 = random.split(key, 3)
    gradFw0 = gradF(w0, random.permutation(subkey0, N)[:batch_size])
    theta0 = jnp.inf
    Dhat0 = jnp.maximum(jnp.abs(D0), alpha)
                        
    invDhat0 = 1/Dhat0
    w1 = w0 - eta0 * jnp.conj(invDhat0 * gradFw0)
    
    gradFw1 = gradF(w1, random.permutation(subkey1, N)[:batch_size])

    nsamp = 1
    D1sum = D0
    loss_list = []
    for idx_epoch in range(1, N_epoch+1):
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
            
            h_steps = 20

            z = random.rademacher(zkeys[k-1], jnp.flip(jnp.append(n, h_steps))).astype(w0.dtype)
            #z = random.rademacher(zkeys[k-1], n).astype(w0.dtype)

            #D1 = beta2 * D0 + (1-beta2) * (z * hvpF(w1, z, idx_batches_grad[k-1]))

            hvp_step = [zi * hvpF(w1, zi, idx_batches_grad[k-1]) for zi in z]
            hvp_step = jnp.mean(jnp.array(hvp_step), axis=0)
            D1sum = D1sum + hvp_step 
            nsamp = nsamp + 1
            D1 = D1sum/nsamp

            Dhat1 = jnp.maximum(jnp.abs(D1), alpha)
            invDhat1 = 1/Dhat1

            tl = jnp.sqrt(1 + theta0)*eta0

            gradFw1 = gradF(w1, idx_batches_grad[k-1])
            gradFw0 = gradF(w0, idx_batches_grad[k-1])

            wd = w1-w0
            gfd = gradFw1 - gradFw0
            tr = 1/2 * jnp.sqrt(jnp.real(jnp.sum(jnp.conj(wd) * Dhat1 * wd)) / jnp.real(jnp.sum(jnp.conj(gfd) * invDhat1 * gfd))) 
            #print(jnp.max(jnp.imag(wd)))
            #print(jnp.max(jnp.imag(gfd)))

            eta1 = jnp.minimum(tl, tr) 

            w2 = w1 - eta1 * jnp.conj(invDhat1 * gradFw1)

            theta1 = eta1/eta0

            w0 = w1
            w1 = w2
            D0 = D1

            eta0 = eta1
            theta0 = theta1

            loss_iter = F(w1, idx_batches_grad[k-1])
            
            if idx_epoch % iter_display == 0:
                pbar.set_postfix(loss = f"{loss_iter : .3e}")
            

        loss_epoch = []
        for k in pbar:
            loss_iter = F(w1, idx_batches_grad[k-1])
            loss_epoch.append(loss_iter)
        loss_epoch = jnp.mean(jnp.array(loss_epoch)) 

        loss_list.append(loss_epoch)
        
        if idx_epoch % iter_display == 0:
            print(f"  Loss = {loss_epoch : .3e}")            
            
    return w1, jnp.array(loss_list)

