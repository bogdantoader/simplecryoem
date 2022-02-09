import time
import datetime
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import mrcfile

from src.algorithm import *
from src.utils import create_3d_mask, create_2d_mask, crop_fourier_volume, rescale_larger_grid, crop_fourier_images, generate_uniform_orientations
from src.jaxops import *
from src.fsc import plot_angles



def ab_initio(project_func, imgs, shifts_true, ctf_params, x_grid, use_sgd, N_iter = 100, N_vol_iter = 300, learning_rate = 1, batch_size = -1, P = None, N_samples = 40000, radius0 = 0.1, dr = 0.05, alpha = 0, interp_method = 'tri', opt_vol_first = False, verbose = True, save_to_file = True, out_dir = './'):
    """Ab initio reconstruction.

    Parameters:
    ----------
    imgs : N x nx*nx array
        The 2d images, vectorised.
    
    x_grid : [dx, nx]
        The Fourier grid of the images.

    alpha : regularisation parameter

    Returns:
    
    """

    assert(imgs.ndim == 2)

    N = imgs.shape[0]
    nx = jnp.sqrt(imgs.shape[1]).astype(jnp.int64)
    v0 = jnp.array(np.random.randn(nx,nx,nx) + 1j * np.random.randn(nx,nx,nx))
    v=v0

    if use_sgd:
        if batch_size == -1:
            batch_size = N
        if P == None:
            P = jnp.ones(v0.shape)

    if opt_vol_first:
        mask3d = jnp.ones([nx,nx,nx])
        _, grad_loss_volume_batched, grad_loss_volume_sum = get_jax_ops_iter(project_func, x_grid, mask3d, alpha, interp_method)

        angles = generate_uniform_orientations(N)

        if use_sgd:
            sgd_grad_func = get_sgd_vol_ops(grad_loss_volume_batched, angles, shifts_true, ctf_params, imgs)
            v = sgd(sgd_grad_func, N, v0, learning_rate, N_vol_iter, batch_size, P, verbose = verbose)
        else:
            AA, Ab = get_cg_vol_ops(grad_loss_volume_sum, angles, shifts_true, ctf_params, imgs, v0.shape)
            v, _ = conjugate_gradient(AA, Ab, v0, N_vol_iter, verbose = verbose)

        if verbose:
            plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
            plt.colorbar()
            plt.show()
   
    imgs = imgs.reshape([N, nx,nx])
    radius = radius0

    for idx_iter in range(N_iter):
        if verbose:
            print("Iter ", idx_iter)
   
        # The nx of the volume at the current iteration        
        mask3d = create_3d_mask(x_grid, (0,0,0), radius)
        nx_iter = jnp.sum(mask3d[0,0,:]).astype(jnp.int64)
        # Ensure that we work with even images so that all the masking stuff works
        if jnp.mod(nx_iter,2) == 1:
            nx_iter +=1

        # At the first iteration, we reduce the size (from v0) while 
        # afterwards, we increase it (frequency marching).
        if idx_iter == 0:
            v, _ = crop_fourier_volume(v, x_grid, nx_iter)
        else:
            v, _ = rescale_larger_grid(v, x_grid_iter, nx_iter) 

        # Crop the images to the right size
        imgs_iter, x_grid_iter = crop_fourier_images(imgs, x_grid, nx_iter)
        imgs_iter = imgs_iter.reshape([N,nx_iter*nx_iter])
        mask3d = create_3d_mask(x_grid_iter, (0,0,0),  radius)
        mask2d = mask3d[0].reshape(1,-1)

        if use_sgd and P is not None:
            P_iter, _ = crop_fourier_volume(P, x_grid, nx_iter)

        # Get the operators for the dimensions at this iteration.
        slice_func_array_angles_iter, grad_loss_volume_batched_iter, grad_loss_volume_sum_iter  = get_jax_ops_iter(project_func, x_grid_iter, mask3d, alpha, interp_method)

        # Sample the orientations
        t0 = time.time()    
        #angles = sample_new_angles_vmap(loss_func_angles, v*mask3d, shifts_true, ctf_params, imgs*mask2d, N_samples) 
        angles = sample_new_angles_cached(loss_func_imgs_batched, slice_func_array_angles_iter, v*mask3d, shifts_true, ctf_params, imgs_iter*mask2d, N_samples)    
        if verbose:
            print("  Time orientations sampling =", time.time()-t0)
        
        #TODO: make the above function return the loss numbers as well so they don't have to be recomputed below
        #loss_min = loss_func_sum(v*mask3d, angles, shifts_true, ctf_params, imgs)/jnp.sum(mask2d)
        #print("angles loss", loss_min)

        # Optimise volume
        t0 = time.time()
        v0 = jnp.array(np.random.randn(nx_iter,nx_iter,nx_iter) + 1j * np.random.randn(nx_iter,nx_iter,nx_iter))

        if use_sgd:
            sgd_grad_func_iter = get_sgd_vol_ops(grad_loss_volume_batched_iter, angles, shifts_true, ctf_params, imgs_iter*mask2d)
            v = sgd(sgd_grad_func_iter, N, v0, learning_rate, N_vol_iter, batch_size, P_iter, verbose = verbose)
        else:
            AA, Ab = get_cg_vol_ops(grad_loss_volume_sum_iter, angles, shifts_true, ctf_params, imgs_iter*mask2d, v0.shape)
            v, _ = conjugate_gradient(AA, Ab, v0, N_vol_iter, verbose = verbose)

        if verbose:
            print("  Time vol optimisation =", time.time()-t0)

        # Increase radius
        # TODO: makke this a parameter of the algorithm
        if jnp.mod(idx_iter, 20)==0:
            if verbose:
                print(datetime.datetime.now())
                print("  nx =", nx_iter)
                plot_angles(angles[:500])
                plt.show()


            if save_to_file:
                with mrcfile.new(out_dir + '/rec_iter_' + str(idx_iter) + '.mrc', overwrite=True) as mrc:
                    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
                    mrc.set_data(vr.astype(np.float32))
                radius += dr

            if v.shape[0] == nx:
                break

                                    



    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
    if save_to_file:
        with mrcfile.new(out_dir + '/rec_final.mrc', overwrite=True) as mrc:
                mrc.set_data(vr.astype(np.float32))

    return v, angles


def get_jax_ops_iter(project_func, x_grid, mask, alpha = 0, interp_method = 'tri'):
    slice_func,slice_func_array, slice_func_array_angles = get_slice_funcs(project_func, x_grid, mask, interp_method)
    loss_func, loss_func_batched, loss_func_sum, _ = get_loss_funcs(slice_func, alpha = alpha)
    grad_loss_volume, grad_loss_volume_batched, grad_loss_volume_sum = get_grad_v_funcs(loss_func, loss_func_sum)
    return slice_func_array_angles, grad_loss_volume_batched, grad_loss_volume_sum

# Cached angles sampling

def loss_func_imgs_batched(img0, imgs):
    """Compute the loss between img0 and each image in the imgs array."""
    return jax.vmap(l2sq, in_axes = (None, 0))(img0, imgs)


def get_min_loss_index(img0, imgs, loss_func_array):
    """Given img0 and the array imgs, return the index in imgs of the image
    with the lowest loss with img0."""

    loss = loss_func_array(img0, imgs) 
    return jnp.argmin(loss)


def get_min_loss_indices(imgs1, imgs2, loss_func_array):
    return jax.vmap(get_min_loss_index, in_axes=(0, None, None))(imgs1, imgs2, loss_func_array)

def sample_new_angles_cached(loss_func_imgs_batched, slice_func_array_angles, vol, shifts_true, ctf_params, imgs, N_samples):
    """This function assumes ctf_params and shifts_true are the same accros 
    the first dimension, so it only uses the first row of each."""

    ang_samples = generate_uniform_orientations(N_samples)
    imgs_sampled = slice_func_array_angles(vol, ang_samples, shifts_true[0], ctf_params[0])
    indices = get_min_loss_indices(imgs, imgs_sampled, loss_func_imgs_batched) 

    return ang_samples[indices]

# Non cached angles sampling

def get_loss_func_angles(loss_func):
    return jax.jit(jax.vmap(loss_func, in_axes = (None, 0, None, None, None)))

def sample_new_angles_one_img(loss_func_angles, vol, shifts_true, ctf_params, img, N_samples):
    ang_samples = generate_uniform_orientations(N_samples)
    loss = loss_func_angles(vol, ang_samples, shifts_true, ctf_params, img)
    li = jnp.argmin(loss)
    return ang_samples[li]

def sample_new_angles(loss_func_angles, vol, shifts_true, ctf_params, imgs, N_samples):
    angles = []
    for ai in range(N):
        ang_samples = generate_uniform_orientations(N_samples)
        loss = loss_func_angles(vol, ang_samples, shifts_true[ai], ctf_params[ai], imgs[ai])
        li = jnp.argmin(loss)
        angles.append(ang_samples[li])
    return jnp.array(angles) 

sample_new_angles_vmap = jax.vmap(sample_new_angles_one_img, in_axes = (None, None, 0, 0, 0, None))
                                                
