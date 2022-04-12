import time
import datetime
import pickle
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from matplotlib import pyplot as plt
import mrcfile

from src.algorithm import *
from src.utils import *
from src.jaxops import *
from src.fsc import plot_angles




def ab_initio(project_func, imgs, sigma_noise, shifts_true, ctf_params, x_grid, use_sgd, N_iter = 100, N_vol_iter = 300, learning_rate = 1, batch_size = -1, P = None, N_samples = 40000, radius0 = 0.1, dr = None, alpha = 0, eps_vol = 1e-16, interp_method = 'tri', opt_vol_first = True, verbose = True, save_to_file = True, out_dir = './'):
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

    # Determine the frequency marching step size, if not given 
    if dr is None:
        x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
        X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dr = r[1,1,1]
    if verbose:
        max_radius = x_grid[0]*x_grid[1]/2
        n_steps = (jnp.floor((max_radius-radius0)/dr) + 1).astype(jnp.int64)

        print("Fourier radius: " + str(max_radius))
        print("Starting radius: " + str(radius0))
        print("Frequency marching step size: " + str(dr))
        print("Number of frequency marching steps:", str(n_steps))
        print("------------------------------------\n")


    if use_sgd:
        if batch_size == -1:
            batch_size = N
        if P == None:
            P = jnp.ones([nx,nx,nx])

    if opt_vol_first:
        v, _ = initialize_ab_initio_vol(project_func, imgs, shifts_true, ctf_params, x_grid, N_vol_iter, eps_vol, sigma_noise, use_sgd, learning_rate, batch_size,  P, interp_method, verbose)
    else:    
        v = jnp.array(np.random.randn(nx,nx,nx) + np.random.randn(nx,nx,nx)*1j)

    imgs = imgs.reshape([N, nx,nx])
    radius = radius0

    # Reshaping sigma_noise this way so that we can apply crop_fourier_images 
    # at each iteration.
    sigma_noise = sigma_noise.reshape([1, nx, nx])

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
        sigma_noise_iter, _ = crop_fourier_images(sigma_noise, x_grid, nx_iter)
        sigma_noise_iter = sigma_noise_iter.reshape(-1)
        mask3d = create_3d_mask(x_grid_iter, (0,0,0),  radius)
        mask2d = mask3d[0].reshape(1,-1)

        v = v * mask3d

        if use_sgd and P is not None:
            P_iter, _ = crop_fourier_volume(P, x_grid, nx_iter)

        # Get the operators for the dimensions at this iteration.
        slice_func_array_angles_iter, grad_loss_volume_sum_iter, loss_func_angles, loss_func_batched0_iter, loss_func_sum_iter = get_jax_ops_iter(project_func, x_grid_iter, mask3d, alpha, interp_method)

        # Sample the orientations
        t0 = time.time()    
        #angles = sample_new_angles_vmap(loss_func_angles, v*mask3d, shifts_true, ctf_params, imgs_iter*mask2d, N_samples, sigma_noise_iter) 
        angles = sample_new_angles_cached(loss_func_imgs_batched, slice_func_array_angles_iter, v*mask3d, shifts_true, ctf_params, imgs_iter*mask2d, N_samples, sigma_noise_iter)    

        diagnostics = False

        if verbose:
            print("  Time orientations sampling =", time.time()-t0)
            
            if diagnostics:
                plot_angles(angles[:500])
                plt.show()


        #TODO: make the above function return the loss numbers as well so they don't have to be recomputed below
        #loss_min = loss_func_sum(v*mask3d, angles, shifts_true, ctf_params, imgs)/jnp.sum(mask2d)
        #print("angles loss", loss_min)

        # Optimise volume
        t0 = time.time()
        v0 = jnp.zeros([nx_iter,nx_iter,nx_iter])* 1j

        if use_sgd:
            sgd_grad_func_iter = get_sgd_vol_ops(grad_loss_volume_sum_iter, angles, shifts_true, ctf_params, imgs_iter*mask2d, sigma_noise_iter)
            v = sgd(sgd_grad_func_iter, N, v0, learning_rate, N_vol_iter, batch_size, P_iter, eps_vol, verbose = verbose)
        else:
            AA, Ab = get_cg_vol_ops(grad_loss_volume_sum_iter, angles, shifts_true, ctf_params, imgs_iter*mask2d, v0.shape, sigma_noise_iter)
            v, _ = conjugate_gradient(AA, Ab, v0, N_vol_iter, eps_vol, verbose = verbose)

        if verbose:
            print("  Time vol optimisation =", time.time()-t0)

            if diagnostics:
                #ff ,lf =  get_diagnostics_funs_iter(project_func, x_grid_iter, mask3d, alpha, interp_method)
                #fid = ff(v, angles, shifts_true, ctf_params, imgs_iter*mask2d, sigma_noise_iter)
                #reg = 1/2 * l2sq(v) * alpha
                #loss = lf(v, angles, shifts_true,ctf_params, imgs_iter*mask2d,sigma_noise_iter)
                #print("  fid =", fid)
                #print("  reg =", reg)
                #print("  loss =", loss)

                plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0])))
                #plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
                plt.colorbar()
                plt.show()


        if jnp.mod(idx_iter, 1)==0:
            if verbose:
                print(datetime.datetime.now())
                print("  nx =", nx_iter)

                plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0])))
                #plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
                plt.colorbar()
                plt.show()

                #plt.imshow(jnp.fft.fftshift((sigma_noise_iter*mask2d).reshape([nx_iter, nx_iter]))); 
                #plt.colorbar()
                #plt.show()

                plot_angles(angles[:500])
                plt.show()

            if save_to_file:
                with mrcfile.new(out_dir + '/rec_iter_' + str(idx_iter) + '.mrc', overwrite=True) as mrc:
                    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
                    mrc.set_data(vr.astype(np.float32))

                file = open(out_dir + '/rec_iter_' + str(idx_iter) + '_angles', 'wb')
                pickle.dump(angles, file)
                file.close()

                file3 = open(out_dir + '/rec_iter_' + str(idx_iter) + '_shifts', 'wb')
                pickle.dump(shifts, file3)
                file3.close()


        # Increase radius
        # TODO: make this a parameter of the algorithm
        if jnp.mod(idx_iter, 8)==0:
            radius += dr
            if v.shape[0] == nx:
                break

    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
    if save_to_file:
        with mrcfile.new(out_dir + '/rec_final.mrc', overwrite=True) as mrc:
                mrc.set_data(vr.astype(np.float32))

    return v, angles



def ab_initio_mcmc(
        key, 
        project_func,
        rotate_and_interpolate_func,
        apply_shifts_and_ctf_func,
        imgs, 
        sigma_noise, 
        ctf_params, 
        x_grid, 
        vol0 = None, 
        angles0 = None, 
        shifts0 = None,
        N_iter = 100, 
        learning_rate = 1, 
        sgd_batch_size = -1, 
        N_samples_vol = 100, 
        N_samples_angles_global = 1000, 
        N_samples_angles_local = 100, 
        N_samples_shifts = 1000,
        dt_list_hmc = [0.5], 
        sigma_perturb_list = jnp.array([1, 0.1, 0.01, 0.001]),
        L_hmc = 10, 
        radius0 = 0.1, 
        dr = None, 
        alpha = 0, 
        eps_vol = 1e-16, 
        B_list = [1],
        freq_marching_step_iters = 1,
        interp_method = 'tri', 
        opt_vol_first = True, 
        verbose = True,
        diagnostics = False,
        save_to_file = True, 
        out_dir = './'):
    """Ab initio reconstruction using MCMC.

    Parameters:
    ----------
    imgs : N x nx*nx array
        The 2d2images, vectorised.
    
    x_grid : [dx, nx]
        The Fourier grid of the images.

    alpha : regularisation parameter

    Returns:
    
    """

    assert(imgs.ndim == 3)

    N_batch_shape = jnp.array(imgs.shape[:2])
    N1 = N_batch_shape[0]
    N2 = N_batch_shape[1]
    nx = jnp.sqrt(imgs.shape[2]).astype(jnp.int64)

    # Determine the frequency marching step size, if not given 
    if dr is None:
        x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1/(x_grid[0]*x_grid[1]))
        X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dr = r[1,1,1]
    if verbose:
        max_radius = x_grid[0]*x_grid[1]/2
        n_steps = (jnp.floor((max_radius-radius0)/dr) + 1).astype(jnp.int64)

        print("Fourier radius: " + str(max_radius))
        print("Starting radius: " + str(radius0))
        print("Frequency marching step size: " + str(dr))
        print("Number of frequency marching steps:", str(n_steps))
        print("------------------------------------\n")


    if sgd_batch_size == -1:
        sgd_batch_size = N

    key, subkey = random.split(key)
    if vol0 is None and opt_vol_first:
        N_vol_iter = 3000
        v, angles, shifts = initialize_ab_initio_vol(key, project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, imgs, ctf_params, x_grid, N_vol_iter, eps_vol, sigma_noise, True, learning_rate, sgd_batch_size,  None, B_list, interp_method, verbose)

        if diagnostics:
            #plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
            plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0])))
            plt.colorbar()
            plt.show()
    elif vol0 is None:    
        v = jnp.array(np.random.randn(nx,nx,nx) + np.random.randn(nx,nx,nx)*1j)
    else:
        v = vol0

    #TODO: should have separate options to indicate that we don't want to estimate angles/shifts
    # or that we want to estimate them but start from shifts0, angles0. Same for vol0
    if shifts0 is not None:
        shifts = shifts0
    if angles0 is not None:
        angles = angles0

    imgs = imgs.reshape([N1, N2, nx,nx])
    radius = radius0

    # Reshaping sigma_noise this way so that we can apply crop_fourier_images 
    # at each iteration.
    sigma_noise = sigma_noise.reshape([1, nx, nx])

    nx_iter = 0
    recompile = True
    for idx_iter in range(N_iter):
        #if nx_iter == nx and jnp.mod(idx_iter, 7)==0:
        #   N_samples_angles = 1000
        #   N_samples_vol = 100


        if verbose:
            print("Iter ", idx_iter)
  
        if recompile:
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
            imgs_iter, x_grid_iter = crop_fourier_images_batch(imgs, x_grid, nx_iter)
            imgs_iter = imgs_iter.reshape([N1, N2 ,nx_iter*nx_iter])
            sigma_noise_iter, _ = crop_fourier_images(sigma_noise, x_grid, nx_iter)
            sigma_noise_iter = sigma_noise_iter.reshape(-1)
            mask3d = create_3d_mask(x_grid_iter, (0,0,0),  radius)
            mask2d = mask3d[0].reshape(1,-1)
            imgs_iter = imgs_iter*mask2d
            #v = v * mask3d

            #M_iter = 1/jnp.max(sigma_noise_iter)**2 * jnp.ones([nx_iter, nx_iter, nx_iter])
            M_iter = 1/jnp.max(sigma_noise)**2 * jnp.ones([nx_iter, nx_iter, nx_iter])
            # Get the operators for the dimensions at this iteration.
            slice_func_array_angles_iter, grad_loss_volume_sum_iter, loss_func_angles, loss_func_batched0_iter, loss_func_sum_iter, loss_proj_func_batched0_iter, rotate_and_interpolate_iter = get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid_iter, mask3d, alpha, interp_method)

            proposal_func_orientations_unif, proposal_func_orientations_pert, proposal_func_shifts, proposal_func_vol, proposal_func_vol_batch = get_jax_proposal_funcs(loss_func_batched0_iter, loss_proj_func_batched0_iter, loss_func_sum_iter, grad_loss_volume_sum_iter, sigma_noise_iter, B_list, dt_list_hmc, L_hmc, M_iter)

            if N1 > 1:
                proposal_func_vol = proposal_func_vol_batch


        key, key_angles, key_shifts = random.split(key, 3)

        # Sample the orientations
        print("Sampling orientations") 


        t0 = time.time()    
     
        if angles0 is None:
            # First, sample orientations uniformly on the sphere.

            #TODO: do the same for shifts
            if idx_iter < 48 or jnp.mod(idx_iter, 8) == 4:
                angles_new = []
                for i in jnp.arange(N1):
                    if verbose and N1 > 1:
                        print("batch ", i)
                    params_orientations = {'v':v, 'shifts':shifts[i], 'ctf_params':ctf_params[i], 'imgs_iter' : imgs_iter[i]}
                    _, r_samples_angles, samples_angles = mcmc(key_angles, proposal_func_orientations_unif, angles[i], N_samples_angles_global, params_orientations, N2, 1, verbose = True)
                    angles_new.append(samples_angles[N_samples_angles_global-2])
                angles = jnp.array(angles_new)


                if verbose:
                    print("  Time global orientations sampling =", time.time()-t0)
                    print("  mean(a_angles) =", jnp.mean(r_samples_angles), flush=True)

                    #plot_angles(angles[:500])
                    #plt.show()

            # And now sample local perturbations of the orientations.
            angles_new = []
            for i in jnp.arange(N1):
                if verbose and N1 > 1:
                    print("batch ", i)
                params_orientations = {'v':v, 'shifts':shifts[i], 'ctf_params':ctf_params[i], 'imgs_iter' : imgs_iter[i], 'sigma_perturb': sigma_perturb_list}
                _, r_samples_angles, samples_angles = mcmc(key_angles, proposal_func_orientations_pert, angles[i], N_samples_angles_local, params_orientations, N2, 1, verbose = True)
                angles_new.append(samples_angles[N_samples_angles_local-2])
            angles = jnp.array(angles_new)


            if verbose:
                print("  Time local orientations sampling =", time.time()-t0)
                print("  mean(a_angles) =", jnp.mean(r_samples_angles), flush=True)

                #plot_angles(angles[:500])
                #plt.show()

        #TODO: make this similar to the orientations batch sampling 
        # Sample the shifts - not working right now
        if shifts0 is None:
            print("Sampling shifts")
            #proj = rotate_and_interpolate_iter(jnp.array(v), angles)
            proj = jnp.array([rotate_and_interpolate_iter(v, angles[i]) for i in jnp.arange(angles.shape[0])])

            params_shifts = {'v':v, 'proj':proj}

            t0 = time.time()    
            _, r_samples_shifts, samples_shifts = mcmc(key_angles, proposal_func_shifts, shifts, N_samples_shifts, params_shifts, N_batch_shape, 1, verbose = True)
            shifts = samples_shifts[N_samples_shifts-2] 

            if verbose:
                print("  Time shifts sampling =", time.time()-t0)
                print("  mean(a_shifts) =", jnp.mean(r_samples_shifts), flush=True)


        # Sample the volume
        print("Sampling the volume")


        if N1 == 1:
            params_vol = {'angles':angles[0], 'shifts':shifts[0], 'ctf_params':ctf_params[0], 'imgs_iter':imgs_iter[0]}
        else:
            params_vol = {'angles':angles, 'shifts':shifts, 'ctf_params':ctf_params, 'imgs_iter':imgs_iter}

        t0 = time.time()
        v_hmc_mean, r_hmc, v_hmc_samples = mcmc(subkey, proposal_func_vol, v, N_samples_vol, params_vol, save_samples = -1)
        #v = v_hmc_mean 
        #v = v_hmc_samples[N_samples_vol-2] 
        v = v_hmc_samples[0] 
        v = jnp.array(v*mask3d)


        if verbose:
            print("  Time volume sampling =", time.time()-t0)
            print("  mean(a_vol) =", jnp.mean(r_hmc), flush=True)

            #if diagnostics:
                #ff ,lf =  get_diagnostics_funs_iter(project_func, x_grid_iter, mask3d, alpha, interp_method)
                #fid = ff(v, angles, shifts, ctf_params, imgs_iter, sigma_noise_iter)
                #reg = 1/2 * l2sq(v) * alpha
                #loss = lf(v, angles, shifts,ctf_params, imgs_iter,sigma_noise_iter)
                #print("  fid =", fid)
                #print("  reg =", reg)
                #print("  loss =", loss)

                #plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0])))
                #plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
                #plt.colorbar()
                #plt.show()



        
        if jnp.mod(idx_iter, freq_marching_step_iters)==0 and verbose:
            print(datetime.datetime.now())
            print("  nx =", nx_iter, flush=True)

            if diagnostics:
                plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0]*mask3d[:,:,0])))
                plt.colorbar()
                plt.show()

                plot_angles(angles[0, :500])
                plt.show()

        if jnp.mod(idx_iter, 1)==0 and save_to_file:
            with mrcfile.new(out_dir + '/rec_iter_' + str(idx_iter) + '.mrc', overwrite=True) as mrc:
                vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
                mrc.set_data(vr.astype(np.float32))

            #TODO: should probably print to a star file instead
            file = open(out_dir + '/rec_iter_' + str(idx_iter) + '_angles', 'wb')
            pickle.dump(angles, file)
            file.close()

            file3 = open(out_dir + '/rec_iter_' + str(idx_iter) + '_shifts', 'wb')
            pickle.dump(shifts, file3)
            file3.close()


        # Increase radius
        if jnp.mod(idx_iter,  freq_marching_step_iters)==0:
            radius += dr
            recompile = True

            if nx_iter == nx:
                break

        else:
            recompile = False


    # At the end, take the mean 
    # TODO: take the mean over all samples, not only the last run
    # A bit tricky because they have different dimensions. Should be able
    # to just same each Iter's mean, enlarge to full size, and then average 
    # all
    v = v_hmc_mean 
    v = v*mask3d

    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
    if save_to_file:
        with mrcfile.new(out_dir + '/rec_final.mrc', overwrite=True) as mrc:
                mrc.set_data(vr.astype(np.float32))

    #return v, angles, shifts, samples_angles, r_samples_angles,  v_hmc_samples , r_hmc
    return v, angles, shifts



def initialize_ab_initio_vol(key, 
        project_func, 
        rotate_and_interpolate_func,
        apply_shifts_and_ctf_func, 
        imgs, 
        ctf_params, 
        x_grid, 
        N_vol_iter, 
        eps_vol, 
        sigma_noise = 1, 
        use_sgd = True, 
        learning_rate = 1, 
        batch_size = -1,  
        P = None, 
        B = 1, 
        interp_method = 'tri', 
        verbose = True):
    if verbose:
        print("Initialitsing volume")
    
    N1 = imgs.shape[0]
    N2 = imgs.shape[1]
    nx = jnp.sqrt(imgs.shape[2]).astype(jnp.int64)

    v0 = jnp.array(np.random.randn(nx,nx,nx) + np.random.randn(nx,nx,nx)*1j)
    mask3d = jnp.ones([nx,nx,nx])

    _, grad_loss_volume_sum, _, _, _,_,_ = get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid, mask3d, 0, interp_method)

    key1, key2 = random.split(key)

    angles = generate_uniform_orientations_jax_batch(key, N1, N2)
    shifts = jnp.zeros([N1, N2, 2]) #generate_uniform_shifts(key, N, B)
    #shifts = generate_gaussian_shifts(key, N, B)


    #grad_loss_volume_batched_sum = lambda v, a, s, c, imgs, sig : grad_loss_volume_batched(v, a, s, c, imgs, sig) / a.shape[0]

    if use_sgd:
        sgd_grad_func = get_sgd_vol_ops(grad_loss_volume_sum, angles[0], shifts[0], ctf_params[0], imgs[0], sigma_noise)
        v = sgd(sgd_grad_func, N2, v0, learning_rate, N_vol_iter, batch_size, P, eps_vol, verbose = verbose)
    else:
        AA, Ab = get_cg_vol_ops(grad_loss_volume_sum, angles, shifts, ctf_params, imgs*mask2d, v0.shape, sigma_noise)
        v, _ = conjugate_gradient(AA, Ab, v0, N_vol_iter, eps_vol, verbose = verbose)


    return v, angles, shifts


def get_diagnostics_funs_iter(project_func, x_grid, mask, alpha = 0, interp_method = 'tri'):
    slice_func,slice_func_array, slice_func_array_angles = get_slice_funcs(project_func, x_grid, mask, interp_method)
    loss_func, loss_func_batched, loss_func_sum = get_loss_funcs(slice_func, alpha = alpha)
    fid_func, fid_func_batched, fid_func_sum = get_loss_funcs(slice_func, alpha = 0)

    return fid_func_sum, loss_func_sum



def get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid, mask, alpha = 0, interp_method = 'tri'):
    slice_func,slice_func_array, slice_func_array_angles = get_slice_funcs(project_func, x_grid, mask, interp_method)
    loss_func, loss_func_batched, loss_func_sum  = get_loss_funcs(slice_func, alpha = alpha)
    grad_loss_volume, grad_loss_volume_sum = get_grad_v_funcs(loss_func, loss_func_sum)
    loss_func_angles = get_loss_func_angles(loss_func)
    _, loss_func_batched0, _ = get_loss_funcs(slice_func, alpha = 0)
    loss_proj_func_batched0 = get_loss_proj_funcs(apply_shifts_and_ctf_func, x_grid, alpha = 0)

    @jax.jit
    def rotate_and_interpolate(v, angles):
        return jax.vmap(rotate_and_interpolate_func, in_axes=(None,0,None,None))(v*mask, angles, x_grid, x_grid)

    return slice_func_array_angles, grad_loss_volume_sum, loss_func_angles, loss_func_batched0, loss_func_sum, loss_proj_func_batched0, rotate_and_interpolate


#TODO: also write a version of this function that returns the
# jitted non batched version to work with small datasets
# (e.g. relion tutorial or simulated), in which case it would
# be faster
def get_jax_proposal_funcs(loss_func_batched0_iter, loss_proj_func_batched0_iter, loss_func_sum_iter, grad_loss_volume_sum_iter, sigma_noise_iter, B_list, dt_list_hmc, L_hmc, M_iter):

    
    def proposal_func_orientations_batch(key, angles0, logPiX0, v, shifts):
        #logPi = lambda a : -loss_func_batched0_iter(v, a, shifts, ctf_params, imgs_iter, sigma_noise_iter)

        def logPi(a):
            a_i = [-loss_func_batched0_iter(v, a[i], shifts[i], ctf_params[i], imgs_iter[i], sigma_noise_iter) for i in jnp.arange(angles0.shape[0])]
            return jnp.array(a_i)

        #logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
        #    true_fun = lambda _ : logPi(angles0),
        #    false_fun = lambda _ : logPiX0,
        #    operand = None)

        if jnp.sum(logPiX0) == jnp.inf:
            logPiX0 = logPi(angles0)

        N1 = angles0.shape[0]
        N2 = angles0.shape[1]
        angles1 = generate_uniform_orientations_jax_batch(key, N1, N2)

        logPiX1 = logPi(angles1)
        r = jnp.exp(logPiX1 - logPiX0)

        return angles1, r, logPiX1, logPiX0
  
    #TODO: do this thing for shifts too and delete the the batch proposal func for orientations and shifts
    def proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_orientations_func, params_orientations):
        logPi = lambda a : -loss_func_batched0_iter(v, a, shifts, ctf_params, imgs_iter, sigma_noise_iter)

        angles1 = generate_orientations_func(key, angles0, **params_orientations)

        logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
            true_fun = lambda _ : logPi(angles0),
            false_fun = lambda _ : logPiX0,
            operand = None)

        logPiX1 = logPi(angles1)
        r = jnp.exp(logPiX1 - logPiX0)

        return angles1, r, logPiX1, logPiX0 

    @jax.jit
    def proposal_func_orientations_uniform(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter):
        empty_params = {}
        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_uniform_orientations_jax, empty_params)

    @jax.jit
    def proposal_func_orientations_perturb(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, sigma_perturb):
        key, subkey = random.split(key)
        sig_p = random.permutation(subkey, sigma_perturb)[0]
        orient_params = {'sig' : sig_p}

        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_perturbed_orientations, orient_params)

    #@jax.jit
    def proposal_func_shifts(key, shifts0, logPiX0, v, proj):
        #logPi = lambda sh : -loss_proj_func_batched0_iter(v, proj, sh, ctf_params, imgs_iter, sigma_noise_iter)

        def logPi(sh):
            sh_i = [-loss_proj_func_batched0_iter(v, proj[i], sh[i], ctf_params[i], imgs_iter[i], sigma_noise_iter) for i in jnp.arange(shifts0.shape[0])]
            return jnp.array(sh_i)

        #logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
        #    true_fun = lambda _ : logPi(shifts0),
        #    false_fun = lambda _ : logPiX0,
        #    operand = None)

        if jnp.sum(logPiX0) == jnp.inf:
            logPiX0 = logPi(shifts0)

        key, subkey =  random.split(key)
        B0 = random.permutation(subkey, B_list)[0]

        N1 = shifts0.shape[0]
        N2 = shifts0.shape[1]
        shifts1 = generate_gaussian_shifts_batch(key, N1, N2, B0)

        logPiX1 = logPi(shifts1)
        r = jnp.exp(logPiX1 - logPiX0)

        return shifts1, r, logPiX1, logPiX0
   

    def proposal_func_vol_batch(key, v0, logPiX0, angles, shifts, ctf_params, imgs_iter):
        def logPi_vol(v):
            loss_i = [-loss_func_sum_iter(v, angles[i], shifts[i], ctf_params[i], imgs_iter[i], sigma_noise_iter) for i in jnp.arange(angles.shape[0])]
            return jnp.mean(jnp.array(loss_i), axis=0)

        def gradLogPi_vol(v): 
            grad_i = [-jnp.conj(grad_loss_volume_sum_iter(v, angles[i], shifts[i], ctf_params[i], imgs_iter[i], sigma_noise_iter)) for i in jnp.arange(angles.shape[0])]
            return jnp.mean(jnp.array(grad_i), axis=0)
 
        # Moved this to the proposal_hmc function
        #if logPiX0 == jnp.inf:
        #    logPiX0 = logPi_vol(v0)

        return proposal_hmc(key, v0, logPiX0, logPi_vol, gradLogPi_vol, dt_list_hmc, L_hmc, M_iter)


    @jax.jit
    def proposal_func_vol(key, v0, logPiX0, angles, shifts, ctf_params, imgs_iter):
        logPi_vol = lambda v : -loss_func_sum_iter(v, angles, shifts, ctf_params, imgs_iter, sigma_noise_iter)
        gradLogPi_vol = lambda v : -jnp.conj(grad_loss_volume_sum_iter(v, angles, shifts, ctf_params, imgs_iter, sigma_noise_iter))

        # Moved the below to the proposal_hmc function for generality.
        #logPiX0 = jax.lax.cond(logPiX0 == jnp.inf,
        #    true_fun = lambda _ : logPi_vol(v0),
        #    false_fun = lambda _ : logPiX0,
        #    operand = None)
        
        return proposal_hmc(key, v0, logPiX0, logPi_vol, gradLogPi_vol, dt_list_hmc, L_hmc, M_iter)


    return proposal_func_orientations_uniform, proposal_func_orientations_perturb, proposal_func_shifts, proposal_func_vol, proposal_func_vol_batch





# Cached angles sampling

def loss_func_imgs_batched(img0, imgs, sigma):
    """Compute the loss between img0 and each image in the imgs array."""
    return jax.vmap(wl2sq, in_axes = (None, 0, None))(img0, imgs, 1/sigma**2)


def get_min_loss_index(img0, imgs, loss_func_array, sigma):
    """Given img0 and the array imgs, return the index in imgs of the image
    with the lowest loss with img0."""

    loss = loss_func_array(img0, imgs, sigma) 
    return jnp.argmin(loss)


def get_min_loss_indices(imgs1, imgs2, loss_func_array, sigma):
    return jax.vmap(get_min_loss_index, in_axes=(0, None, None, None))(imgs1, imgs2, loss_func_array, sigma)

def sample_new_angles_cached(loss_func_imgs_batched, slice_func_array_angles, vol, shifts_true, ctf_params, imgs, N_samples, sigma_noise):
    """This function assumes ctf_params and shifts_true are the same accros 
    the first dimension, so it only uses the first row of each."""

    ang_samples = generate_uniform_orientations(N_samples)
    imgs_sampled = slice_func_array_angles(vol, ang_samples, shifts_true[0], ctf_params[0])
    indices = get_min_loss_indices(imgs, imgs_sampled, loss_func_imgs_batched, sigma_noise) 

    return ang_samples[indices]

# Non cached angles sampling

def get_loss_func_angles(loss_func):
    return jax.jit(jax.vmap(loss_func, in_axes = (None, 0, None, None, None, None)))

def sample_new_angles_one_img(loss_func_angles, vol, shifts_true, ctf_params, img, N_samples, sigma_noise):
    ang_samples = generate_uniform_orientations(N_samples)
    loss = loss_func_angles(vol, ang_samples, shifts_true, ctf_params, img, sigma_noise)
    li = jnp.argmin(loss)
    return ang_samples[li]

def sample_new_angles(loss_func_angles, vol, shifts_true, ctf_params, imgs, N_samples, sigma_noise):
    angles = []
    for ai in range(N):
        ang_samples = generate_uniform_orientations(N_samples)
        loss = loss_func_angles(vol, ang_samples, shifts_true[ai], ctf_params[ai], imgs[ai], sigma_noise)
        li = jnp.argmin(loss)
        angles.append(ang_samples[li])
    return jnp.array(angles) 

sample_new_angles_vmap = jax.vmap(sample_new_angles_one_img, in_axes = (None, None, 0, 0, 0, None, None))

# Not really working for now
def gradLogPi_split(v, angles, shifts, ctf_params, imgs_f, sigma_noise, grad_loss_volume_sum, number_of_batches):
   
    # IMPORTANT to use np.array_split and not the jnp
    # version, as we don't want to load the images into GPU
    # memory just yet.
    angles_b = np.array_split(angles, number_of_batches)
    shifts_b = np.array_split(shifts, number_of_batches)
    ctf_params_b = np.array_split(ctf_params, number_of_batches)
    imgs_f_b = np.array_split(imgs_f, number_of_batches)

    grad_loss_vol = [grad_loss_volume_sum(v, angles_b[i], shifts_b[i], ctf_params_b[i], imgs_f_b[i], sigma_noise) 
                     for i in range(len(angles_b))]

    return -jnp.conj(jnp.sum(jnp.array(grad_loss_vol))/angles.shape[0])
                                               


