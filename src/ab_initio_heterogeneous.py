import time
import datetime
import pickle
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from matplotlib import pyplot as plt
import mrcfile
from jax.scipy.special import gammaln

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
        v, _ = initialize_ab_initio_vol(project_func, imgs, shifts_true, ctf_params, x_grid, N_vol_iter, eps_vol, sigma_noise, learning_rate, batch_size,  P, interp_method, verbose)
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
  
        print(f"v.shape = {v.shape}")

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
        slice_func_array_angles_iter, grad_loss_volume_sum_iter, grad_loss_volume_sum_z_iter, loss_func_angles, loss_func_batched0_iter, loss_func_sum_iter = get_jax_ops_iter(project_func, x_grid_iter, mask3d, alpha, interp_method)

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
        z0 = None,
        N_iter = 100, 
        learning_rate = 1, 
        sgd_batch_size = -1, 
        N_samples_vol = 100, 
        N_samples_angles_global = 1000, 
        N_samples_angles_local = 100, 
        N_samples_shifts = 1000,
        N_samples_z = 100,
        dt_list_hmc = [0.5], 
        sigma_perturb_list = jnp.array([1, 0.1, 0.01, 0.001]),
        L_hmc = 10, 
        radius0 = 0.1, 
        dr = None, 
        alpha = 0, 
        eps_vol = 1e-16, 
        B_list = [1],
        K = 1,
        alpha_d = [1],
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

        v, angles, shifts, z = initialize_ab_initio_vol(key, project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, imgs, ctf_params, x_grid, N_vol_iter, eps_vol, sigma_noise, learning_rate, sgd_batch_size,  B_list, K, interp_method, verbose)

        if diagnostics:
            #plt.imshow(jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[0,:,:]))))
            if K == 1:
                plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0])))
            else:
                plt.imshow(jnp.abs(jnp.fft.fftshift(v[0, :,:,0])))
            plt.colorbar()
            plt.show()
    elif vol0 is None:    
        v = jnp.array(np.random.randn(nx,nx,nx) + np.random.randn(nx,nx,nx)*1j)
    else:
        v = vol0

    print(f"v.shape after initialization = {v.shape}")

    #TODO: should have separate options to indicate that we don't want to estimate angles/shifts
    # or that we want to estimate them but start from shifts0, angles0. Same for vol0
    if shifts0 is not None:
        shifts = shifts0
    if angles0 is not None:
        angles = angles0

    if z0 is not None:
        z = z0


    imgs = imgs.reshape([N1, N2, nx,nx])
    radius = radius0

    # Reshaping sigma_noise this way so that we can apply crop_fourier_images 
    # at each iteration.
    sigma_noise = sigma_noise.reshape([1, nx, nx])

    nx_iter = 0
    recompile = True
    for idx_iter in range(1, N_iter+1):
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
            if idx_iter == 1:
                if K == 1:
                    v, _ = crop_fourier_volume(v, x_grid, nx_iter)
                else:
                    v_list = []
                    for i in jnp.arange(K):
                        v_i, _ = crop_fourier_volume(v[i], x_grid, nx_iter)
                        v_list.append(v_i)
                    v = jnp.array(v_list)
                    
            else:
                if K == 1:
                    v, _ = rescale_larger_grid(v, x_grid_iter, nx_iter) 
                else:
                    v_list = []
                    for i in jnp.arange(K):
                        v_i, _ = rescale_larger_grid(v[i], x_grid_iter, nx_iter) 
                        v_list.append(v_i)
                    v = jnp.array(v_list)

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
            slice_func_array_angles_iter, grad_loss_volume_sum_iter, grad_loss_volume_sum_z_iter, loss_func_angles, loss_func_batched0_iter, loss_func_sum_iter, loss_func_sum_z_iter, loss_func_batched_z_iter, loss_proj_func_batched0_iter, rotate_and_interpolate_iter = get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid_iter, mask3d, alpha, interp_method)

            proposal_func_orientations_unif, proposal_func_orientations_pert, proposal_func_shifts, proposal_func_vol, proposal_func_vol_batch = get_jax_proposal_funcs(loss_func_batched_z_iter, loss_proj_func_batched0_iter, loss_func_sum_iter, grad_loss_volume_sum_iter, sigma_noise_iter, B_list, dt_list_hmc, L_hmc, M_iter)


            proposal_z, proposal_angles_z, proposal_angles_z_mtm = get_class_proposal_func(loss_func_batched_z_iter, loss_func_sum_z_iter, sigma_noise_iter, alpha_d, K)

            if N1 > 1:
                proposal_func_vol = proposal_func_vol_batch


        key, key_angles, key_shifts, key_z = random.split(key, 4)


        # Sample the orientations
        if angles0 is None:
            print("Sampling orientations") 
            # First, sample orientations uniformly on the sphere.

            t0 = time.time()    
            if idx_iter < 64 or jnp.mod(idx_iter, 8) == 4:
                angles_new = []
                for i in jnp.arange(N1):
                    if verbose and N1 > 1:
                        print("batch ", i)

                    #TODO: make this work for multiple batches...how? The batches inside the proposal function I think (needed for the Dirichlet prior)
                    if z0 is None:
                        print("and class assignments")
                        
                        params_orientations_z = {'v':v, 'shifts':shifts[i], 'ctf_params':ctf_params[i], 'imgs' : imgs_iter[i]}

                        az0 = jnp.concatenate([angles[i], z], axis = 1)
                        _, r_samples_angles, samples_az = mcmc(key_angles, proposal_angles_z_mtm, az0, N_samples_angles_global, params_orientations_z, N2, 1, verbose = True)

                        angles_new.append(samples_az[N_samples_angles_global-2,:, :3])
                        z = samples_az[N_samples_angles_global-2, :, [3]].transpose().astype(jnp.int64)
                    else:
                        params_orientations = {'v':v, 'shifts':shifts[i], 'ctf_params':ctf_params[i], 'imgs_iter' : imgs_iter[i], 'z': z }
                        _, r_samples_angles, samples_angles = mcmc(key_angles, proposal_func_orientations_unif, angles[i], N_samples_angles_global, params_orientations, N2, 1, verbose = True)
                        angles_new.append(samples_angles[N_samples_angles_global-2])
                angles = jnp.array(angles_new)

                #print(r_samples_angles)
                if verbose:
                    print(f"  Time global orientations sampling = {time.time()-t0}")
                    print(f"  mean(a_angles) = {jnp.mean(r_samples_angles[~jnp.isnan(r_samples_angles)])} ({jnp.sum(jnp.isnan(r_samples_angles))} nans)", flush=True)
                    print(f"  max(a_angles) = {jnp.max(r_samples_angles)}, min(a_angles) = {jnp.min(r_samples_angles)}")

                    #plot_angles(angles[:500])
                    #plt.show()

            # And now sample local perturbations of the orientations.
            if idx_iter >= 64:
                t0 = time.time()    
                angles_new = []
                for i in jnp.arange(N1):
                    if verbose and N1 > 1:
                        print("batch ", i)
                    params_orientations = {'v':v, 'shifts':shifts[i], 'ctf_params':ctf_params[i], 'imgs_iter' : imgs_iter[i], 'sigma_perturb': sigma_perturb_list, 'z': z}
                    _, r_samples_angles, samples_angles = mcmc(key_angles, proposal_func_orientations_pert, angles[i], N_samples_angles_local, params_orientations, N2, 1, verbose = True)
                    angles_new.append(samples_angles[N_samples_angles_local-2])
                angles = jnp.array(angles_new)


                if verbose:
                    print(f"  Time local orientations sampling = {time.time()-t0}")
                    print(f"  mean(a_angles) = {jnp.mean(r_samples_angles)}", flush=True)

                    #plot_angles(angles[:500])
                    #plt.show()

        # SAMPLING SHIFTS IS NOT CURRENTLY WORKING
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
        
        t0 = time.time()
        v_iter = []
        for k in jnp.arange(K):
            print(f"Sampling volume {k}")

            #if N1 == 1:
            #    params_vol = {'angles':angles[0], 'shifts':shifts[0], 'ctf_params':ctf_params[0], 'imgs_iter':imgs_iter[0], 'z': z}
            #else:
            #    params_vol = {'angles':angles, 'shifts':shifts, 'ctf_params':ctf_params, 'imgs_iter':imgs_iter}

            zkidx = (z[:,0] == k)
            if N1 == 1:
                params_vol = {'angles':angles[0,zkidx], 'shifts':shifts[0,zkidx], 'ctf_params':ctf_params[0,zkidx], 'imgs_iter':imgs_iter[0,zkidx]}
            else:
                params_vol = {'angles':angles[:,zkidx], 'shifts':shiftsl[:,zkidx], 'ctf_params':ctf_params[:,zkidx], 'imgs_iter':imgs_iter[:,zkidx]}

            v_hmc_mean, r_hmc, v_hmc_samples = mcmc(subkey, proposal_func_vol, v[k], N_samples_vol, params_vol, save_samples = -1)
            #v = v_hmc_mean 
            #v = v_hmc_samples[N_samples_vol-2] 
            vk = v_hmc_samples[0] 
            vk = vk*mask3d 
            #v = jnp.array([vi*mask3d for vi in v])

            v_iter.append(vk)

            if verbose:
                print(f"  Time volume {k} sampling = {time.time()-t0}")
                print(f"  mean(a_vol_{k}) = {jnp.mean(r_hmc)}", flush=True)

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
        v = jnp.array(v_iter)
        
        if jnp.mod(idx_iter, freq_marching_step_iters)==0 and verbose:
            print(datetime.datetime.now())
            print("  nx =", nx_iter, flush=True)

            if diagnostics:
                print(f"v.shape={v.shape}")
                #plt.imshow(jnp.abs(jnp.fft.fftshift(v[:,:,0]*mask3d[:,:,0])))
                plt.imshow(jnp.abs(jnp.fft.fftshift(v[0,:,:,0]*mask3d[:,:,0])))
                plt.colorbar()
                plt.show()

                plot_angles(angles[0, :500])
                plt.show()

                nbins = 100
                counts, bins = np.histogram(z, bins=nbins)
                _ = plt.hist(bins[:-1], bins, weights=counts)
                plt.show()

        if jnp.mod(idx_iter, 1)==0 and save_to_file:
            for class_idx in jnp.arange(K):
                with mrcfile.new(f"{out_dir}/rec_iter_{idx_iter}_class{class_idx}.mrc", overwrite=True) as mrc:
                    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[class_idx])))
                    mrc.set_data(vr.astype(np.float32))

            #TODO: should probably print to a star file instead
            file = open(out_dir + '/rec_iter_' + str(idx_iter) + '_angles', 'wb')
            pickle.dump(angles, file)
            file.close()

            file2 = open(out_dir + '/rec_iter_' + str(idx_iter) + '_shifts', 'wb')
            pickle.dump(shifts, file2)
            file2.close()

            file3 = open(out_dir + '/rec_iter_' + str(idx_iter) + '_z', 'wb')
            pickle.dump(z, file3)
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
    #v = v_hmc_mean 
    #v = v*mask3d
    v = jnp.array([vi*mask3d for vi in v])

    if save_to_file:
        for class_idx in jnp.arange(K):
            vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v[class_idx])))
            with mrcfile.new(f"{out_dir}/rec_final_class{class_idx}.mrc", overwrite=True) as mrc:
                mrc.set_data(vr.astype(np.float32))

    #return v, angles, shifts, samples_angles, r_samples_angles,  v_hmc_samples , r_hmc
    return v, angles, shifts, z



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
        learning_rate = 1, 
        batch_size = -1,  
        B = 1, 
        K = 1,
        interp_method = 'tri', 
        verbose = True):
    if verbose:
        print("Initialitsing volume")
    
    N1 = imgs.shape[0]
    N2 = imgs.shape[1]
    nx = jnp.sqrt(imgs.shape[2]).astype(jnp.int64)

    #v0 = jnp.array(np.random.randn(K, nx,nx,nx) + np.random.randn(K, nx,nx,nx)*1j)
    v0 = jnp.array(np.random.randn(nx,nx,nx) + np.random.randn(nx,nx,nx)*1j)
    mask3d = jnp.ones([nx,nx,nx])

    _, grad_loss_volume_sum, grad_loss_volume_sum_z, _, _,_,_, _,_,_ = get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid, mask3d, 0, interp_method)

    key1, key2, key3 = random.split(key, 3)

    angles = generate_uniform_orientations_jax_batch(key, N1, N2)
    shifts = jnp.zeros([N1, N2, 2]) #generate_uniform_shifts(key, N, B)
    #shifts = generate_gaussian_shifts(key, N, B)

    #z = random.randint(subkey, [N1, N2], 0, K) 
    z = random.randint(key3, (N2,1), 0, K) 

    #grad_loss_volume_batched_sum = lambda v, a, s, c, imgs, sig : grad_loss_volume_batched(v, a, s, c, imgs, sig) / a.shape[0]
    #TODO: should move the contents of get_sgd_vol_ops here (it's only one line), as it's not used anywhere else anyway.

    #sgd_grad_func = get_sgd_vol_ops(grad_loss_volume_sum_z, angles[0], shifts[0], ctf_params[0], imgs[0], z, sigma_noise)
    sgd_grad_func = get_sgd_vol_ops(grad_loss_volume_sum, angles[0], shifts[0], ctf_params[0], imgs[0], sigma_noise)
    v = sgd(sgd_grad_func, N2, v0, learning_rate, N_vol_iter, batch_size, None, eps_vol, verbose = verbose)

    if K > 1:
        v = jnp.repeat(v[jnp.newaxis, :, :], K, axis=0) 
        v += jnp.array(np.random.randn(K, nx,nx,nx) + np.random.randn(K, nx,nx,nx)*1j)

    return v, angles, shifts, z


def get_diagnostics_funs_iter(project_func, x_grid, mask, alpha = 0, interp_method = 'tri'):
    slice_func,slice_func_array, slice_func_array_angles = get_slice_funcs(project_func, x_grid, mask, interp_method)
    loss_func, loss_func_batched, loss_func_sum = get_loss_funcs(slice_func, alpha = alpha)
    fid_func, fid_func_batched, fid_func_sum = get_loss_funcs(slice_func, alpha = 0)

    return fid_func_sum, loss_func_sum



def get_jax_ops_iter(project_func, rotate_and_interpolate_func, apply_shifts_and_ctf_func, x_grid, mask, alpha = 0, interp_method = 'tri'):
    slice_func,slice_func_array, slice_func_array_angles = get_slice_funcs(project_func, x_grid, mask, interp_method)
    loss_func, loss_func_batched, loss_func_sum  = get_loss_funcs(slice_func, alpha = alpha)

    grad_loss_volume, grad_loss_volume_sum = get_grad_v_funcs(loss_func, loss_func_sum)

    loss_func_z, loss_func_z_batched, loss_func_z_sum, grad_loss_volume_z, grad_loss_volume_sum_z = get_loss_grad_funcs_classes(loss_func)

    loss_func_angles = get_loss_func_angles(loss_func)
    _, loss_func_batched0, _ = get_loss_funcs(slice_func, alpha = 0)
    loss_proj_func_batched0 = get_loss_proj_funcs(apply_shifts_and_ctf_func, x_grid, alpha = 0)

    @jax.jit
    def rotate_and_interpolate(v, angles):
        return jax.vmap(rotate_and_interpolate_func, in_axes=(None,0,None,None))(v*mask, angles, x_grid, x_grid)


    return slice_func_array_angles, grad_loss_volume_sum, grad_loss_volume_sum_z, loss_func_angles, loss_func_batched0, loss_func_sum, loss_func_z_sum, loss_func_z_batched, loss_proj_func_batched0, rotate_and_interpolate


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
    def proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, z, generate_orientations_func, params_orientations):
        logPi = lambda a : -loss_func_batched0_iter(v, a, shifts, ctf_params, imgs_iter, z, sigma_noise_iter)

        angles1 = generate_orientations_func(key, angles0, **params_orientations)

        logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
            true_fun = lambda _ : logPi(angles0),
            false_fun = lambda _ : logPiX0,
            operand = None)

        logPiX1 = logPi(angles1)
        r = jnp.exp(logPiX1 - logPiX0)

        return angles1, r, logPiX1, logPiX0 

    @jax.jit
    def proposal_func_orientations_uniform(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, z):
        empty_params = {}
        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, z, generate_uniform_orientations_jax, empty_params)

    @jax.jit
    def proposal_func_orientations_perturb(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, z, sigma_perturb):
        key, subkey = random.split(key)
        sig_p = random.permutation(subkey, sigma_perturb)[0]
        orient_params = {'sig' : sig_p}

        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, z, generate_perturbed_orientations, orient_params)

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

def get_class_proposal_func(loss_func_batched, loss_func_sum, sigma_noise, alpha_d, K):

    def logPi(v, angles, shifts, ctf_params, imgs, z, sigma_noise):
        n_k = calc_nk_jit(z)
        term1 = -loss_func_sum(v, angles, shifts, ctf_params, imgs, z, sigma_noise)

        logPZalpha = jnp.sum(gammaln(n_k + alpha_d))            

        return term1 + logPZalpha
   
    calc_nk_k = lambda z, k : jnp.sum(z == k)
    calc_nk = lambda z : jax.vmap(calc_nk_k, in_axes = (None, 0))(z, jnp.arange(K))
    calc_nk_jit = jax.jit(calc_nk)


        
    #TODO: this is how shifts should be done too? even included in this particular function maybe
    # It's not really working anyway. Probably need to do multiple-try Metropolis, see below
    @jax.jit
    def proposal_z_angles_batch(key, az0, logPiX0, v, shifts, ctf_params, imgs):
        """A combination of the proposal_z_batch function and proposal_func_orientation
        function (with uniform proposal), to be used when calling proposal_z_batch_correct
        or on its own.

        az0 and az1 are N x 4, where the first 3 columns correspond to the Euler
        angles and the last collumn is the z class assignment.
        """

        N = az0.shape[0]
        angles0 = az0[:,:3]
        z0 = az0[:,[3]].astype(jnp.int64)

        key1, key2 = random.split(key)

        angles1 = generate_uniform_orientations_jax(key1, angles0)
        z1 = random.randint(key2, z0.shape, 0, K) 
        az1 = jnp.concatenate([angles1, z1], axis = 1)

        logPiX0 = -loss_func_batched(v, angles0, shifts, ctf_params, imgs, z0, sigma_noise)
        logPiX1 = -loss_func_batched(v, angles1, shifts, ctf_params, imgs, z1, sigma_noise)
        r = jnp.exp(logPiX1 - logPiX0)

        return az1, r, logPiX1, logPiX0
        

    @jax.jit
    def proposal_angles_z_mtm_batch(key, az0, logPiX0, v, shifts, ctf_params, imgs):
        """A combination of the proposal_z_batch function and proposal_func_orientation
        function (with uniform proposal), to be used when calling proposal_z_batch_correct
        or on its own.

        az0 and az1 are N x 4, where the first 3 columns correspond to the Euler
        angles and the last collumn is the z class assignment.
        """

        #print(f"  proposal start logPiX0 = {jnp.mean(logPiX0)}")


        N_class_samples = 2 

        N = az0.shape[0]
        angles0 = az0[:,:3]
        z0 = az0[:,3].astype(jnp.int64)
        #print("    ", az0.shape)
        #print("    ", angles0.shape)
        #print("    ", z0.shape)

        key, key1, key2 = random.split(key, 3)

        # The proposed state and the weights 
        wyk, angles1, z1, z1idx = multiple_try_metropolis_states(key1, angles0, N_class_samples, v, shifts, ctf_params, imgs)
           

        # The reference set x_1, ..., x_{k-1} (without the current point x_k =x) 
        # and their corresponding weights. Actually only need the weights
        #wxk, _, _ = multiple_try_metropolis_states(key2, angles0, N_class_samples-1, v, shifts, ctf_params, imgs)
    
        # Calculate the weights corresponding to the proposed state and the current state 
        wy = -loss_func_batched(v, angles1, shifts, ctf_params, imgs, z1, sigma_noise)
        wx = -loss_func_batched(v, angles0, shifts, ctf_params, imgs, z0, sigma_noise)

        # And now the weights of the reference states where we re-use the proposed candidate states (except the proposed one)
        wxk = jax.vmap(lambda wyki, z1idxi, wxi : wyki.at[z1idxi].set(wxi), in_axes = (0,0, 0))(wyk, z1idx, wx)

        # And finally the acceptance probability
        #r = jnp.sum(jnp.exp(wyk), axis=1) / jnp.sum(jnp.exp(wxk), axis=1)
        r = jax.vmap(ratio_sum_exp, in_axes=(0,0))(wyk, wxk)


        logPiX0 = wx
        logPiX1 = wy 
        #print(f"  proposal end logPiX0 = {jnp.mean(logPiX0)}")
        #print("  ", jnp.mean(logPiX1, axis=0))

        az1 = jnp.column_stack((angles1, z1))

        #print(f"  z0 = {z0[:10]}")
        #print(f"  z1 = {z1[:10]}")
        #print(f"  a = {jnp.minimum(1, r)[:10]}")

        #print("    ", az1.shape)
        #print("    ", angles1.shape)
        #print("    ", z1.shape)

        return az1, r, logPiX1, logPiX0
     
    
    def multiple_try_metropolis_states(key, angles0, N_class_samples, v, shifts, ctf_params, imgs):
        """Function which calculates new states and their corresponding weights 
        for multiple-try Metropolis for
        uniform rotations with class assignment sampling."""

        key, key1, key2 = random.split(key, 3)
        N = angles0.shape[0]
        # Propose new angles and 2K new z
        angles1 = generate_uniform_orientations_jax(key1, angles0)
        z1 = random.randint(key2, (N, N_class_samples), 0, K) 

        # Compute the weights w(y_j, x) = pi(y_j) for each new state y_j
        # i.e. compute pi for the new angles for each new class, then just select using the
        # index in the new z. 
        # wk are the weights for the new angles1 and each possible class k = 1,...,K
        Krange = jnp.tile(jnp.arange(K), (N, 1))
        wk = jax.vmap(lambda zk : -loss_func_batched(v, angles1, shifts, ctf_params, imgs, zk, sigma_noise), in_axes = (1))(Krange).transpose()
        # wj are the actual weights of the new proposed states (angles1, z1)
        wj = jax.vmap(lambda wki, z1i : wki[z1i], in_axes = (0,0))(wk, z1)

        # Select y from y_j with probability proportional to wj 
        keys = random.split(key, N)
        z1idx = jax.vmap(lambda keyi, wji : jax.random.categorical(keyi, wji), in_axes=(0,0))(keys, wj)
        z1 = jax.vmap(lambda z1i, z1idxi: z1i[z1idxi], in_axes=(0,0))(z1, z1idx)
        
        return wj, angles1, z1, z1idx

    @jax.jit
    def ratio_sum_exp(a, b):
        """Given two arrays a=[A1, ..., An], b=[B1,..., Bn],
        compute the ratio sum(exp(a1)) / sum(exp(a2)) in a way
        that doesn't lead to nan's."""

        log_ratio = a[0] - b[0] \
            + jnp.log(jnp.sum(jnp.exp(a-a[0]))) \
            - jnp.log(jnp.sum(jnp.exp(b-b[0])))

        return jnp.exp(log_ratio)                



    @jax.jit
    def proposal_z_batch(key, z0, logPiX0, v, angles, shifts, ctf_params, imgs):
        """Batch mode proposal function for z, with no Dirichlet prior 
        for the classes.

        If the posterior depends on the states z_1,...,z_N (e.g. due to 
        the Dirichlet prior), then running MCMC with this proposal function
        is NOT correct. However, this function is used in the
        proposal_z_batch_correct" function, which takes advantage of these
        parallel proposals in a proper MCMC way."""

        N = angles.shape[0]
        z1 = random.randint(key, z0.shape, 0, K) 

        logPiX0 = -loss_func_batched(v, angles, shifts, ctf_params, imgs, z0, sigma_noise)
        logPiX1 = -loss_func_batched(v, angles, shifts, ctf_params, imgs, z1, sigma_noise)
        r = jnp.exp(logPiX1 - logPiX0)

        return z1, r, logPiX1, logPiX0

    @jax.jit
    def proposal_z_batch_correct(key, z0, logPiX0, v, angles, shifts, ctf_params, imgs):
        """If using the Dirichlet posterior, then the proposal_z_batch is not 
        proper MCMC sampling. Here, we run a number of "incorect" MCMC steps to 
        sample all the entries of z in parallel, and then use the latest 
        proposed z as a proposal for a "correct" MCMC proposal. The acceptance rate
        is pretty good in this toy example and the convergence great."""

        logPi_local = lambda z : logPi(v, angles, shifts, ctf_params, imgs, z, sigma_noise)

        N = angles.shape[0]
        N_samples_z_local = 20
        keys = random.split(key, 2*N_samples_z_local)
        params_z = {"v" : v, "angles" : angles, "shifts" : shifts, "ctf_params" : ctf_params, "imgs" : imgs}

        logPiZ0 = logPi_local(z0)

        #logPiX0 = logPiZ0
        #for i in jnp.arange(N_samples_z_local):
        #    z1, r, logPiX1, logPiX0 = proposal_z_batch(keys[2*i], z0, logPiX0, **params_z)
        #    a = jnp.minimum(1, r)
        #
        #    unif_var = random.uniform(keys[2*i+1], (N,))
        #    z1, logPiX1 = accept_reject_vmap(unif_var, a, z0, z1, logPiX0, logPiX1)
        #
        #    z0 = z1
        #    logPiX0 = logPiX1

        # The fori_loop version of the above for loop to save compilation time.
        def body_fun(i, z0logPiX0):
            z0, logPiX0 = z0logPiX0

            z1, r, logPiX1, logPiX0 = proposal_z_batch(keys[2*i], z0, logPiX0, **params_z)
            a = jnp.minimum(1, r)

            unif_var = random.uniform(keys[2*i+1], (N,))
            z1, logPiX1 = accept_reject_vmap(unif_var, a, z0, z1, logPiX0, logPiX1)

            return z1, logPiX1

        z1, _ = jax.lax.fori_loop(0, N_samples_z_local, body_fun, (z0, logPiZ0*jnp.ones((N,))))

        logPiZ1 = logPi_local(z1)
        r = jnp.exp(logPiZ1 - logPiZ0)       

        return z1, r, logPiZ1, logPiZ0

    return proposal_z_batch_correct, proposal_z_angles_batch, proposal_angles_z_mtm_batch





# I think everythink below is broken and/or not used anywhere, make sure that's right and then remove it

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
                                               

