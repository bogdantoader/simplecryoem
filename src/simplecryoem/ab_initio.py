import time
import datetime
import pickle
import jax.numpy as jnp
from jax import random
import numpy as np
from matplotlib import pyplot as plt
import mrcfile

from simplecryoem.optimization import sgd, get_sgd_vol_ops
from simplecryoem.sampling import mcmc_sampling, CryoProposals
from simplecryoem.forwardmoel import Slice
from simplecryoem.loss import Loss, GradV
from simplecryoem.utils import (
    create_3d_mask,
    crop_fourier_volume,
    rescale_larger_grid,
    crop_fourier_images,
    crop_fourier_images_batch,
    generate_uniform_orientations_jax_batch,
    plot_angles,
)


def ab_initio_mcmc(
    key,
    project_func,
    rotate_and_interpolate_func,
    apply_shifts_and_ctf_func,
    imgs,
    sigma_noise,
    ctf_params,
    x_grid,
    vol0=None,
    angles0=None,
    shifts0=None,
    N_iter=100,
    learning_rate=1,
    sgd_batch_size=-1,
    N_samples_vol=100,
    N_samples_angles_global=1000,
    N_samples_angles_local=100,
    N_samples_shifts_global=100,
    N_samples_shifts_local=1000,
    dt_list_hmc=[0.5],
    sigma_perturb_list=jnp.array([1, 0.1, 0.01, 0.001]),
    L_hmc=10,
    radius0=0.1,
    dr=None,
    alpha=0,
    eps_vol=1e-16,
    B=1,
    B_list=[1],
    minibatch_factor=None,
    freq_marching_step_iters=8,
    interp_method="tri",
    opt_vol_first=True,
    verbose=True,
    diagnostics=False,
    save_to_file=True,
    out_dir="./",
):
    """Ab initio reconstruction using MCMC.

    Parameters:
    ----------
    imgs : N1 x N2 x nx*nx array
        The 2d images, vectorised and batched.

    x_grid : [dx, nx]
        The Fourier grid of the images.

    alpha : regularisation parameter

    Returns:

    """

    assert imgs.ndim == 3

    N_batch_shape = jnp.array(imgs.shape[:2])
    N1 = N_batch_shape[0]
    N2 = N_batch_shape[1]
    nx = jnp.sqrt(imgs.shape[2]).astype(jnp.int64)

    # Determine the frequency marching step size, if not given
    if dr is None:
        x_freq = jnp.fft.fftfreq(int(x_grid[1]), 1 / (x_grid[0] * x_grid[1]))
        X, Y, Z = jnp.meshgrid(x_freq, x_freq, x_freq)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        dr = r[1, 1, 1]
    if verbose:
        max_radius = x_grid[0] * x_grid[1] / 2
        n_steps = (jnp.floor((max_radius - radius0) / dr) + 1).astype(jnp.int64)

        print(f"Fourier radius: {max_radius}")
        print(f"Minibatch factor: {minibatch_factor}")
        print(f"Starting radius: {radius0}")
        print(f"Frequency marching step size: {dr}")
        print(f"Number of frequency marching steps: {n_steps}")
        print(f"Number of iterations: {n_steps * freq_marching_step_iters}")
        print(f"B = {B}")
        print(f"B_list = {B_list}")
        print("------------------------------------\n", flush=True)

    if sgd_batch_size == -1:
        sgd_batch_size = N2

    key, subkey = random.split(key)
    if vol0 is None and opt_vol_first:
        N_vol_iter = 3000

        slice_full = Slice(x_grid)
        loss_full = Loss(slice_full, alpha=alpha)
        gradv_full = GradV(loss_full)
        v, angles, shifts = initialize_ab_initio_vol(
            subkey,
            imgs,
            ctf_params,
            gradv_full,
            N_vol_iter,
            eps_vol,
            sigma_noise,
            learning_rate,
            sgd_batch_size,
            shifts0,
            verbose,
        )

        if diagnostics:
            plt.imshow(jnp.abs(jnp.fft.fftshift(v[:, :, 0])))
            plt.colorbar()
            plt.show()
    elif vol0 is None:
        v = jnp.array(np.random.randn(nx, nx, nx) + np.random.randn(nx, nx, nx) * 1j)
    else:
        v = vol0

    # TODO: should have separate options to indicate that we don't want to estimate
    # angles/shifts or that we want to estimate them but start from shifts0, angles0.
    # Same for vol0
    if shifts0 is not None:
        shifts = shifts0
    if angles0 is not None:
        angles = angles0

    imgs = imgs.reshape([N1, N2, nx, nx])
    radius = radius0

    # Reshaping sigma_noise this way so that we can apply crop_fourier_images
    # at each iteration.
    sigma_noise = sigma_noise.reshape([1, nx, nx])

    nx_iter = 0
    recompile = True
    for idx_iter in range(N_iter):
        if (
            nx_iter == nx
            and jnp.mod(idx_iter, freq_marching_step_iters - 1) == 0
            and N_samples_vol < 100
        ):
            N_samples_vol = 500

        if verbose:
            print(f"Iter {idx_iter}")

        if recompile:
            # The nx of the volume at the current iteration
            mask3d = create_3d_mask(x_grid, (0, 0, 0), radius)
            nx_iter = jnp.sum(mask3d[0, 0, :]).astype(jnp.int64)

            # Ensure that we work with even images so that all the masking stuff works
            if jnp.mod(nx_iter, 2) == 1:
                nx_iter += 1

            # At the first iteration, we reduce the size (from v0) while
            # afterwards, we increase it (frequency marching).
            if idx_iter == 0:
                v, _ = crop_fourier_volume(v, x_grid, nx_iter)
            else:
                v, _ = rescale_larger_grid(v, x_grid_iter, nx_iter)

            # Crop the images to the right size
            imgs_iter_full, x_grid_iter = crop_fourier_images_batch(
                imgs, x_grid, nx_iter
            )
            imgs_iter_full = imgs_iter_full.reshape([N1, N2, nx_iter * nx_iter])
            sigma_noise_iter, _ = crop_fourier_images(sigma_noise, x_grid, nx_iter)
            sigma_noise_iter = sigma_noise_iter.reshape(-1)
            mask3d = create_3d_mask(x_grid_iter, (0, 0, 0), radius)
            mask2d = mask3d[0].reshape(1, -1)
            imgs_iter_full = imgs_iter_full * mask2d
            M_iter = (
                1 / jnp.max(sigma_noise) ** 2 * jnp.ones([nx_iter, nx_iter, nx_iter])
            )

            # Get the operators for the dimensions at this iteration.
            slice_iter = Slice(
                x_grid_iter,
                mask3d,
                project_func,
                rotate_and_interpolate_func,
                apply_shifts_and_ctf_func,
                interp_method,
            )
            loss_iter = Loss(slice_iter, alpha=alpha)
            gradv_iter = GradV(loss_iter)

            proposals = CryoProposals(
                sigma_noise_iter,
                B,
                B_list,
                dt_list_hmc,
                L_hmc,
                M_iter,
                slice_iter,
                loss_iter,
                gradv_iter,
            )

        if minibatch_factor is not None and N1 == 1:
            minibatch = True
            minibatch_size = nx_iter * minibatch_factor
            key, subkey = random.split(key)
            idx_img = random.permutation(subkey, N2)[:minibatch_size]

            print(f"Minibatch size = {minibatch_size}")
        else:
            idx_iter = np.arange(N2)

        imgs_iter = imgs_iter_full[:, idx_img]
        angles_iter = angles[:, idx_img]
        shifts_iter = shifts[:, idx_img]
        ctf_params_iter = ctf_params[:, idx_img]

        key, key_volume, key_angles_unif, key_angles_pert, key_shifts = random.split(
            key, 5
        )

        # Sample the orientations
        if angles0 is None:
            # First, sample orientations and shifts uniformly and at the same time using
            # multiple-try Monte Carlo

            t0 = time.time()
            if idx_iter < 8 * freq_marching_step_iters:  # or jnp.mod(idx_iter, 8) == 4:
                print("Sampling global orientations and shifts")

                angles_new = []
                shifts_new = []
                for i in jnp.arange(N1):
                    if verbose and N1 > 1:
                        print("batch ", i)

                    if shifts0 is None:
                        params_mtm = {
                            "v": v,
                            "ctf_params": ctf_params_iter[i],
                            "imgs": imgs_iter[i],
                            "N_samples_shifts": N_samples_shifts_global,
                        }
                        as0 = jnp.concatenate([angles_iter[i], shifts_iter[i]], axis=1)
                        _, r_samples_as, samples_as = mcmc_sampling(
                            key_angles_unif,
                            proposals.proposal_mtm_orientations_shifts,
                            as0,
                            N_samples_angles_global,
                            params_mtm,
                            imgs_iter.shape[1],
                            1,
                            verbose=True,
                            iter_display=50,
                        )
                        as1 = samples_as[N_samples_angles_global - 2]

                        angles_new.append(as1[:, :3])
                        shifts_new.append(as1[:, 3:])
                    else:
                        params_orientations = {
                            "v": v,
                            "ctf_params": ctf_params_iter[i],
                            "imgs": imgs_iter[i],
                            "shifts": shifts_iter[i],
                        }
                        _, r_samples_as, samples_angles = mcmc_sampling(
                            key_angles_unif,
                            proposals.proposal_orientations_uniform,
                            angles_iter[i],
                            N_samples_angles_global,
                            params_orientations,
                            imgs_iter.shape[1],
                            1,
                            verbose=True,
                            iter_display=50,
                        )

                        angles_new.append(samples_angles[N_samples_angles_global - 2])

                angles_iter = jnp.array(angles_new)
                if shifts0 is None:
                    shifts_iter = jnp.array(shifts_new)

                if verbose:
                    print(
                        "  Time global orientations and shifts sampling =",
                        time.time() - t0,
                    )
                    print(
                        "  mean(a_angles_shifts) =", jnp.mean(r_samples_as), flush=True
                    )

            # And now sample local perturbations of the orientations.
            print("Sampling local orientations")

            t0 = time.time()
            angles_new = []
            for i in jnp.arange(N1):
                if verbose and N1 > 1:
                    print(f"batch {i}")
                params_orientations = {
                    "v": v,
                    "shifts": shifts_iter[i],
                    "ctf_params": ctf_params_iter[i],
                    "imgs": imgs_iter[i],
                    "sigma_perturb": sigma_perturb_list,
                }
                _, r_samples_angles, samples_angles = mcmc_sampling(
                    key_angles_pert,
                    proposals.proposal_orientations_perturb,
                    angles_iter[i],
                    N_samples_angles_local,
                    params_orientations,
                    imgs_iter.shape[1],
                    1,
                    verbose=True,
                    iter_display=10,
                )
                angles_new.append(samples_angles[N_samples_angles_local - 2])
            angles_iter = jnp.array(angles_new)

            if verbose:
                print("  Time local orientations sampling =", time.time() - t0)
                print("  mean(a_angles) =", jnp.mean(r_samples_angles), flush=True)

        # Sample the shifts locally
        if shifts0 is None:
            print("Sampling local shifts")
            proj = jnp.array(
                [
                    slice_iter.rotate_and_interpolate_vmap(v, angles_iter[i])
                    for i in jnp.arange(angles_iter.shape[0])
                ]
            )

            t0 = time.time()
            shifts_new = []
            for i in jnp.arange(N1):
                if verbose and N1 > 1:
                    print(f"batch {i}")

                params_shifts = {
                    "v": v,
                    "proj": proj[i],
                    "ctf_params": ctf_params_iter[i],
                    "imgs": imgs_iter[i],
                }
                _, r_samples_shifts, samples_shifts = mcmc_sampling(
                    key_shifts,
                    proposals.proposal_shifts_local,
                    shifts_iter[i],
                    N_samples_shifts_local,
                    params_shifts,
                    imgs_iter.shape[1],
                    1,
                    verbose=True,
                    iter_display=10,
                )
                shifts_new.append(samples_shifts[N_samples_shifts - 2])
            shifts_iter = jnp.array(shifts_new)

            if verbose:
                print("  Time shifts sampling =", time.time() - t0)
                print("  mean(a_shifts) =", jnp.mean(r_samples_shifts), flush=True)

        # Sample the volume
        print("Sampling the volume")

        if N1 == 1:
            params_vol = {
                "angles": angles_iter[0],
                "shifts": shifts_iter[0],
                "ctf_params": ctf_params_iter[0],
                "imgs": imgs_iter[0],
            }
            proposal_vol = proposals.proposal_vol
        else:
            params_vol = {
                "angles": angles_iter,
                "shifts": shifts_iter,
                "ctf_params": ctf_params_iter,
                "imgs": imgs_iter,
            }
            proposal_vol = proposals.proposal_vol_batch

        t0 = time.time()
        v_hmc_mean, r_hmc, v_hmc_samples = mcmc_sampling(
            key_volume,
            proposal_vol,
            v,
            N_samples_vol,
            params_vol,
            save_samples=-1,
            iter_display=10,
        )
        v = v_hmc_samples[0]
        v = jnp.array(v * mask3d)

        # Update the full arrays of angles, shifts etc with
        # the values computed at the current iteration.
        if minibatch:
            angles = angles.at[:, idx_img].set(angles_iter)
            shifts = shifts.at[:, idx_img].set(shifts_iter)

        if verbose:
            print("  Time volume sampling =", time.time() - t0)
            print("  mean(a_vol) =", jnp.mean(r_hmc), flush=True)

        if jnp.mod(idx_iter, freq_marching_step_iters) == 0 and verbose:
            print(datetime.datetime.now())
            print("  nx =", nx_iter, flush=True)

            if diagnostics:
                plt.imshow(jnp.abs(jnp.fft.fftshift(v[:, :, 0] * mask3d[:, :, 0])))
                plt.colorbar()
                plt.show()

                plot_angles(angles_iter[0, :500])
                plt.show()

        if jnp.mod(idx_iter, 1) == 0 and save_to_file:
            with mrcfile.new(
                out_dir + "/rec_iter_" + str(idx_iter) + ".mrc", overwrite=True
            ) as mrc:
                vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
                mrc.set_data(vr.astype(np.float32))

            file = open(out_dir + "/rec_iter_" + str(idx_iter) + "_angles", "wb")
            pickle.dump(angles, file)
            file.close()

            file3 = open(out_dir + "/rec_iter_" + str(idx_iter) + "_shifts", "wb")
            pickle.dump(shifts, file3)
            file3.close()

        # Increase radius
        if jnp.mod(idx_iter, freq_marching_step_iters) == 0:
            radius += dr
            recompile = True

            if nx_iter == nx:
                break
        else:
            recompile = False

    # At the end, take the mean
    v = v_hmc_mean
    v = v * mask3d

    vr = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v)))
    if save_to_file:
        with mrcfile.new(out_dir + "/rec_final.mrc", overwrite=True) as mrc:
            mrc.set_data(vr.astype(np.float32))

    return v, angles, shifts


def initialize_ab_initio_vol(
    key,
    imgs,
    ctf_params,
    gradv_obj,
    N_vol_iter,
    eps_vol,
    sigma_noise=1,
    learning_rate=1,
    batch_size=-1,
    shifts=None,
    verbose=True,
):
    if verbose:
        print("Initialitsing volume")

    key1, key2 = random.split(key)

    N1 = imgs.shape[0]
    N2 = imgs.shape[1]
    nx = jnp.sqrt(imgs.shape[2]).astype(jnp.int64)

    v0 = jnp.array(np.random.randn(nx, nx, nx) + np.random.randn(nx, nx, nx) * 1j)
    angles = generate_uniform_orientations_jax_batch(key, N1, N2)
    if shifts is None:
        shifts = jnp.zeros([N1, N2, 2])

    sgd_grad_func = get_sgd_vol_ops(
        gradv_obj, angles[0], shifts[0], ctf_params[0], imgs[0], sigma_noise
    )
    v = sgd(
        sgd_grad_func,
        N2,
        v0,
        learning_rate,
        N_vol_iter,
        batch_size,
        None,
        eps_vol,
        verbose=verbose,
    )

    return v, angles, shifts
