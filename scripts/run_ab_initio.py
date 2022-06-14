import argparse
import time
import pickle

import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from src.utils import *
from src.projection import *
from src.interpolate import *
from src.jaxops import *
from src.noise import noise 
from src.fsc import *
from src.algorithm import *
from src.ab_initio import ab_initio_mcmc
from src.residual import get_volume_residual
from src.preprocess import preprocess

import jax
import mrcfile



def parse_args(parser):
    parser.add_argument("data_dir", help="Location of the star and mrcs files.")
    parser.add_argument("star_file", help="Name of the star file, relative to data_dir.")
    parser.add_argument("out_dir", help="Directory to save the output at each iteration (vol, angles, shifts).")
    parser.add_argument("-N", "--N_imgs", type=int, help="Only keep N particle images.")
    parser.add_argument("-nx", "--nx_crop", type=int, help="Crop the images to nx x nx pixels.")
    parser.add_argument("-Npxn", "--N_px_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-Nin", "--N_imgs_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-Nb", "--N_batch", type=int, help="Number of batches for calculating the loss and gradient.", default=1)
    parser.add_argument("-Ni", "--N_iter", type=int, help="Number of iterations.", default=10000000)
    parser.add_argument("-r0", "--radius0", type=float, help="Starting radius for frequency marching.")
    parser.add_argument("-a", "--alpha", type=float, help="Regularisation parameter.", default=1e-9)
    parser.add_argument("-sgdbs", "--sgd_batch_size", type=int, help="Batch size for initialisation SGD run.", default=300)
    parser.add_argument("-sgdlr", "--sgd_learning_rate", type=float, help="Learning rate for initialisation SGD run.", default=1e6)
    parser.add_argument("-ei", "--eps_init", type=float, help="Stopping criterion epsilon for initialisation SGD run.", default=2e-7)
    parser.add_argument("-Nsv", "--N_samples_vol", type=int, help="Number of MCMC samples of the volume at each iteration.", default=101)
    parser.add_argument("-Nsag", "--N_samples_angles_global", type=int, help="Number of global MCMC samples of the orientations at each iteration.", default=1001)
    parser.add_argument("-Nsal", "--N_samples_angles_local", type=int, help="Number of local MCMC samples of the orientations at each iteration.", default=201)
    parser.add_argument("-Nss", "--N_samples_shifts", type=int, help="Number of MCMC samples of the shifts at each iteration.", default=101)
    parser.add_argument("-L", "--L_hmc", type=int, help="Number of step sizes for HMC integration.", default=5)
    parser.add_argument("-FM", "--freq_marching_steps", type=int, help="Number of iterations before increasing the Fourier radius.", default=8)
    parser.add_argument("-Mbs", "--minibatch_factor", type=int, help="Size of minibatch to work with at each iteration is nx_iter * minibatch_factor", default = 50)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-nf", "--noise_free", action="store_true")

    args = parser.parse_args()
    return args

def main(args):
    params0, imgs0 = load_data(args.data_dir, args.star_file, load_imgs = True, fourier = False)


    print(f"min(shifts) = {jnp.min(shifts0)}")
    print(f"max(shifts) = {jnp.max(shifts0)}")

    # Only keep the first N images
    if args.N_imgs:
        assert (args.N_imgs <= imgs0.shape[0]), 'N cannot be smaller than the number of particles.'
        N = args.N_imgs
        shuffle = True
    else:
        N = imgs0.shape[0]
        shuffle = False

    # Estimate the noise   
    if args.noise_free:
        N_px_noise = 0
    else:    
        if args.N_px_noise:
            N_px_noise = args.N_px_noise
        else:
            N_px_noise = nx 

        if args.N_imgs_noise:
            N_imgs_noise = args.N_imgs_noise
        else:
            N_imgs_noise = N
        
    processed_data = preprocess(imgs0, params0, args.out_dir, nx_crop = args.nx_crop, N = N, shuffle = shuffle, N_px_noise = N_px_noise, N_imgs_noise = N_imgs_noise)

    imgs_f = processed_data["imgs_f"]
    pixel_size = processed_data["pixel_size"]
    angles = processed_data["angles"]
    shifts = processed_data["shifts"]
    ctf_params = processed_data["ctf_params"]
    idxrand = processed_data["idxrand"]
    nx = processed_data["nx"]
    x_grid = processed_data["x_grid"]
    mask = processed_data["mask"]
    sigma_noise = processed_data["sigma_noise"]
    N = imgs_f.shape[0]

    nx0 = imgs.shape[2]
    # Delete the initial large images.
    del(imgs0)

    # Split in batches, note that imgs_batch stays on the CPU (i.e. np not jnp)
    imgs_batch = np.array(np.array_split(imgs_f, args.N_batch))
    angles_batch = jnp.array(np.array_split(angles, args.N_batch))
    shifts_batch = jnp.array(np.array_split(shifts, args.N_batch))
    ctf_params_batch = jnp.array(np.array_split(ctf_params, args.N_batch))

    if args.radius0:
        radius0 = args.radius0
    else:
        radius0 = 2.1 * pixel_size[0]*(nx0/nx)

    # List of step sizes for HMC. TODO: tune this automatically.
    dt_list_hmc = jnp.array([0.1, 0.5, 1, 5, 10])

    pixel_size_crop = pixel_size[0] * nx0/nx
    B = pixel_size[0] * nx0/8
    B_list = jnp.array([B/4, B/8, B/16, B/32])
    sigma_perturb_list = jnp.array([1, 0.1, 0.01, 0.001])

    vol0 = None
    angles0 = None
    shifts0 = None 

    print(f"B = {B}")
    print(f"B_list = {B_list}")
    print(f"sigma_perturb_list = {sigma_perturb_list}")
    
    key = random.PRNGKey(int(jnp.floor(np.random.rand()*1000)))
    v_rec, angles_rec, shifts_rec = ab_initio_mcmc(key, 
                                   project, 
                                   rotate_and_interpolate,
                                   apply_shifts_and_ctf,
                                   imgs_batch, 
                                   sigma_noise, 
                                   ctf_params_batch, 
                                   x_grid, 
                                   vol0,
                                   angles0,
                                   shifts0,
                                   args.N_iter, 
                                   args.sgd_learning_rate, 
                                   args.sgd_batch_size, 
                                   args.N_samples_vol,
                                   args.N_samples_angles_global, 
                                   args.N_samples_angles_local, 
                                   args.N_samples_shifts,
                                   dt_list_hmc,
                                   sigma_perturb_list, 
                                   args.L_hmc, 
                                   radius0, 
                                   None, 
                                   args.alpha, 
                                   args.eps_init,
                                   B,
                                   B_list,
                                   args.minibatch_factor,
                                   args.freq_marching_steps,
                                   'tri', 
                                   True, 
                                   args.verbose, 
                                   False,
                                   True,
                                   args.out_dir)

    v_rec_l, x_grid_l = rescale_larger_grid(v_rec, x_grid, nx0)
    v_rec_rl = jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(v_rec_l)))

    with mrcfile.new(args.out_dir + '/rec_final_nx0.mrc', overwrite=True) as mrc:
        mrc.set_data(v_rec_rl.astype(np.float32))

    #TODO: save the parameters to the folder too


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
