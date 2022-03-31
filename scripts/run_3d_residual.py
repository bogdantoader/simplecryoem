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
from src.fsc import *
from src.algorithm import *
from src.ab_initio import ab_initio, ab_initio_mcmc
from src.residual import get_volume_residual
import jax

import mrcfile


def parse_args(parser):
    parser.add_argument("data_dir", help="Location of the star and mrcs files.")
    parser.add_argument("star_file", help="Name of the star file, relative to data_dir.")
    parser.add_argument("out_dir", help="Directory to save the output.")
    parser.add_argument("out_file", help="File name for the residual.")
    parser.add_argument("-N", "--N_imgs", type=int, help="Only keep N particle images.")
    parser.add_argument("-nx", "--nx_crop", type=int, help="Crop the images to nx x nx pixels.")
    parser.add_argument("-Npxn", "--N_px_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-Nin", "--N_imgs_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-r", "--radius", type=float, help="Mask radius.", default=np.inf)
    parser.add_argument("-Nb", "--N_batches", type=int, help="Number of batches to process the coordinates on the GPU.", default=600)

    args = parser.parse_args()
    return args


def main(args):
    
    params0, imgs0 = load_data(args.data_dir, args.star_file, load_imgs = True, fourier = False)
    ctf_params0 = params0["ctf_params"]
    pixel_size0 = params0["pixel_size"]
    angles0 = params0["angles"]
    shifts0 = params0["shifts"]

    print(f'imgs0.shape = {imgs0.shape}')
    print(f'pixel_size0 = {pixel_size0.shape}')
    print(f'angles0.shape = {angles0.shape}')
    print(f'shifts0.shape = {shifts0.shape}')
    print(f'ctf_params0.shape = {ctf_params0.shape}')


    # Only keep the first N images
    if args.N_imgs:
        assert (args.N_imgs <= imgs0.shape[0]), 'N cannot be smaller than the number of particles.'
        N = args.N_imgs
        idxrand = np.random.permutation(imgs0.shape[0])[:N]
    else:
        N = imgs0.shape[0]
        idxrand = jnp.arange(N)
        
    print(f'N = {N}')

    imgs0 = imgs0[idxrand]
    pixel_size = pixel_size0[idxrand]
    angles = angles0[idxrand]
    shifts = shifts0[idxrand]
    ctf_params = ctf_params0[idxrand]
    
    file2 = open(args.out_dir + '/idxrand','wb')
    pickle.dump(idxrand, file2)
    file2.close()

    # Take the FFT of the images
    print("Taking FFT of the images...", end="", flush=True)
    t0 = time.time()
    imgs_f = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs0])
    print(f"done. Time: {time.time()-t0} seconds.") 

    # Assume the pixel size is the same for all images
    nx = imgs_f.shape[-1]
    px = pixel_size[0]
    N = imgs_f.shape[0]

    x_grid = create_grid(nx, px)
    y_grid = x_grid
    z_grid = x_grid
    print(f"x_grid = {x_grid}")

    # Crop the images
    if args.nx_crop:
        nx = args.nx_crop
        imgs_f, x_grid = crop_fourier_images(imgs_f, x_grid, nx)
        y_grid = x_grid
        z_grid = x_grid
        print(f"new x_grid = {x_grid}")

    # Vectorise images
    imgs_f = imgs_f.reshape(N, -1)
    print(f"imgs_f.shape = {imgs_f.shape}")

    # Estimate the noise   
    if args.N_px_noise:
        N_px_noise = args.N_px_noise
    else:
        N_px_noise = nx 

    if args.N_imgs_noise:
        N_imgs_noise = args.N_imgs_noise
    else:
        N_imgs_noise = N
    
    print(f"Estimating the noise using the {N_px_noise} x {N_px_noise} corners of the first {N_imgs_noise} images...", end="", flush=True)

    t0 = time.time()
    sigma_noise_estimated = estimate_noise_imgs(imgs0[:N_imgs_noise], nx_empty = N_px_noise, nx_final = nx).reshape([nx,nx])
    sigma_noise_avg = average_radially(sigma_noise_estimated, x_grid)
    sigma_noise = sigma_noise_avg.reshape(-1)
    print(f"done. Time: {time.time()-t0} seconds.", flush=True) 
   
    # Delete the initial large images.
    del(imgs0)

    # Mask for the result
    mask = create_3d_mask(x_grid, (0,0,0), args.radius)

    # The heavy lifting
    v_resid = get_volume_residual(imgs_f, angles, sigma_noise, nx, args.N_batches)

    # And print to file
    v_resid_print = jnp.fft.fftshift(v_resid*mask)
    with mrcfile.new(args.out_dir + args.out_file, overwrite=True) as mrc:
            mrc.set_data(v_resid_print.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
