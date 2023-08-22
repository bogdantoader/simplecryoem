import argparse
import time
import pickle
import mrcfile
import numpy as np
import jax.numpy as jnp
from simplecryoem.utils import *
from simplecryoem.projection import *
from simplecryoem.interpolate import *
from simplecryoem.jaxops import *
from simplecryoem.fsc import *
from simplecryoem.algorithm import *
from simplecryoem.residual import get_volume_residual


def parse_args(parser):
    parser.add_argument("data_dir", help="Location of the star and mrcs files.")
    parser.add_argument("star_file_proj", help="Name of the star file for the projections, relative to data_dir.")
    parser.add_argument("star_file_img", help="Name of the star file for the images, relative to data_dir.")
    parser.add_argument("out_dir", help="Directory to save the output.")
    parser.add_argument("out_file", help="File name for the residual.")
    parser.add_argument("-N", "--N_imgs", type=int, help="Only keep N particle images.")
    parser.add_argument("-nx", "--nx_crop", type=int, help="Crop the images to nx x nx pixels.")
    parser.add_argument("-Npxn", "--N_px_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-Nin", "--N_imgs_noise", type=int, help="Length of the corner side to use for noise estimation.")
    parser.add_argument("-r", "--radius", type=float, help="Mask radius.", default=np.inf)
    parser.add_argument("-Nb", "--N_batches", type=int, help="Number of batches to process the coordinates on the GPU.", default=600)
    parser.add_argument("-sp", "--spatial", help="If this flag is used, don't take the Fourier transofrm of the images.", action="store_true")
    parser.add_argument("-sig", "--sigma_noise", help="If this flag is used, estimate sigma_noise and store in a separate volume.", action="store_true")

    args = parser.parse_args()
    return args


def main(args):
   
    # Assume that params are the same for both sets.
    _, imgs = load_data(args.data_dir, args.star_file_proj, load_imgs = True, fourier = False)
    params0, imgs0 = load_data(args.data_dir, args.star_file_img, load_imgs = True, fourier = False)
    ctf_params0 = params0["ctf_params"]
    pixel_size0 = params0["pixel_size"]
    angles0 = params0["angles"]
    shifts0 = params0["shifts"]

    print(f'imgs.shape = {imgs.shape}')
    print(f'imgs0.shape = {imgs0.shape}')
    print(f'pixel_size0 = {pixel_size0.shape}')
    print(f'angles0.shape = {angles0.shape}')
    print(f'shifts0.shape = {shifts0.shape}')
    print(f'ctf_params0.shape = {ctf_params0.shape}')

    # Save the noisy images in a separate variable for noise estimation
    imgs_noisy = imgs0

    # Only keep the first N images
    if args.N_imgs:
        assert (args.N_imgs <= imgs0.shape[0]), 'N cannot be smaller than the number of particles.'
        N = args.N_imgs
        idxrand = np.random.permutation(imgs0.shape[0])[:N]
    else:
        N = imgs0.shape[0]
        idxrand = jnp.arange(N)
        
    print(f'N = {N}')



    imgs = imgs[idxrand]
    imgs0 = imgs0[idxrand]  
    pixel_size = pixel_size0[idxrand]
    angles = angles0[idxrand]
    shifts = shifts0[idxrand]
    ctf_params = ctf_params0[idxrand]
    
    file2 = open(args.out_dir + '/idxrand','wb')
    pickle.dump(idxrand, file2)
    file2.close()

    if args.spatial:
        print("--spatial flag used, NOT taking the Fourier transform of the images.")
    else:
        # Take the FFT of the images
        print("Taking FFT of the images...", end="", flush=True)
        t0 = time.time()
        imgs = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs])
        imgs0 = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs0])
        print(f"done. Time: {time.time()-t0} seconds.") 

    # Assume the pixel size is the same for all images
    nx = imgs.shape[-1]
    px = pixel_size[0]
    N = imgs.shape[0]

    x_grid = create_grid(nx, px)
    y_grid = x_grid
    z_grid = x_grid
    print(f"x_grid = {x_grid}")

    # Crop the images
    if args.nx_crop:
        nx = args.nx_crop
        imgs, x_grid = crop_fourier_images(imgs, x_grid, nx)
        imgs0, x_grid = crop_fourier_images(imgs0, x_grid, nx)
        y_grid = x_grid
        z_grid = x_grid
        print(f"new x_grid = {x_grid}")

    # Vectorise images
    imgs = imgs.reshape(N, -1)
    imgs0 = imgs0.reshape(N, -1)
    print(f"imgs.shape = {imgs.shape}")

    if args.sigma_noise:
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
        sigma_noise_estimated = estimate_noise_imgs(imgs_noisy[:N_imgs_noise], nx_empty = N_px_noise, nx_final = nx).reshape([nx,nx])
        sigma_noise_avg = average_radially(sigma_noise_estimated, x_grid)
        sigma_noise = sigma_noise_avg.reshape(-1)
        print(f"done. Time: {time.time()-t0} seconds.", flush=True) 
    else: 
        sigma_noise = np.ones(imgs0.shape[1])

    # No need to keep the noisy images in the memory now.
    del(imgs_noisy)

    # Mask for the result
    mask = create_3d_mask(x_grid, (0,0,0), args.radius)

    # The (not that) heavy lifting
    vol, _, _ = get_volume_residual(imgs-imgs0, angles, sigma_noise, x_grid, args.radius, args.N_batches)
    vol0, vol_sigma, vol_counts = get_volume_residual(imgs0, angles, sigma_noise, x_grid, args.radius, args.N_batches)

    # And print to file
    with mrcfile.new(f"{args.out_dir}/{args.out_file}_{nx}_resid.mrc", overwrite=True) as mrc:
        mrc.set_data(jnp.fft.fftshift(vol).astype(np.float32))
   
    with mrcfile.new(f"{args.out_dir}/{args.out_file}_{nx}_imgs.mrc", overwrite=True) as mrc:
        mrc.set_data(jnp.fft.fftshift(vol0).astype(np.float32))

    with mrcfile.new(f"{args.out_dir}/{args.out_file}_{nx}_counts.mrc", overwrite=True) as mrc:
        mrc.set_data(jnp.fft.fftshift(vol_counts).astype(np.float32))

    if args.sigma_noise:
        with mrcfile.new(f"{args.out_dir}/{args.out_file}_{nx}_sigma.mrc", overwrite=True) as mrc:
            mrc.set_data(jnp.fft.fftshift(vol_sigma).astype(np.float32))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_args(parser)
    main(args)
