import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import time
import pickle

from src.utils import *
from src.noise import estimate_noise_radial


def preprocess(imgs0, params0, out_dir, nx_crop = None, N = None, shuffle = False, N_px_noise = 0, N_imgs_noise = None):
    """Basic preprocessing of particle images to extract relevant information 
    for reconstruction.

    Parameters:
    -----------
    imgs0 : [N x nx x nx] array
        Stack of particle images as returned by emfiles.load_data

    params0 : dict
        Dictionary with keys 'ctf_params', 'pixel_size', 'angles', 'shifts',
        each containing an array.

    out_dir : str
        Name of directory to write output to.

    nx_crop: int
        Downsample images to dimensions nx_crop x nx_crop.
        This is done by cropping the images in the Fourier domain.

    N : int
        If N < number of images, only keep N images, 
        either the first N if shuffle = False, or random
        N images if shuffle = True.

    shuffle : bool
        Shuffle the data set if True, do not shuffle otherwise.

    N_px_noise : int
        The size of the corner crop that is used to estimate the noise.
        If 0, the noise is not estimated and sigma_noise is set to ones.
    
    N_imgs_noise : int
        Number of images that are used to estimate the noise.

    Returns:
    -------
    processed_data : dict
        Dictionary with the processed data. 
    """

    ctf_params0 = params0["ctf_params"]
    pixel_size0 = params0["pixel_size"]
    angles0 = params0["angles"]
    shifts0 = params0["shifts"]

    print(f"imgs0.shape = {imgs0.shape}")
    print(f"pixel_size0.shape = {pixel_size0.shape}")
    print(f"angles0.shape = {angles0.shape}")
    print(f"shifts0.shape = {shifts0.shape}")
    print(f"ctf_params0.shape = {ctf_params0.shape}", flush = True)
    nx0 = imgs0.shape[-1]

    # Keep N points at random.
    if N is None or N > imgs0.shape[0]:
        N = imgs0.shape[0]

    if shuffle:
        idxrand = np.random.permutation(imgs0.shape[0])[:N]
        print("Shuffle = True")
    else:
        idxrand = np.arange(N)
        print("Shuffle = False")
                    
    print(f'N = {N}', flush = True)

    imgs0 = imgs0[idxrand]
    pixel_size = pixel_size0[idxrand]
    angles = angles0[idxrand]
    shifts = shifts0[idxrand]
    ctf_params = ctf_params0[idxrand]

    file = open(out_dir + '/idxrand','wb')
    pickle.dump(idxrand, file)
    file.close()

    # Take FFT
    print("Taking FFT of the images...", end="", flush=True)
    t0 = time.time()
    imgs_f = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs0])
    print(f"done. Time: {time.time()-t0} seconds.", flush = True) 

    # Create grids
    # Assume the pixel size is the same for all images
    nx = imgs_f.shape[-1]
    px = pixel_size[0]
    N = imgs_f.shape[0]

    x_grid = create_grid(nx, px)
    print(f"x_grid = {x_grid}", flush = True)

    # Crop images
    if nx_crop is not None:
        nx = nx_crop
        imgs_f, x_grid = crop_fourier_images(imgs_f, x_grid, nx)
        print(f"new x_grid = {x_grid}", flush = True)
    
    # Vectorise images
    imgs_f = imgs_f.reshape(N, -1)
    print(f"Vectorised imgs_f.shape = {imgs_f.shape}", flush = True)

    # Create mask
    centre = (0,0,0)
    if nx % 2 == 0:
        radius = (x_grid[1]/2 - 1) * x_grid[0] 
    else:
        radius = (x_grid[1]-1)/2 * x_grid[0] 
    mask = create_3d_mask(x_grid, centre, radius)
    print(f"Mask radius = {radius}", flush = True)

    if N_px_noise > 0:
        if N_imgs_noise is None or N_imgs_noise > N:
            N_imgs_noise = N

        print(f"Estimating the noise using the {N_px_noise} x {N_px_noise} corners of the first {N_imgs_noise} images.", flush=True)
        t0 = time.time()
        sigma_noise = estimate_noise_radial(imgs0[:N_imgs_noise], nx_empty = N_px_noise, nx_final = nx)
        print(f"Noise estimation done. Time: {time.time()-t0} seconds.", flush=True) 
    else:
        print(f"Noise free - setting sigma_noise = 1", flush=True)
        sigma_noise = np.ones((nx*nx,))

    processed_data = {
        "imgs_f" : imgs_f, 
        "pixel_size" : pixel_size, 
        "angles" : angles,
        "shifts" : shifts,
        "ctf_params" : ctf_params, 
        "idxrand" : idxrand,
        "nx" : nx,
        "x_grid" : x_grid,
        "mask" : mask,
        "sigma_noise" : sigma_noise
    }

    return processed_data
