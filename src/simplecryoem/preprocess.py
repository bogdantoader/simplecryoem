import numpy as np
import time
from simplecryoem.utils import *
from simplecryoem.noise import estimate_noise_radial


def preprocess(imgs0, params0, nx_crop=None, idx=None, N_px_noise=0, N_imgs_noise=None):
    """Basic preprocessing of particle images to extract relevant information
    for reconstruction.

    Parameters:
    -----------
    imgs0 : [N0 x nx x nx] array
        Stack of particle images as returned by emfiles.load_data

    params0 : dict
        Dictionary with keys 'ctf_params', 'pixel_size', 'angles', 'shifts',
        each containing an array.

    nx_crop: int
        Downsample images to dimensions nx_crop x nx_crop.
        This is done by cropping the images in the Fourier domain.

    idx : list[int]
        If given, only apply the preprocessing steps to imgs0[idx].

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
    print(f"ctf_params0.shape = {ctf_params0.shape}", flush=True)
    nx0 = imgs0.shape[-1]

    N0 = imgs0.shape[0]
    print(f"N0 = {N0}", flush=True)

    if idx is None:
        idx = np.arange(N0)
        print("idx not provided")
    else:
        assert np.max(idx) < N0
        assert idx.shape[0] <= N0
        print("idx provided")

    N = idx.shape[0]
    print(f"N = {N}", flush=True)

    imgs0 = imgs0[idx]
    pixel_size = pixel_size0[idx]
    angles = angles0[idx]
    shifts = shifts0[idx]
    ctf_params = ctf_params0[idx]

    # Take FFT
    print("Taking FFT of the images:", flush=True)
    t0 = time.time()

    # Hardcoded batch for now, as this works (or should work) most
    # of the time.
    n_batches = 10
    imgs0_batches = np.array_split(imgs0, n_batches)
    imgs_f = []
    for idx_batch, imgs0_batch in enumerate(imgs0_batches):
        print(f"Batch {idx_batch+1}/{n_batches} ", end="")
        t01 = time.time()
        imgs_f_batch = np.array(
            [np.fft.fft2(np.fft.ifftshift(img)) for img in imgs0_batch]
        )
        imgs_f.append(imgs_f_batch)
        print(f"{time.time()-t01 : .2f} sec.")
    imgs_f = np.concatenate(imgs_f, axis=0)
    print(f"FFT done. Time: {time.time()-t0 : .2f} sec.", flush=True)

    # Create grids
    # Assume the pixel size is the same for all images
    nx = imgs_f.shape[-1]
    px = pixel_size[0]
    N = imgs_f.shape[0]

    x_grid = create_grid(nx, px)
    print(f"x_grid = {x_grid}", flush=True)

    # Crop images
    if nx_crop is not None:
        nx = nx_crop
        imgs_f, x_grid = crop_fourier_images(imgs_f, x_grid, nx)
        print(f"new x_grid = {x_grid}", flush=True)

    # Vectorise images
    imgs_f = imgs_f.reshape(N, -1)
    print(f"Vectorised imgs_f.shape = {imgs_f.shape}", flush=True)

    # Create mask
    centre = (0, 0, 0)
    if nx % 2 == 0:
        radius = (x_grid[1] / 2 - 1) * x_grid[0]
    else:
        radius = (x_grid[1] - 1) / 2 * x_grid[0]
    mask = create_3d_mask(x_grid, centre, radius)
    print(f"Mask radius = {radius}", flush=True)

    if N_px_noise > 0:
        if N_imgs_noise is None or N_imgs_noise > N:
            N_imgs_noise = N

        print(
            f"Estimating the noise using the {N_px_noise} x {N_px_noise} corners of the first {N_imgs_noise} images.",
            flush=True,
        )
        t0 = time.time()
        sigma_noise = estimate_noise_radial(
            imgs0[:N_imgs_noise], nx_empty=N_px_noise, nx_final=nx
        )
        print(f"Noise estimation done. Time: {time.time()-t0 : .2f} sec.", flush=True)
    else:
        print("Noise free, setting sigma_noise = 1", flush=True)
        sigma_noise = np.ones((nx * nx,))

    processed_data = {
        "imgs_f": imgs_f,
        "pixel_size": pixel_size,
        "angles": angles,
        "shifts": shifts,
        "ctf_params": ctf_params,
        "idx": idx,
        "nx": nx,
        "x_grid": x_grid,
        "mask": mask,
        "sigma_noise": sigma_noise,
    }

    return processed_data
