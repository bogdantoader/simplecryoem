import numpy as np
import time 

from src.emfiles import load_data
from src.utils import create_grid, crop_fourier_images


def create_het_dataset(data_dirs, star_files, N_set, nx_crop):
    """Create a discrete heterogeneous dataset from two different sets.

    Parameters:
    ----------
    data_dirs, star_files : 
            Lists of directory paths and star file paths in the respective directories

    N_set   : array[int] 
            Number of particles to keep from each set

    nx_crop      : Int
            Crop each image to be nx x nx pixels

    Returns:
    --------

    """

    assert(len(data_dirs) == len(star_files))
    assert(len(N_set) == len(data_dirs))

    # Read and process each dataset
    imgs = []
    imgs_f = []
    ctf_params = []
    pixel_size = []
    angles = []
    shifts = []
    z = []
    x_grids = []
    for i in range(len(data_dirs)):
        print(f"Reading dataset {i}")

        params_i, imgs_i = load_data(data_dirs[i], star_files[i], load_imgs = True, fourier = False)

        ctf_params_i = params_i["ctf_params"]
        pixel_size_i = params_i["pixel_size"]
        angles_i = params_i["angles"]
        shifts_i = params_i["shifts"]

        print(f"imgs_{i}.shape = {imgs_i.shape}")
        print(f"pixel_size_{i}.shape = {pixel_size_i.shape}")
        print(f"angles_{i}.shape = {angles_i.shape}")
        print(f"shifts_{i}.shape = {shifts_i.shape}")
        print(f"ctf_params_{i}.shape = {ctf_params_i.shape}")

        nx0_i = imgs_i.shape[-1]

        # Only keep N_set[i] particles
        idxrand_i = np.random.permutation(imgs_i.shape[0])[:N_set[i]]
                        
        imgs_i = imgs_i[idxrand_i]
        pixel_size_i = pixel_size_i[idxrand_i]
        angles_i = angles_i[idxrand_i]
        shifts_i = shifts_i[idxrand_i]
        ctf_params_i = ctf_params_i[idxrand_i]

        # FFT
        print(f"Taking FFT of dataset {i}...", end="", flush=True)
        t0 = time.time()
        imgs_f_i = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs_i])
        print(f"{time.time()-t0} seconds.")

        # Assume the pixel size is the same for all images
        nx_i = imgs_f_i.shape[-1]
        px_i = pixel_size_i[0]
        N_i = imgs_f_i.shape[0]

        # Create grids
        x_grid_i = create_grid(nx_i, px_i)
        print(f"x_grid_{i} = {x_grid_i}")

        # Crop Fourier images to required dimension
        imgs_f_i, x_grid_i = crop_fourier_images(imgs_f_i, x_grid_i, nx_crop)
        print(f"Cropped x_grid_{i} = {x_grid_i}")

        # Finally, vectorize the images
        imgs_f_i = imgs_f_i.reshape(N_i, -1)

        # The class assignment variable
        z_i = i * np.ones((N_set[i], ))


        imgs.append(imgs_i)
        imgs_f.append(imgs_f_i)
        ctf_params.append(ctf_params_i)
        pixel_size.append(pixel_size_i)
        angles.append(angles_i)
        shifts.append(shifts_i)
        z.append(z_i)
        x_grids.append(x_grid_i)

    # Concatenate all the array (except the original images)
    imgs_f = np.concatenate(imgs_f, axis=0)
    ctf_params = np.concatenate(ctf_params, axis=0)
    pixel_size = np.concatenate(pixel_size, axis=0)
    angles = np.concatenate(angles, axis=0)
    shifts = np.concatenate(shifts, axis=0)
    z = np.concatenate(z, axis=0)

    # Make all pixel sizes equal to the pixel size of the first dataset
    pixel_size = pixel_size[0] * np.ones(pixel_size.shape)

    # And also keep the x_grid of the first class
    x_grid = x_grids[0]

    # Final shuffling to mix the classes
    idx = np.random.permutation(imgs_f.shape[0])

    imgs_f = imgs_f[idx]
    ctf_params = ctf_params[idx]
    angles = angles[idx]
    shifts = shifts[idx]
    z = z[idx].astype(np.int64)

    # Note that the real space images are not cropped or shuffled.
    # They are used for estimating the noise.
    return imgs, imgs_f, ctf_params, pixel_size, angles, shifts, z, x_grid





