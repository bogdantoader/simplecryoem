import time
import numpy as np
import jax
import jax.numpy as jnp
from external.pyem.pyem import star
from src.ctf import get_ctf_params_from_df_row
import mrcfile



def load_data(data_dir, star_file, load_imgs = False):
    """Load all the required information from star files and mrcs files.

    Parameters:
    ----------
    data_dir : string
               Directory containing the star file
    star_file: string
               Name of the star file (relative to the directory path).
    Returns:

    """
    
    df = star.parse_star(data_dir + star_file, keep_index = False)

    print("load_data: number of partcles: ", len(df))
    t0 = time.time()
    pixel_size, angles, shifts, ctf_params, imgs = get_data_from_df(df, data_dir, load_imgs)
    t1 = time.time()
    print("load_data: data loaded, time: ", t1-t0) 

    if load_imgs:
        imgs_f = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs])
        t2 = time.time()
        print("load_data: FFT of data, time: ", t2-t1)
    else:
        imgs_f = imgs

    params = {'ctf_params' : ctf_params,
              'pixel_size' : pixel_size,
              'angles'     : angles,
              'shifts'     : shifts}

    return params, imgs_f

# TODO: Move this to utils
# TODO 2: write tests for this function and make with work with odd dimensions too.
def crop_fourier_images(imgs, x_grid, nx):
    """Given an N x nx0 x nx0 array of N images of dimension nx0 x nx0 in the 
    frequency space with the standard ordering, crop the high-frequency entries 
    to reduce the image to the dimensions nx x nx. 
    Also adjust the grid arrays accordingly. 

    Parameters:
    ----------
    imgs : N x nx0 x nx0 array
        N stacked images of dimensions nx0 x nx0 in the Fourier domain 
        and standard ordering.
    x_grid: 2 x 1 array 
        Spacing and length of the Fourier grid in each dimension (we assume
        they are the same in all dimensions), in the format:
        [grid_spacing, grid_length].
    nx : integer
        The target length each dimension of the images after cropping.

    Returns:
    -------
        imgs_cropped: N x nx x nx)
            N stacked cropped images. 
        x_grid_cropped: 2 x 1 array
            The new Fourier grid corresponding to the cropped images.
    """
    
    N = imgs.shape[0]
    mid = imgs.shape[-1]/2

    #t0 = time.time()
    #imgs_cropped = np.zeros([N, nx, nx], dtype = np.complex64)
    #for i, f in enumerate(imgs):
    #    img = np.fft.fftshift(f)
    #    imgs_cropped[i] = np.fft.ifftshift(
    #            img[int(mid-nx/2):int(mid+nx/2), int(mid-nx/2):int(mid+nx/2)])
    #print(time.time()-t0)

    t0 = time.time()
    idx = jnp.concatenate([jnp.arange(nx/2),jnp.arange(-nx/2,0)]).astype(jnp.int64)
    imgs_cropped2 = imgs[jnp.ix_(jnp.arange(N),idx, idx)]
    print(time.time()-t0)

    # <<< IMPORTANT!!!>>> 
    # The grid must not be a Jax object.
    x_grid_cropped = np.array([x_grid[0], nx])

    #return jnp.array(imgs_cropped), imgs_cropped2, x_grid_cropped
    return imgs_cropped2, x_grid_cropped

def crop_fourier_volume(vol, x_grid, nx):
    """Same as above, but a volume."""

    vol = np.fft.fftshift(vol)
    mid = vol.shape[-1]/2

    vol_cropped = np.fft.ifftshift(
            vol[int(mid-nx/2):int(mid+nx/2), int(mid-nx/2):int(mid+nx/2), int(mid-nx/2):int(mid+nx/2)]
            )

    # <<< IMPORTANT!!!>>> 
    # The grid must not be a Jax object.
    x_grid_cropped = np.array([x_grid[0], nx])

    return vol_cropped, x_grid_cropped


def get_data_from_df(df, data_dir, load_imgs = False):
    """Given a data frame as returned by star.parse_star, extract the useful
    information."""

    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)

    imgs = []
    pixel_size = []
    angles = []
    shifts = []
    ctf_params = []

    particle_paths = df[star.UCSF.IMAGE_ORIGINAL_PATH].unique()
    #for path in particle_paths[:100]:
    for path in particle_paths:
        if load_imgs:
            with mrcfile.open(data_dir + path) as mrc:
                group_data = mrc.data
                if group_data.ndim == 2:
                    group_data = np.array([group_data])
        
        group = gb.get_group(path)

        for index, p in group.iterrows():

            angrot = p[star.Relion.ANGLEROT]
            angtilt = p[star.Relion.ANGLETILT]
            angpsi = p[star.Relion.ANGLEPSI]

            px = star.calculate_apix(p) 

            shx = p[star.Relion.ORIGINX] * px
            shy = p[star.Relion.ORIGINY] * px

            angs = np.deg2rad(np.array([angpsi, angtilt, angrot]))
            sh = np.array([shx, shy])
            ctf_p = get_ctf_params_from_df_row(p, px)

            img_index = p[star.UCSF.IMAGE_ORIGINAL_INDEX]
            if load_imgs:
                img = group_data[img_index]
                imgs.append(img)

            pixel_size.append(px)
            angles.append(angs)
            shifts.append(sh)
            ctf_params.append(ctf_p)

    pixel_size = np.array(pixel_size)
    angles = np.array(angles)
    shifts = np.array(shifts)
    ctf_params = np.array(ctf_params)
    imgs = np.array(imgs)

    return pixel_size, angles, shifts, ctf_params, imgs
