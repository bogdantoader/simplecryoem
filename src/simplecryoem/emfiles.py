import time
import numpy as np
from pyem import star
from simplecryoem.ctf import get_ctf_params_from_df_row
import mrcfile


def load_data(data_dir, star_file, load_imgs=False, fourier=True):
    """Load all the required information from star files and mrcs files.

    Parameters:
    ----------
    data_dir : string
               Directory containing the star file

    star_file : string
               Name of the star file (relative to the directory path).

    load_imgs : Boolean
                If true, load the actual images and if false, don't load them.

    fourier : Boolean
                Return the images in the Fourier domain if true, or the
                spatial domain otherwise.
    Returns:

    """

    df = star.parse_star(data_dir + star_file, keep_index=False)

    print(f"load_data: number of partcles: {len(df)}")
    t0 = time.time()
    pixel_size, angles, shifts, ctf_params, imgs = get_data_from_df(
        df, data_dir, load_imgs
    )
    t1 = time.time()
    print(f"load_data: data loaded, time: {t1-t0 : .2f} sec.")

    if fourier:
        imgs = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs])
        t2 = time.time()
        print(f"load_data: FFT of data, time: {t2-t1 : .2f} sec.")

    params = {
        "ctf_params": ctf_params,
        "pixel_size": pixel_size,
        "angles": angles,
        "shifts": shifts,
    }

    return params, imgs


def get_data_from_df(df, data_dir, load_imgs=False):
    """Given a data frame as returned by star.parse_star, extract the useful
    information."""

    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)

    imgs = []
    pixel_size = []
    angles = []
    shifts = []
    ctf_params = []

    particle_paths = df[star.UCSF.IMAGE_ORIGINAL_PATH].unique()
    # for path in particle_paths[:100]:

    for path in particle_paths:
        if load_imgs:
            with mrcfile.open(data_dir + path) as mrc:
                group_data = mrc.data
                if group_data.ndim == 2:
                    group_data = np.array([group_data])

        group = gb.get_group(path)

        for index, p in group.iterrows():
            angrot = p.get(star.Relion.ANGLEROT)
            angtilt = p.get(star.Relion.ANGLETILT)
            angpsi = p.get(star.Relion.ANGLEPSI)

            px = star.calculate_apix(p)

            shx = p.get(star.Relion.ORIGINX)
            shy = p.get(star.Relion.ORIGINY)

            if shx is not None:
                shx = shx * px
            if shy is not None:
                shy = shy * px

            angs = np.array([angpsi, angtilt, angrot])
            if angrot is not None:
                angs = np.deg2rad(angs)
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
