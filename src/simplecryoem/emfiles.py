import time
import numpy as np
import pandas as pd
import mrcfile
import starfile
from pyem import star
from simplecryoem.ctf import get_ctf_params_from_df_row

IMAGE_INDEX = 'ImageIndex'
IMAGE_PATH = 'ImagePath'


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
    --------
    params : dict
        Dictionary containing arrays of ctf_params, pixel_size, angles, shifts
        for all the loaded images.

    imgs : N x nx x nx array
        The images loaded from the file.

    """

    # df = star.parse_star(data_dir + star_file, keep_index=False)

    star_data = starfile.read(data_dir + star_file)
    if type(star_data) is dict:
        df_particles = star_data['particles']
        df_optics = star_data['optics']
        df_optics.set_index(star.Relion.OPTICSGROUP, inplace=True)
    else:
        df_particles = star_data
        df_optics = None

    print(f"load_data: number of partcles: {len(df_particles)}")

    df_particles[IMAGE_INDEX] = pd.to_numeric(
        df_particles[star.Relion.IMAGE_NAME].str.split("@").str[0]
    ) - 1
    df_particles[IMAGE_PATH] = df_particles[star.Relion.IMAGE_NAME].str.split(
        "@").str[1]

    t0 = time.time()
    pixel_size, angles, shifts, ctf_params, imgs = get_data_from_df(
        df_particles, df_optics, data_dir, load_imgs
    )
    t1 = time.time()
    print(f"load_data: data loaded, time: {t1-t0: .2f} sec.")

    if fourier:
        imgs = np.array([np.fft.fft2(np.fft.ifftshift(img)) for img in imgs])
        t2 = time.time()
        print(f"load_data: FFT of data, time: {t2-t1: .2f} sec.")

    params = {
        "ctf_params": ctf_params,
        "pixel_size": pixel_size,
        "angles": angles,
        "shifts": shifts,
    }

    return params, imgs


def get_data_from_df(df, df_optics, data_dir, load_imgs=False):
    """Given a data frame as returned by star.parse_star, extract the useful
    information.

    Parameters:
    ----------
    df : the data frame
    data_dir : string
        Location of the star file.
    load_imgs : boolean
        Load images or only their parameters.

    Returns:
    --------
    pixel_size :
        Array of pixel size for all images.
    angles : N x 3
        Array of Euler angles for all images (or Nones if not present in the file).
    shifts : N x 2
        Array of shifts for all images (or None if non present)
    ctf_params:
        Array containing the ctf parameters for all images
    imgs : N x nx x nx
        Arrays containing the images if load_imgs=True and empty array otherwise.
    """

    gb = df.groupby(IMAGE_PATH)

    imgs = []
    pixel_size = []
    angles = []
    shifts = []
    ctf_params = []

    particle_paths = df[IMAGE_PATH].unique()

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

            px = get_pixel_size(p, df_optics)

            if star.Relion.ORIGINX in p:
                shx = p[star.Relion.ORIGINX] * px
            elif star.Relion.ORIGINXANGST in p:
                shx = p[star.Relion.ORIGINXANGST]
            else:
                shx = None

            if star.Relion.ORIGINY in p:
                shy = p[star.Relion.ORIGINY] * px
            elif star.Relion.ORIGINYANGST in p:
                shy = p[star.Relion.ORIGINYANGST]
            else:
                shy = None

            angs = np.array([angpsi, angtilt, angrot])
            if angrot is not None:
                angs = np.deg2rad(angs)
            sh = np.array([shx, shy])

            ctf_p = get_ctf_params_from_df_row(p, df_optics, px)

            img_index = p[IMAGE_INDEX]
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


def get_pixel_size(df_row, optics_df):
    if star.Relion.IMAGEPIXELSIZE in optics_df:
        return optics_df[star.Relion.IMAGEPIXELSIZE].loc[df_row[star.Relion.OPTICSGROUP]]
    if star.Relion.IMAGEPIXELSIZE in df_row:
        return df_row[star.Relion.IMAGEPIXELSIZE]
    if star.Relion.MICROGRAPHPIXELSIZE in df_row:
        return df_row[star.Relion.MICROGRAPHPIXELSIZE]
    return 10000.0 * df[star.Relion.DETECTORPIXELSIZE] / df[star.Relion.MAGNIFICATION]
