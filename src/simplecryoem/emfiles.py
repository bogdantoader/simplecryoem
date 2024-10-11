import time
import numpy as np
import pandas as pd
import mrcfile
import sys
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

    df = parse_star(data_dir + star_file, keep_index=False)

    print(f"load_data: number of partcles: {len(df)}")

    t0 = time.time()
    pixel_size, angles, shifts, ctf_params, imgs = get_data_from_df(
        df, data_dir, load_imgs)
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


def get_data_from_df(df, data_dir, load_imgs=False):
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

    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)

    imgs = []
    pixel_size = []
    angles = []
    shifts = []
    ctf_params = []

    particle_paths = df[star.UCSF.IMAGE_ORIGINAL_PATH].unique()

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


def parse_star(star_path, keep_index=False, augment=True, nrows=sys.maxsize):
    """Function from pyem, modify to add the 'suffixes' option in the pandas.merge function.
    Without it, the resulting merged dataframe will rename the common columns."""

    tables = star.star_table_offsets(star_path)
    dfs = {t: star.parse_star_table(star_path, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
           for t in tables}
    if star.Relion.OPTICDATA in dfs:
        if star.Relion.PARTICLEDATA in dfs:
            data_table = star.Relion.PARTICLEDATA
        elif star.Relion.MICROGRAPHDATA in dfs:
            data_table = star.Relion.MICROGRAPHDATA
        elif star.Relion.IMAGEDATA in dfs:
            data_table = star.Relion.IMAGEDATA
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs[star.Relion.OPTICDATA],
                          dfs[data_table], on=star.Relion.OPTICSGROUP,
                          suffixes=('_optics', None))
        else:
            df = dfs[star.Relion.OPTICDATA]
    else:
        df = dfs[next(iter(dfs))]
    df = star.check_defaults(df, inplace=True)
    if augment:
        star.augment_star_ucsf(df, inplace=True)
    return df
