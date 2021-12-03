import numpy as np
from external.pyem.pyem import star
from src.ctf import get_ctf_params_from_df_row
import mrcfile

# TODO: need to check if I should use IMAGE_PATH and IMAGE_INDEX instead of
# IMAGE_ORIGINAL_PATH and IMAGE_ORIGINAL_INDEX
def get_data_from_df(df, data_dir):
    """Given a data frame as returned by star.parse_star, extract the useful
    information."""

    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)

    imgs = []
    pixel_size = []
    angles = []
    shifts = []
    ctf_params = []

    particle_paths = df[star.UCSF.IMAGE_ORIGINAL_PATH].unique()
    for path in particle_paths:
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

            angs = np.deg2rad(np.array([angpsi,angtilt, angrot]))
            sh = np.array([shx, shy])
            ctf_p = get_ctf_params_from_df_row(p, px)

            img_index = group[star.UCSF.IMAGE_ORIGINAL_INDEX][index]
            img = group_data[img_index]

            pixel_size.append(px)
            angles.append(angs)
            shifts.append(sh)
            ctf_params.append(ctf_p)
            imgs.append(img)

    pixel_size = np.array(pixel_size)
    angles = np.array(angles)
    shifts = np.array(shifts)
    ctf_params = np.array(ctf_params)
    imgs = np.array(imgs)
    

    #imgs = jnp.concatenate(imgs, axis = 0)
    #pixel_size = jnp.concatenate(pixel_size, axis = 0)
    #angles = jnp.concatenate(angles, axis = 0)    
    #shifts = jnp.concatenate(shifts, axis = 0)
    #ctf_params = jnp.concatenate(ctf_params, axis = 0)

    return imgs, pixel_size, angles, shifts, ctf_params
