import jax.numpy as jnp
from external.pyem.pyem import star
from src.ctf import get_ctf_params_from_df
import mrcfile


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
    for path in particle_paths[:10]:
        with mrcfile.open(data_dir + path) as mrc:
            data = mrc.data
            if data.ndim == 2:
                data = jnp.array([data])

        group = gb.get_group(path)
        angrot = group[star.Relion.ANGLEROT]
        angtilt = group[star.Relion.ANGLETILT]
        angpsi = group[star.Relion.ANGLEPSI]

        giter = group.iterrows()
        px = [star.calculate_apix(p) for _, p in giter]

        shx = group[star.Relion.ORIGINX] * px
        shy = group[star.Relion.ORIGINY] * px

        if angrot.shape[0] != data.shape[0]:
            continue

        px = jnp.array(px)
        angs = jnp.deg2rad(jnp.array([angpsi,angtilt, angrot]).transpose())
        sh = jnp.array([shx, shy]).transpose()
        ctf_p = get_ctf_params_from_df(group, px)


        imgs.append(data)
        pixel_size.append(px)
        angles.append(angs)
        shifts.append(sh)
        ctf_params.append(ctf_p)

    imgs = jnp.concatenate(imgs, axis = 0)
    pixel_size = jnp.concatenate(pixel_size, axis = 0)
    angles = jnp.concatenate(angles, axis = 0)    
    shifts = jnp.concatenate(shifts, axis = 0)
    ctf_params = jnp.concatenate(ctf_params, axis = 0)

    return imgs, pixel_size, angles, shifts, ctf_params
