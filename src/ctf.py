import jax.numpy as jnp
import numpy as np
from external.pyem.pyem import star

def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):
    """CTF function from pyEM.

    :param s, a: r, theta polar coordinates in frequency space
    :param def1: 1st prinicipal underfocus distance (Å).
    :param def2: 2nd principal underfocus distance (Å).
    :param angast: Angle of astigmatism (deg) from x-axis to azimuth.
    :param phase: Phase shift (deg).
    :param kv:  Microscope acceleration potential (kV).
    :param ac:  Amplitude contrast in [0, 1.0].
    :param cs:  Spherical aberration (mm).
    :param bf:  B-factor, divided by 4 in exponential, lowpass positive.
    :param lp:  Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = jnp.deg2rad(angast)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / jnp.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = jnp.pi / 2. * 2 * lamb
    k2 = jnp.pi / 2. * cs * lamb**3
    k3 = jnp.sqrt(1 - ac**2)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = jnp.deg2rad(phase)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1. / lp)
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (jnp.cos(2 * (a - angast)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * jnp.sin(gamma) - ac*jnp.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= jnp.exp(-k4 * s_2)
    return ctf


def get_ctf_params_from_df_row(p, pixel_size):
    """Extract the CTF parameters from a dataframe, as arrays 
    with elements in the same order as the arguments of eval_ctf."""

    ctf_params = np.array([p[star.Relion.DEFOCUSU], 
        p[star.Relion.DEFOCUSV],
        p[star.Relion.DEFOCUSANGLE], 
        p[star.Relion.PHASESHIFT],
        p[star.Relion.VOLTAGE],
        p[star.Relion.AC],
        p[star.Relion.CS],
        0,
        2 * pixel_size])

    return ctf_params


