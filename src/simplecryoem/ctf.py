import jax
import jax.numpy as jnp
from external.pyem.pyem import star


# def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):


def eval_ctf(s, a, params):
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
    # TODO: use constants like in Roy's code
    def1 = params[0]
    def2 = params[1]
    angast = params[2]
    phase = params[3]
    kv = params[4]
    ac = params[5]
    cs = params[6]
    bf = params[7]
    lp = params[8]

    angast = jnp.deg2rad(angast)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / jnp.sqrt(kv * (1.0 + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = jnp.pi / 2.0 * 2 * lamb
    k2 = jnp.pi / 2.0 * cs * lamb**3
    k3 = jnp.sqrt(1 - ac**2)
    k4 = bf / 4.0  # B-factor, follows RELION convention.
    k5 = jnp.deg2rad(phase)  # Phase shift.

    # if lp != 0:  # Hard low- or high-pass.
    #    s *= s <= (1. / lp)

    # Jaxify the above if else
    s *= jax.lax.cond(
        lp != 0,
        true_fun=lambda _: (s <= (1.0 / lp)).astype(jnp.float64),
        false_fun=lambda _: jnp.ones(s.shape),
        operand=None,
    )

    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (jnp.cos(2 * (a - angast)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * jnp.sin(gamma) - ac * jnp.cos(gamma))

    # if bf != 0:  # Enforce envelope.
    #    ctf *= jnp.exp(-k4 * s_2)

    ctf *= jax.lax.cond(
        bf != 0,
        true_fun=lambda _: jnp.exp(-k4 * s_2),
        false_fun=lambda _: jnp.ones(ctf.shape),
        operand=None,
    )

    return ctf


def get_ctf_params_from_df_row(p, pixel_size):
    """Extract the CTF parameters from a dataframe, as arrays
    with elements in the same order as the arguments of eval_ctf."""

    ctf_params = jnp.array(
        [
            p[star.Relion.DEFOCUSU],
            p[star.Relion.DEFOCUSV],
            p[star.Relion.DEFOCUSANGLE],
            p[star.Relion.PHASESHIFT],
            p[star.Relion.VOLTAGE],
            p[star.Relion.AC],
            p[star.Relion.CS],
            0,
            2 * pixel_size,
        ]
    )

    return ctf_params