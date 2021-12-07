import jax.numpy as np

#
# CTF functions
#
CTF_def1   = 0
CTF_def2   = 1
CTF_angast = 2
CTF_phase  = 3
CTF_kv     = 4
CTF_ac     = 5
CTF_cs     = 6
CTF_bf     = 7

CTF_PARAMS_DIM = 8 

CCC = 1

def eval_ctf(s, a, c ):
    """
    Parameters
    ----------
    s: Precomputed frequency points for CTF evaluation (1/Å).
    a: Precomputed frequency points angles.
    c: array
        def1: 1st prinicipal underfocus distance (Å).
        def2: 2nd principal underfocus distance (Å).
        angast: Angle of astigmatism (deg) from x-axis to azimuth.
        phase: Phase shift (deg).
        kv:  Microscope acceleration potential (kV).
        ac:  Amplitude contrast in [0, 1.0].
        cs:  Spherical aberration (mm).
        bf:  B-factor, divided by 4 in exponential, lowpass positive.
    
    Returns
    -------
    ctf :

    Changes from pyEM:
    * no low pass filter
    
    """

    #print( s.shape, a.shape, c.shape )

    def1   = c[CTF_def1]
    def2   = c[CTF_def2]
    angast = c[CTF_angast]
    phase  = c[CTF_phase]
    kv     = c[CTF_kv]
    ac     = c[CTF_ac]
    cs     = c[CTF_cs]
    bf     = c[CTF_bf]
    
    angast = np.deg2rad(angast)

    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / np.sqrt(kv * (1. + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2. * 2 * lamb
    k2 = np.pi / 2. * cs * lamb**3
    k3 = np.sqrt(1 - ac**2)
    k4 = bf / 4.  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    #if lp != 0:  # Hard low- or high-pass.
    #    s *= s <= (1. / lp)
    s_2 = s**2
    s_4 = s_2**2
    dZ = def_avg + def_dev * (np.cos(2 * (a - angast)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * np.sin(gamma) - ac*np.cos(gamma))
    #if bf != 0:  # Enforce envelope.
    #ctf *= np.exp(-k4 * s_2)
    ctf = ctf * np.exp(-k4 * s_2)
    
    return ctf

