#
#
#
#
#
#

from __future__ import print_function
import logging
import numpy as onp
from pyem import star
from pyem import util


from pyem import vop

import jax.numpy as np

from jax import grad, jit, vmap, value_and_grad
from jax import random
import scipy as sp

from jax import device_put
import time
from functools import partial


import cryojax.JEM3.ctf as ctf3


#
# Services
#

def df2rot( df ):
    """ Extracts rotations from dataframe used by pyEM.

    Parameters:
        df : dataframe
    
    returns:
        angs : n x 3 numpy array of rotations rot,tilt,psi (in rad)
        rots : n x 3 x 3 numpy array of rotation matrices 

    """

    angs = onp.array([(onp.deg2rad(p[star.Relion.ANGLEROT]),
            onp.deg2rad(p[star.Relion.ANGLETILT]),
            onp.deg2rad(p[star.Relion.ANGLEPSI])) 
            for jj,p in df.iterrows()])
        
    rots = onp.array([ util.euler2rot(p[0],p[1],p[2]) for p in angs])
    
    return rots, angs

def df2originpx( df ):
    """ Extracts Origin in pixels
    
    """
    originpx = onp.array([ (p[star.Relion.ORIGINX] , p[star.Relion.ORIGINY])
                           for jj,p in df.iterrows()])
    return originpx


def df2apiximg( df ):
    """ Extracts image apix (units?)
    
    """
    apiximg = onp.array([ star.calculate_apix(p)
                           for jj,p in df.iterrows()])
    return apiximg


def df2ctfparam(df):
    """ Extracts image apix (units?)
    
    returns:
        defu : 1st prinicipal underfocus distance (Å). 
        defv : 2st prinicipal underfocus distance (Å).
        defang : Angle of astigmatism (deg) from x-axis to azimuth.
        phaseshift : Phase shift (deg).
        kv : Microscope acceleration potential (kV).
        ac : Amplitude contrast in [0, 1.0].
        cs : Spherical aberration (mm).
        bf : B-factor, divided by 4 in exponential, lowpass positive.
    """

    c = onp.zeros( (df.shape[0], ctf3.CTF_PARAMS_DIM) )
    
    c[:,ctf3.CTF_def1]   = onp.array([ (p[star.Relion.DEFOCUSU])
                           for jj,p in df.iterrows()])

    c[:,ctf3.CTF_def2]   = onp.array([ (p[star.Relion.DEFOCUSV])
                           for jj,p in df.iterrows()])
    c[:,ctf3.CTF_angast] = onp.array([ (p[star.Relion.DEFOCUSANGLE])
                           for jj,p in df.iterrows()])
    c[:,ctf3.CTF_phase]  = onp.array([ (p[star.Relion.PHASESHIFT])
                           for jj,p in df.iterrows()])
    c[:,ctf3.CTF_ac]     = onp.array([ (p[star.Relion.AC])
                           for jj,p in df.iterrows()])
    c[:,ctf3.CTF_cs]     = onp.array([ (p[star.Relion.CS])
                           for jj,p in df.iterrows()])    
    c[:,ctf3.CTF_kv]     = onp.array([ (p[star.Relion.VOLTAGE])
                           for jj,p in df.iterrows()])
    c[:,ctf3.CTF_bf]     = onp.array([ (p[star.Relion.CTFBFACTOR])
                           for jj,p in df.iterrows()])
    return c


def df2ctfparam_old(df):
    """ Extracts image apix (units?)
    
    returns:
        defu : 1st prinicipal underfocus distance (Å). 
        defv : 2st prinicipal underfocus distance (Å).
        defang : Angle of astigmatism (deg) from x-axis to azimuth.
        phaseshift : Phase shift (deg).
        kv : Microscope acceleration potential (kV).
        ac : Amplitude contrast in [0, 1.0].
        cs : Spherical aberration (mm).
        bf : B-factor, divided by 4 in exponential, lowpass positive.
    """


    
    
    defu = onp.array([ (p[star.Relion.DEFOCUSU])
                           for jj,p in df.iterrows()])
    defv = onp.array([ (p[star.Relion.DEFOCUSV])
                           for jj,p in df.iterrows()])
    defang = onp.array([ (p[star.Relion.DEFOCUSANGLE])
                           for jj,p in df.iterrows()])
    phaseshift = onp.array([ (p[star.Relion.PHASESHIFT])
                           for jj,p in df.iterrows()])
    ac = onp.array([ (p[star.Relion.AC])
                           for jj,p in df.iterrows()])
    cs = onp.array([ (p[star.Relion.CS])
                           for jj,p in df.iterrows()])    
    kv = onp.array([ (p[star.Relion.VOLTAGE])
                           for jj,p in df.iterrows()])
    bf = onp.array([ (p[star.Relion.CTFBFACTOR])
                           for jj,p in df.iterrows()])
    #p[star.Relion.DEFOCUSU], p[star.Relion.DEFOCUSV],
    #                     p[star.Relion.DEFOCUSANGLE],
    #                     p[star.Relion.PHASESHIFT], p[star.Relion.VOLTAGE],
    #                     p[star.Relion.AC], p[star.Relion.CS]
    return defu, defv, defang, phaseshift, kv, ac,cs, bf
