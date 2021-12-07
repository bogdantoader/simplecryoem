from typing import TypeVar
import jax.numpy as jnp
#from jax import vmap
#import numpy as onp

#from functools import partial


#from cryojax.JEM3 import utils as jem3utils
#from cryojax.JEM3 import lininterp as jem3lininterp











def inplane_shift_filter(pts  , xyshft):
    """ Produces the frequency domain filter that implements an in-plane shift.
    Should also work for 3-D (d=3) shifts.
    
    Parameters 
    ----------
    pts     : real array N_1 x ... x N_m x d
              frequency domain coordinates in 1/unit
    xyshft  : real array d
              real domain shift in units (e.g., angstrom, pixels)
    
    Returns
    -------
    phs     : complex array N_1 x ... x N_m
              phase shift for each frequency domain point
    
    TODO: check both shifts in pixels and angst. Check def of points ad scaling

    """
    #print( pts.shape, xyshft.shape )
    phs = jnp.exp(-2 * jnp.pi * 1j * (pts * xyshft).sum(-1) )
    #print(phs.shape)
    return phs


