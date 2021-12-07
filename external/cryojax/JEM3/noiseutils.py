from typing import TypeVar
import numpy as onp
from functools import partial

import jax
import jax.numpy as jnp
from   jax import vmap
from   jax import jit
#from jax import tree_util

from cryojax.JEM3 import utils     as jem3utils
from cryojax.JEM3 import lininterp as jem3lininterp
from cryojax.JEM3 import projutils as jem3projutils
from cryojax.JEM3 import coorutils as jem3coorutils


import time



def simple_noise_est_scalar(images_fft):
    """ crude noise estimate
    """
    ns0 = onp.var(images_fft)
    return ns0

def simple_noise_est_2d(images_fft):
    """ crude noise estimate
    """
    ns0 = onp.var(images_fft,axis=0)
    return ns0

def block_noise_est_scalar(images_fft, lsub = 10):
    """ crude noise estimate based on corner of images
    
    parameters
    ----------
    images_fft : numpy array n x L x L
    lsub       : integer. Size of corner
    
    returns
    -------
    ns0        : real number. variance (scaled for use with the fft of the original image).
    
    TODO: fft back?

    """
    images2 = onp.fft.ifft2(images_fft)
    images2_shft = onp.fft.fftshift(images2,axes=(-1,-2))
    images2_sub = images2_shft[:,:lsub,:lsub]
    #print(images2_sub.shape)
    ns0 = onp.var(images2_sub)
    ns0 = ns0 * (images_fft.shape[-2]**2) # scaling correction for fft
    
    return ns0


def noise_weight_correct(ns):
    wgt = 1.0/(2.0*ns)
    #wgt = wgt / 2.0        # we count each point twice (the poitn at -x is the complex conjugate of the point at x). 
    #wgt[0,0] = wgt[0,0] *2 # except for the zero frequency
    return wgt

############################################################
#### Tests
############################################################



def test001():
    L = 100
    n = 500
    images = onp.random.randn(n,L,L)
    print( onp.var(images) )
    images_fft = onp.fft.fft2(images)
    print( onp.var(images_fft) )
    nbs = block_noise_est_scalar(images_fft, lsub=50)
    print(nbs)



if __name__ == '__main__':
    print("==== noiseutils ====")
    test001()
