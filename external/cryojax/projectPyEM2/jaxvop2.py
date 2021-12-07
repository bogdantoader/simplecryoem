#
#
#
# Two types of fft: "full plane" (ifft) and "half plane" (rfft).
#
#
#
#

from __future__ import print_function
import logging
import numpy as npo

from pyem import vop

import jax.numpy as np

from jax import grad, jit, vmap, value_and_grad
from jax import random
import scipy as sp

from jax import device_put
import time
from functools import partial



#
# Services
#




#
# General services
#

#@jit
def rot_coor( rot, coor ):
    # single rotation
    x = (rot @ coor.reshape((-1,3,1))).reshape(coor.shape)    
    return x



#
# Linear interpolation
#



#
# 3D volume FFT in numpy (pyEM/Relion) compatibility
#

def ft3d_nofilters(vol, pfac):
    L = vol.shape[-1]
    
    padvol = np.pad(vol, int((L * pfac - L) // 2), "constant")    
    vol_correct_shift = np.fft.ifftshift(padvol)
    fvol = np.fft.fftn( vol_correct_shift )
    
    return fvol 



def ft3d_withpyemfilters_np(vol_np, pfac):
    """
    FT of a volume that is approximately compatible with pyEM.
    This one produces pfac* L x pfac*L x pfac*L volume (integer part of this if not integer numbers),
    as opposed to something line pfac*L+3 x pfac*L+3 pfac*L//2+1 in pyEM,
    because pyEM uses rfftn and we use fftn, and becuase of the padding used in pyEM.
    The function implements some filters that are used in pyEM. 
    
    TODO: a version that can be differentiated through. 
    
    input:
    * vol_np : volume (numpy)
    * pfac   : padding factor (1 is no padding).
    
    output:
    * FT of the volume 
    
    """
    # fft of volume with padding by pfac.
    # Filters compatible with pyEM. However, this uses fftn and not rfftn
    # TODO: rfftn version?
    
    L = vol_np.shape[-1]
    vol_correct_np = vop.grid_correct(vol_np, pfac=pfac, order=1)    # pyEM's correction. TODO: remove
    
    #padvol_np = npo.pad(vol_correct_np, int((L * pfac - L) // 2), "constant")    
    #vol_correct_shift_np = npo.fft.ifftshift(padvol_np)
    fvol = ft3d_nofilters(vol_correct_np, pfac)
    
    #
    # pyEM filter, unclear purpose. TODO: remove
    #
    #L1 = padvol_np.shape[0]
    L1 = fvol.shape[-2]
    aa = (np.fft.fftfreq(L1) * L1)**2
    pyem_fvol_mask = (np.reshape(aa, [-1,1,1] ) + np.reshape(aa, [1,-1,1] ) + np.reshape(aa, [1,1,-1] ) ) < (L**2)
    
    
    return fvol * pyem_fvol_mask



def ft_np(vol_np, pfac):
    """ OLD  FUNCTION TO REMOVE
    FT of a volume that is approximately compatible with pyEM.
    This one produces pfac* L x pfac*L x pfac*L volume (integer part of this if not integer numbers),
    as opposed to something line pfac*L+3 x pfac*L+3 pfac*L//2+1 in pyEM,
    because pyEM uses rfftn and we use fftn, and becuase of the padding used in pyEM.
    The function implements some filters that are used in pyEM. 
    
    TODO: a version that can be differentiated through. 
    
    input:
    * vol_np : volume (numpy)
    * pfac   : padding factor (1 is no padding).
    
    output:
    * FT of the volume 
    
    """

    #warning: DEPRECATED
    # fft of volume with padding by pfac.
    # Filters compatible with pyEM. However, this uses fftn and not rfftn
    # TODO: rfftn version?
    
    L = vol_np.shape[-1]
    vol_correct_np = vop.grid_correct(vol_np, pfac=pfac, order=1)    # pyEM's correction. TODO: remove
    padvol_np = npo.pad(vol_correct_np, int((L * pfac - L) // 2), "constant")    
    vol_correct_shift_np = npo.fft.ifftshift(padvol_np)
    fvol = npo.fft.fftn( vol_correct_shift_np )
    
    #
    # pyEM filter, unclear purpose. TODO: remove
    #
    L1 = padvol_np.shape[0]
    aa = (np.fft.fftfreq(L1) * L1)**2
    pyem_fvol_mask = (np.reshape(aa, [-1,1,1] ) + np.reshape(aa, [1,-1,1] ) + np.reshape(aa, [1,1,-1] ) ) < (L**2)
    
    
    return fvol * pyem_fvol_mask







