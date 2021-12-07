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



###########################################################
#### Cube down/up sampling
###########################################################


def get_subgrid3d(v0, new_L):
    """ Extract subgrid of 3d grid. The subset is a built to minic a low pass filter in the Fourier domain.
    
    Parameters
    ----------
    v0    : L x L x L array (numpy or jax)
    new_L : integer. dimension of new array. new_L <= L.

    Returns
    -------
    vsub  : new_L x new_L x new_L array of same type as v0
    
    For example, L=4, for k0,k1,k2 in [0,1,-2,-1] we have vsub[k0,k1,k2] = v0[k0,k1,k2].

    """
    assert( len(v0.shape) == 3 )
    assert( v0.shape[0]== v0.shape[1] )
    assert( v0.shape[1]== v0.shape[2] )
    assert( new_L <= v0.shape[0] )
    ax = onp.round(onp.fft.fftfreq(new_L)*new_L).astype(int)
    xx = ax
    yy = xx[...,onp.newaxis]
    zz = yy[...,onp.newaxis]
    #print(xx,yy,zz)
    #print(ax)
    #print(v0[ax,ax,ax])
    #print(v0[xx,yy,zz])
    #print(v0[ax,:,:][:,ax,:][:,:,ax])
    #print(v0[ax,][ax,][ax,])
    #print(v0)
    #return v0[xx,yy,zz]
    return v0[zz,yy,xx]


def put_subgrid3d(v0,vsub):
    """ Updates a subcube of a 3d array. "Inverse" of getsubgrid3d.

    Parameters
    ----------
    v0    : L x L x L array (numpy or jax)
    vsub  : sub_L x sub_L x sun_L array of same type as v0

    Returns
    -------
    v     : L x L x L array of same type as v0

    For example, L=4 would return entries [0,1,-2,-1] of the first axis, [0,1,-2,-1] of the second axis and [0,1,-2,-1] of the third axis of v0.
    """
    assert( len(v0.shape) == 3 )
    assert( v0.shape[0]== v0.shape[1] )
    assert( v0.shape[1]== v0.shape[2] )
    assert( len(vsub.shape) == 3 )
    assert( vsub.shape[0]== vsub.shape[1] )
    assert( vsub.shape[1]== vsub.shape[2] )
    assert( vsub.shape[0] <= v0.shape[0] )
    
    sub_L = vsub.shape[-2]
    ax = onp.round(onp.fft.fftfreq(sub_L)*sub_L).astype(int)
    xx = ax
    yy = xx[...,onp.newaxis]
    zz = yy[...,onp.newaxis]
    #print(vsub)
    v=v0
    if isinstance(v,onp.ndarray):
        v = v.copy()
        #v[xx,yy,zz] = vsub
        v[zz,yy,xx] = vsub
    else:
        #v=jax.ops.index_update(v,(xx,yy,zz),vsub)
        v=jax.ops.index_update(v,(zz,yy,xx),vsub)
    return v



############################################################
#### Radial subsets tools
############################################################


def get_px_radius(L:int) ->int :
    """ For a grid of length L (in FFT convention), find the distance from 0 to the nearest end.
    
    Parameters
    ----------
    L : Integer
    
    Returns
    -------
    Integer
    
    When L is and odd number this is (L-1)/2. When L is an even number it is L//2-1.
    For example, for L=4, the grid is at point [0,1,-2,1] and the radius is 4/2-1 = 1.
    """
    if L%2 == 0:
        return ((L//2)-1)
    else:
        return ((L-1)//2)
    #return round((L-1.01)/2)
    

def get_u_radius(L:int,u:float)->float:
    return get_px_radius(L)*u


def get_rad_mask(p:jnp.ndarray, r:float)-> jnp.ndarray:
    """ mask points up to radius r
    
    parameters
    ----------
    p : array of dim n_1 x ... n_N x d of coordinate in R^d. jax or numpy
    r : float. max radius
    
    returns
    -------
    v : array of dim n_1 x ... n_N jax or numpy. binary.
    
    NOTE: it's best to take r+eps or r-eps if there is a reasonable chance that the distance for some point would be exactly r
    """

    return (p**2).sum(-1) <= (r**2)


def get_array_subset(a: jnp.ndarray, msk: jnp.ndarray) -> jnp.ndarray:
    """ 
    
    parameters
    ----------
    a   : jax or numpy array of dim n_1 x ... n_N x d of coordinate in R^d 
          or dim n_1 x ... n_N 
    msk : jax or numpy binary array of dim n_1 x ... n_N. 
    
    returns
    -------
    v : array of dim n_1 \cdot ...\cdot n_N or n_1 \cdot ...\cdot n_N x d
    
    returns only the entries of a that are selected by the mask. The outpot is flattened. 
    
    """

    #print(a.shape, msk.shape, msk.shape[:len(a.shape)])
    if  (a.shape[:len(msk.shape)]!=msk.shape):
        raise Exception("get_array_subset","dimensions mismatch")
    
    if len(a.shape)==len(msk.shape):
        return a.reshape(-1)[msk.reshape(-1)]
    if len(a.shape)== (len(msk.shape)+1):
        return a.reshape(-1,a.shape[-1])[msk.reshape(-1)]

    raise Exception ("get_array_subset","shapes mismatch")
    return None #Error!



def get_array_subset_npvectorize(a: onp.ndarray, msk: onp.ndarray) -> onp.ndarray:
    """ ++++++++++++++++++ BROKEN ++++++++++++++++++++
    
    parameters
    ----------
    a   : jax or numpy array of dim nn x n_1 x ... n_N x d of coordinate in R^d 
          or dim nn x n_1 x ... n_N 
    msk : jax or numpy binary array of dim n_1 x ... n_N. 
    
    returns
    -------
    v : array of dim nn x n_1 \cdot ...\cdot n_N or nn x n_1 \cdot ...\cdot n_N x d
    
    returns only the entries of a that are selected by the mask. The outpot is flattened. 
    
    """

    #print(a.shape, msk.shape, msk.shape[:len(a.shape)])
    if  (a[0].shape[:len(msk.shape)]!=msk.shape):
        raise Exception("get_array_subset","dimensions mismatch")
    
    return onp.vstack([get_array_subset(a[j], msk) for j in range(a.shape[0])] )

    #raise Exception ("get_array_subset","shapes mismatch")
    #return None #Error!




vmap_get_array_subset = vmap(get_array_subset,(0,None),0)
""" get_array_subset applied to array of array """





############################################################
### mask loss
############################################################




############################################################
#### Tests
############################################################


def test020():
    print(get_px_radius(5))
    print(get_px_radius(4))
    print(get_px_radius(3))
    x = get_subgrid3d( onp.arange(4**3).reshape(4,4,4), 2)
    print(type(x))
    print(x)
    y=put_subgrid3d( onp.zeros((4,4,4)), x)
    print(type(y))
    print(y)

    p = onp.random.randn(2,3,3)
    r = 2.0

    print(p)
    msk = get_rad_mask(p, r)
    print(msk)

    q = get_array_subset(p, msk)
    print(q)
    
    return


def test021():
    onp.random.seed(123)
    v = onp.random.randn(6,6,6)
    #v = onp.zeros([6,6,6])
    v[0,0,0]=2
    v[0,0,1]=1
    v[0,0,-1]=-1
    v2= get_subgrid3d(v,4 )
    print(v.shape,v2.shape)
    print(v[0,0,1],v2[0,0,1])
    print(v[0,0,-1],v2[0,0,-1])
    print(v2)

    vv=copy.deepcopy(v)
    vvjax = jnp.array(copy.deepcopy(v))

    vvA = put_subgrid3d(vv,v2)
    vvAjax = put_subgrid3d(vvjax,jnp.array(v2))

    print( jnp.linalg.norm(vvA-vvAjax), jnp.linalg.norm(v-vvAjax) )
    
if __name__ == '__main__':
    import copy
    from  matplotlib import pyplot as plt
    #import mrcfile
    #test020()
    test021()

