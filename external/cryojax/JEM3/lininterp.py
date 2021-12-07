from typing import TypeVar
import jax.numpy as jnp
#from jax import vmap
import numpy as onp
from functools import partial

#import jax.lax
from jax import lax

from cryojax.JEM3 import utils as jem3utils


""" 
Public functions:

* linear_interp_3d
* rotate_and_project

"""



########## Interpolation tools ####################3

def linear_interp_tool_coor(x: "jnp.array(real)", L: int):
    """ Helper function for linear_interp_{} functions.
    """

    # add dimention for multiple points
    # x = jnp.reshape(x, x.shape+(1,))
    #print(x.shape)
    # corner identities
    xptid = jnp.array([0, 1])
    #
    xp = jnp.floor(x)+xptid
    #
    #xwgt = 1.0-jnp.abs(x-xp)
    xwgt = 1.0-jnp.abs(x-xp)
    #
    xp = (xp%L).astype(jnp.int32)
    #xp = (xp).astype(jnp.int32)
    return xp, xwgt, xptid
#
#
#
def linear_interp_3d(f:jnp.ndarray, psa: jnp.ndarray) ->jnp.ndarray:#, Ls: "tuple(int)"):
    """ Linear interpolation, for 3D case.
    
    Parameters
    ----------
    f  : array of complex numbers
         3d grid representing a volume
    psa : array of real numbers
         each array of size N_1 x ... x N_n x 3 (or broadcasts to this shape)
         first array is x coordinate, second is y coordinate, third is z coordinate
    
    Returns
    -------
    vv : array of complex numbers 
         shape: N_1 x ... x N_n
    
    Evaluates the trilinear interpolation of f at certain coordinates.
    f is treated as if it were periodics, so that if the coordinates are out of bounds, they are evaluated modulo the size of f in each axis.
    
    
    Comments:
    * Note that axes are expected to be in the order x,y,z (coordinates order), but the axis order of f is reversed.

    """
    
    assert( psa.shape[-1] == 3 )
    ps = psa.split(psa.shape[-1],axis=-1)
    
    
    #print(f.shape)
    zp, zwgt, zptid = linear_interp_tool_coor(ps[2], f.shape[-3])
    yp, ywgt, yptid = linear_interp_tool_coor(ps[1], f.shape[-2])
    xp, xwgt, xptid = linear_interp_tool_coor(ps[0], f.shape[-1])
    
    #print(xp.shape)
    #print("px:",ps[0], f.shape[-1])
    #print("new: ",xp, zp )
    #print("new: ",xwgt ,zwgt )
    
    xp   = jnp.expand_dims(xp,   axis=(-2,-3))
    xwgt = jnp.expand_dims(xwgt, axis=(-2,-3))
    yp   = jnp.expand_dims(yp,   axis=(-1,-3))
    ywgt = jnp.expand_dims(ywgt, axis=(-1,-3))
    zp   = jnp.expand_dims(zp,   axis=(-1,-2))
    zwgt = jnp.expand_dims(zwgt, axis=(-1,-2))

    # double precision complex numbers require special handling here due to XLA bug 
    #vv = lax.cond(f.dtype == jnp.complex128,
    #                  ((xwgt*ywgt*zwgt)*(jnp.real(f)[zp,yp,xp]+1j*jnp.imag(f)[zp,yp,xp])).sum(axis=(-3,-2,-1)),
    #                  ((xwgt*ywgt*zwgt)*f[zp,yp,xp]).sum(axis=(-3,-2,-1)) ,
    #              operand=None)
    
    if f.dtype == jnp.complex128: # due to Jax/ XLA bug. 
        vv = ((xwgt*ywgt*zwgt)*(jnp.real(f)[zp,yp,xp]+1j*jnp.imag(f)[zp,yp,xp])).sum(axis=(-3,-2,-1))
        #vv = ((xwgt*ywgt*zwgt)*(f.real[zp,yp,xp]+1j*f.imag[zp,yp,xp])).sum(axis=(-3,-2,-1))
        # todo: check if the reason is type of variable sent back (is it complex 64 not complex 128?) - i.e., if this is caused by combining promotion and scatter.
    else:
        vv = ((xwgt*ywgt*zwgt)*f[zp,yp,xp]).sum(axis=(-3,-2,-1))
        
    return vv




def rotate_and_project( fv, r, pts_px ):
    """ Rotate and project
    
    Parameters
    ----------
    fv      : 3d array, complex numbers
              grid representation of the volume in the Fourier domain
              
    r       : 3 x 3 matrix
              rotation matrix
    
    pts_px  : N_1 x ... x N_n x 3 array
              coordinates for volume evaluation
              Note: 
              * The convention for coordinates is (x,y,z)
              * The convention for the arrays is (z,y,x)
              * The scale of the points is in pixels (frequency pixels), not angrtom or 1/angstrom.
              In other word, pts_px = [1,0,0] would produce the point fv[0,0,1].

    Returns
    -------
    v       : N_1 x ... x N_n array
              fv interpolated to the points pts_px
    
    """

    pts_px_rot = jem3utils.rot_coor(r.T, pts_px )
    v = linear_interp_3d(fv, pts_px_rot)
    #v = linear_interp_3d_v2(fv, pts_px_rot)  # the original version appears to be much faster than the autovectorized version when @jax.jit used. It is about 2x slower, but much faster for complex double when grad is needed. Seems to require more RAM for double.
    
    return v



#########################################
####    Dev versions - do not use    ####
#########################################


def rotate_and_project_dev( fv, r, pts_px ):
    """ Rotate and project
    
    Parameters
    ----------
    fv      : 3d array, complex numbers
              grid representation of the volume in the Fourier domain
              
    r       : 3 x 3 matrix
              rotation matrix
    
    pts_px  : N_1 x ... x N_n x 3 array
              coordinates for volume evaluation
              Note: 
              * The convention for coordinates is (x,y,z)
              * The convention for the arrays is (z,y,x)
              * The scale of the points is in pixels (frequency pixels), not angrtom or 1/angstrom.
              In other word, pts_px = [1,0,0] would produce the point fv[0,0,1].

    Returns
    -------
    v       : N_1 x ... x N_n array
              fv interpolated to the points pts_px
    
    """

    pts_px_rot = jem3utils.rot_coor(r.T, pts_px )
    v = slice_extractor(fv, pts_px_rot)
    #v = linear_interp_3d_v2(fv, pts_px_rot)  # the original version appears to be much faster than the autovectorized version when @jax.jit used. It is about 2x slower, but much faster for complex double when grad is needed. Seems to require more RAM for double.
    
    return v

def indexer(volume: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Given a three-dimensional of volume of size `D x D x D` and a list of
    indices of size `B x D x D x 3`, construct a matrix of size `B x D x D`
    whose entries are computed by indexing the volume at the desired indices
    in the last dimension of the index variable.

    This implementation is a modified version of the techniques in [1] but
    using a different indexing procedure. [1] was provided under a GPL-3.0
    License.

    [1] https://github.com/KarenUllrich/Pytorch-Backprojection

    Args:
        volume: Array of complex numbers representing the three-dimensional
            volume.
        idx: Array of indices at which to index the volume.

    Returns:
        y: The indexed volume at the desired integer coordinates.

    """
    limit = volume.shape[0]
    #idx = idx % (limit - 1)  ######### TO CHECK: why limit-1
    idx = idx % (limit )      ############################ CAREFUL USING %?
    y = volume[(*idx.transpose((3, 0, 1, 2)), )]
    return y

def slice_extractor(volume: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    """Extract a slice of the three-dimensional Fourier transform by
    interpolating between the grid frequencies.

    Args:
        volume: Array of complex numbers representing the three-dimensional
            volume.
        grid: Grid of frequencies at which to compute the interpolation of the
            three-dimensional Fourier transform of the volume.

    Returns:
        c: The interpolated slice through the Fourier transform of the volume.

    """

    #print(grid.shape)
    grid = grid.reshape( (1,)+grid.shape )
    
    # Extract the x-, y-, and z-coordinates of the grid.
    ix = grid[..., 0]
    iy = grid[..., 1]
    iz = grid[..., 2]

    # Get the nearest integer frequencies above and below the grid.
    px_0 = jnp.floor(ix).astype(jnp.int32)
    py_0 = jnp.floor(iy).astype(jnp.int32)
    pz_0 = jnp.floor(iz).astype(jnp.int32)
    px_1 = jnp.ceil(ix).astype(jnp.int32)
    py_1 = jnp.ceil(iy).astype(jnp.int32)
    pz_1 = jnp.ceil(iz).astype(jnp.int32)

    dx = ix - px_0
    dy = iy - py_0
    dz = iz - pz_0
    #print(dx.shape)

    c_000 = indexer(volume, jnp.stack([pz_0, py_0, px_0], axis=-1))
    c_100 = indexer(volume, jnp.stack([pz_0, py_0, px_1], axis=-1))
    c_00 = c_000 * (1. - dx) + c_100 * (dx)
    del c_000, c_100

    c_010 = indexer(volume, jnp.stack([pz_0, py_1, px_0], axis=-1))
    c_110 = indexer(volume, jnp.stack([pz_0, py_1, px_1], axis=-1))
    c_10 = c_010 * (1. - dx) + c_110 * (dx)
    del c_010, c_110

    c_0 = c_00 * (1. - dy) + c_10 * (dy)
    del c_00, c_10

    c_001 = indexer(volume, jnp.stack([pz_1, py_0, px_0], axis=-1))
    c_101 = indexer(volume, jnp.stack([pz_1, py_0, px_1], axis=-1))
    c_01 = c_001 * (1. - dx) + c_101 * (dx)
    del c_001, c_101

    c_011 = indexer(volume, jnp.stack([pz_1, py_1, px_0], axis=-1))
    c_111 = indexer(volume, jnp.stack([pz_1, py_1, px_1], axis=-1))
    c_11 = c_011 * (1. - dx) + c_111 * (dx)
    del c_011, c_111

    c_1 = c_01 * (1. - dy) + c_11 * (dy)
    del c_11, c_01

    c = c_0 * (1. - dz) + c_1 * (dz)
    return c



#
# interpolate using jax vectorize
#


def interp_mod(x,L):
    """ x mod L structured as helper function for linear interpolation code
    
    """
    return (x%L).astype(jnp.int32)


#
# Linear interpolation for one 
#
@partial(jnp.vectorize, signature='(a,b,c),(3)->()')
def linear_interp_3d_v2( f, p ):
    """ Linear interpolation, for 3D case.
    Should be functionally identical to linear_interp_3d, but this is defined for a single point, then vectorized using jax.vectorize.
    This version seem to be very wasteful (time and cmpute).
    
    """
    
    pint = jnp.floor(p)
    pdf  = jnp.abs(p-pint)
    #print(pint)
    #print(pdf)
    
    vv =    (
          (1.0-pdf[0])*(1.0-pdf[1])*(1.0-pdf[2]) * f[ interp_mod(pint[2]  ,f.shape[-3])  ,interp_mod(pint[1]  ,f.shape[-2])  , interp_mod(pint[0]  ,f.shape[-1])  ] +       
          (    pdf[0])*(1.0-pdf[1])*(1.0-pdf[2]) * f[ interp_mod(pint[2]  ,f.shape[-3])  ,interp_mod(pint[1]  ,f.shape[-2])  , interp_mod(pint[0]+1,f.shape[-1])  ] +
          (1.0-pdf[0])*(    pdf[1])*(1.0-pdf[2]) * f[ interp_mod(pint[2]  ,f.shape[-3])  ,interp_mod(pint[1]+1,f.shape[-2])  , interp_mod(pint[0]  ,f.shape[-1])  ] +       
          (    pdf[0])*(    pdf[1])*(1.0-pdf[2]) * f[ interp_mod(pint[2]  ,f.shape[-3])  ,interp_mod(pint[1]+1,f.shape[-2])  , interp_mod(pint[0]+1,f.shape[-1])  ] +
          (1.0-pdf[0])*(1.0-pdf[1])*(    pdf[2]) * f[ interp_mod(pint[2]+1,f.shape[-3])  ,interp_mod(pint[1]  ,f.shape[-2])  , interp_mod(pint[0]  ,f.shape[-1])  ] +       
          (    pdf[0])*(1.0-pdf[1])*(    pdf[2]) * f[ interp_mod(pint[2]+1,f.shape[-3])  ,interp_mod(pint[1]  ,f.shape[-2])  , interp_mod(pint[0]+1,f.shape[-1])  ] +
          (1.0-pdf[0])*(    pdf[1])*(    pdf[2]) * f[ interp_mod(pint[2]+1,f.shape[-3])  ,interp_mod(pint[1]+1,f.shape[-2])  , interp_mod(pint[0]  ,f.shape[-1])  ] +       
          (    pdf[0])*(    pdf[1])*(    pdf[2]) * f[ interp_mod(pint[2]+1,f.shape[-3])  ,interp_mod(pint[1]+1,f.shape[-2])  , interp_mod(pint[0]+1,f.shape[-1])  ]   ) 

    #print(vv)
    return vv



def linear_interp_OLD_tool_coor(x: "jnp.array(real)", L: int):
    """ Helper function for linear_interp_{} functions.
    """

    # add dimention for multiple points
    x = jnp.reshape(x, x.shape+(1,))
    #print(x.shape)
    # corner identities
    xptid = jnp.array([0, 1])
    #
    xp = jnp.floor(x)+xptid
    #
    #xwgt = 1.0-jnp.abs(x-xp)
    xwgt = 1.0-jnp.abs(x-xp)
    #
    xp = (xp%L).astype(jnp.int32)
    return xp, xwgt, xptid

#
#
#
def linear_interp_3d_old(f:"jnp.array(Ls[2],Ls[1],Ls[0])", ps: "(jnp.array(real),jnp.array(real),jnp.array(real))"):#, Ls: "tuple(int)"):
    """ Linear interpolation, for 3D case.
    
    Parameters
    ----------
    f  : array of complex numbers
         3d grid representing a volume
    ps : list of 3 arrays of real numbers
         each array of size N_1 x ... x N_n  (or broadcasts to this shape)
         first array is x coordinate, second is y coordinate, third is z coordinate
    
    Returns
    -------
    vv : array of complex numbers 
         shape: N_1 x ... x N_n
    
    Evaluates the trilinear interpolation of f at certain coordinates.
    f is treated as if it were periodics, so that if the coordinates are out of bounds, they are evaluated modulo the size of f in each axis.
    
    
    Comments:
    * Note that axes are expected to be in the order x,y,z (coordinates order), but the axis order of f is reversed.

    """
    
    #print(f.shape)
    zp, zwgt, zptid = linear_interp_OLD_tool_coor(ps[2], f.shape[-3])
    yp, ywgt, yptid = linear_interp_OLD_tool_coor(ps[1], f.shape[-2])
    xp, xwgt, xptid = linear_interp_OLD_tool_coor(ps[0], f.shape[-1])
    
    #print(xp.shape)
    #print("px:",ps[0], f.shape[-1])
    #print("new: ",xp, zp )
    #print("new: ",xwgt ,zwgt )
    
    xp   = jnp.expand_dims(xp,   axis=(-2,-3))
    xwgt = jnp.expand_dims(xwgt, axis=(-2,-3))
    yp   = jnp.expand_dims(yp,   axis=(-1,-3))
    ywgt = jnp.expand_dims(ywgt, axis=(-1,-3))
    zp   = jnp.expand_dims(zp,   axis=(-1,-2))
    zwgt = jnp.expand_dims(zwgt, axis=(-1,-2))

    vv = ((xwgt*ywgt*zwgt)*f[zp,yp,xp]).sum(axis=(-3,-2,-1))
    return vv


    






