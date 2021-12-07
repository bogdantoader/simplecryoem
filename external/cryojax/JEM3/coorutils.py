from typing import TypeVar
import jax.numpy as jnp
#from jax import vmap
import numpy as onp

from functools import partial


from cryojax.JEM3 import utils as jem3utils
from cryojax.JEM3 import lininterp as jem3lininterp
from cryojax.JEM3 import projutils as jem3projutils







"""

    Notation:
        _u indicates universal coordinates (1/A), 
        _fpx means frequency pixels (i.e., coordinates relative to frequency grid), 
        _rad is angle in radians, 
        f_step is frequency pixel size, 
        _sstep is spatial pixel size

    Typical use for managing coordinates
        # Spatial pixel size of volume:
        vol_sstep = ...
        # Volume frequency pixel size = 1/(size of the real space image including padding)
        # Ater padding, the effective size of the volume is vol_size * pfac * vol_sstep (but we don't want to use pfac directly in case there is some rounding).
        # In the Fourier domain, fvol.shape[-2] = pfac * vol_sstep  (rounded). We are assuming here that all the dimensions of the volume are equal. fvol here is the fft of the padded volume.
        vol_fstep =  1/(vol_sstep * fv0.shape[-2])

        # Spatial pixel size of the image
        # Image pixel size. In principal, can be different for each image, but we usually assume the same for all images batched together.
        image_sstep = .... (often: = vol_sstep)
        # Precompute image points. Given in 1/A.
        # Produces several different representations of the coordinates: 3d coordinates (1/A), 2d coordinates (1/A), radial distance (1/A) and angle (rad). Also gives the frequency step for images. 
        # There is redundancy here, different variables are used in different functions. We're avoiding recomputing them internally so we can manage things better.
        pts3_u, pts2_u, pts_s_u, pts_rad, image_fstep =  get_image_points(L, image_sstep)

        # Scale image points for volume. We can have multiple volumes superimposed, each with its own vol_fpx, so we seperate this from the point precomputation
        pts3_fpx = scale_image_points_to_volume_for_interpolation( pts3_u, vol_fstep )


        We would then often take subsets of the points, and we have to make sure we do this for all the different versions of the points. 
     

"""



################## Coordinates #####################



#
#
#
def fftfreq_points_2d_full(size , isincludez = True , isintpoints = True):
    """  Produce a 2d grid of points

    Parameters 
    ----------
    size  : integer 
            image dimensions are size x size (even number by PyEM convension)
    isincludez : boolean  
            should the output be 3d coodinates (with z set to 0)?
    isintpoint : boolean : 
            if True, the convention is points at integer frequencies (approximately integer points, the values is still a float number).
            if  False, leaves in original format, where the points are in the 
                range [-0.5,0.5] (possibly not at ends of the intervals)
    
    Returns
    -------
        xyz: size x size x 3 array of 2d or 3d (if isxyz) coordinates.
             last axis is (x,y,z) coordinates.
        x  : size x size array of x coordinates
        y  : same for y
    
    type of return variables is default jax real float type.
    """

    # x axis and y axis
    #ax = np.arange(size//2+1)
    ax = jnp.fft.fftfreq(size)
    ay = jnp.fft.fftfreq(size) 
    if isintpoints:
        ax = ax * size        
        ay = ay * size    # TODO: check is this hack is still needed
    # hack for pyem coordinates:
    #ay0 = np.arange(size)
    #ay  = ay0 - size*(ay0> size//2+0.01 )
  
    # 2d coordinates
    x,y = jnp.meshgrid(ax,ay, indexing='xy')
    if isincludez:
        # stack 3d coodrinates
        xyz = jnp.stack((x, y, 0.0*x) , axis=-1)
        #print(xyz)
        #print(size,ay)
        return xyz,x,y
    else:
        #only 2d coordinates
        xyz = jnp.stack((x, y) , axis=-1)
        #print(xyz)
        #print(size,ay)
        return xyz,x,y

    

def points_to_polar(pts):
    """
    Translate 3-d coordinates to polar coordinates (z is ignored).
    
    parameters
    ----------
    pts : array m_1 x m_2 x ... x m_N x 3
          3-d coordinates. z assumed to be 0
    
    returns
    -------
    pts_s : array array m_1 x m_2 x ... x m_N
            radius component of polar.
    pts_a : array array m_1 x m_2 x ... x m_N
            angle component of polar representation (rad).
    
    """

    #
    # polar coordinates, assuming z=0
    # 
    #tmpx,tmpy,tmpz = onp.split(pts, [1,2], axis=-1)
    tmpx,tmpy,tmpz = pts.split( [1,2], axis=-1)
    tmpx = tmpx.reshape( tmpx.shape[:-1] )
    tmpy = tmpy.reshape( tmpy.shape[:-1] )    
    pts_a = jnp.arctan2( tmpy, tmpx )
    
    #pts_s = onp.sqrt((pts**2).sum(-1))
    pts_s = onp.sqrt(tmpx**2 + tmpy**2)
    
    return pts_s, pts_a
    
def get_image_fstep(L, pixelsize):
    imagefreqstep = 1.0 / (L * pixelsize)
    return imagefreqstep

def get_image_points(L, pixelsize = 1.0, isintegers = False, isflatten = False):
    """
    Produces a grid of 2-D image coordinates (Fourier domain)
    
    Parameters
    ----------
    L : dimension of the image.
    pixel size : size of each pixel in A.
    isintegers : boolean (defualt False)
         should the coordinates be scaled but the frequency step size.
    isflatten : flatten 2-d gris to 1-d
    
    Returns
    -------
    pts3  : array L x L x 3 
            3-D coordinates of each grid point (z=0)
    pts2  : array L x L x 2
    pts_s : array L x L
            radius component of polar coordinates
    pts_a : array L x L
            angular component of polar coordinates (rad)
    imagefreqstep : real number
            The spacing between grid points (1/A)
    
    if isintegers == True, the coordinates are integer numbers (type is real, but value approximately integral).
    if isintegers == Flase (default), the coordinates are scaled by imgfreqstep 

    """

    
    
    pts3,_,_  = fftfreq_points_2d_full(L , isincludez = True , isintpoints = True)
    pts_s, pts_a = points_to_polar(pts3)
    pts2 = pts3[:,:,:2]

    imagefreqstep = get_image_fstep(L, pixelsize)
    if ~isintegers: # scale the pixels
        pts3 = pts3 * imagefreqstep
        pts2 = pts2 * imagefreqstep
        pts_s = pts_s * imagefreqstep

    if isflatten:
        pts3 = pts3.reshape(-1,3)
        pts2 = pts2.reshape(-1,3)
        pts_s = pts_s.reshape(-1)
        pts_a = pts_a.reshape(-1)
        
    return pts3, pts2, pts_s, pts_a,  imagefreqstep



def scale_image_points_to_volume_for_interpolation( pts, volume_step ):
    """
    This function should only be applied to coordinates that are going to be passed to linear interpolation.
    It rescales the coordinates from the natural 1/A to the volume's natural units, which can be different for different volumes.
    
    """

    #assert( len(volume_step) == 1 ) # volume step can technically be a vector, but this need to be treated with greater care and restricts the processing, so not supported here.

    pts = pts / volume_step

    return pts



#############################################################
### For the 3d volume
#############################################################
    
#
#
#
def fftfreq_points_3d_full(size , isintpoints = True):
    """  Produce a 3d grid of points
    
    Uses numpy and not jax.numpy.

    Parameters 
    ----------
    size  : integer 
            image dimensions are size x size (even number by PyEM convension)
    isintpoint : boolean : 
            if True, the convention is points at integer frequencies (approximately integer points, the values is still a float number).
            if  False, leaves in original format, where the points are in the 
                range [-0.5,0.5] (possibly not at ends of the intervals)
    
    Returns
    -------
        xyz: size x size x size x 3 array of 3d coordinates.
             last axis is (x,y,z) coordinates.
    
    type of return variables is default jax real float type.
    """

    # x axis and y axis
    #ax = np.arange(size//2+1)
    ax = onp.fft.fftfreq(size)
    ay = onp.fft.fftfreq(size) 
    az = onp.fft.fftfreq(size) 
    if isintpoints:
        ax = ax * size               
        ay = ay * size
        az = az * size
  
    # 3d coordinates
    z,y,x = onp.meshgrid(az,ay,ax, indexing='ij')
    #print(x)
    #print("-")
    #print(y)
    #print("-")
    #print(z)
    # stack 3d coodrinates
    xyz = onp.stack((x, y, z  ) , axis=-1)
    #print(xyz)
    #print(size,ay)
    #print("-")
    return xyz





def get_vol_s_points_mod(L, volfreqstep, isintegers = False, isflatten = False):
    """
    Produces a grid of 3-D image coordinates (Fourier domain)
    
    Parameters
    ----------
    L : dimension of the image.
    volfreqstep
    isintegers : boolean (defualt False)
         should the coordinates be scaled but the frequency step size.
    isflatten : flatten 3-d grid to 1-d
    
    Returns
    -------
    pts3
    pts_s : array L x L x L
            radius component of polar coordinates
    
    if isintegers == True, the coordinates are integer numbers (type is real, but value approximately integral).
    if isintegers == Flase (default), the coordinates are scaled by imgfreqstep 

    """
    
    pts3  = fftfreq_points_3d_full(L, isintpoints = True)
    pts_s = (pts3**2).sum(-1)

    #volfreqstep = 1.0 / (pixelsize * L)
    if ~isintegers: # scale the pixels
        pts3 = pts3 * volfreqstep
        pts_s = pts_s * volfreqstep

    if isflatten:
        pts3 = pts3.reshape(-1,3)
        pts_s = pts_s.reshape(-1)
        
    return pts3, pts_s



###############################################################################

if __name__ == '__main__':
    print(" === CoorUtils ===")

    print( fftfreq_points_2d_full(3 , isintpoints = True)[0] )
    print("---")
    print( fftfreq_points_3d_full(3 , isintpoints = True)    )
