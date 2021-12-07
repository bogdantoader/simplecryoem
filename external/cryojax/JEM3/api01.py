from typing import TypeVar

from functools import partial
import copy

import numpy as onp

import jax
import jax.numpy as jnp
from jax import vmap
from jax import jit
from jax import tree_util

from external.cryojax.JEM3 import utils as jem3utils
from external.cryojax.JEM3 import lininterp as jem3lininterp
from external.cryojax.JEM3 import projutils as jem3projutils
from external.cryojax.JEM3 import coorutils as jem3coorutils
from external.cryojax.JEM3 import emfiles as emfiles3
from external.cryojax.JEM3 import ctf as ctf3
from external.cryojax.JEM3 import wraputils as jem3wrap
from external.cryojax.JEM3 import noiseutils as jem3noiseutils
from external.cryojax.JEM3 import filterutils as jem3filterutils


#legacy
import cryojax.projectPyEM2.jaxvop2 as jvop2

#import numpy as onp
import sys
import time
import os



#########################
# TODO: remove calls to legacy code
#########################



class raw_images_class():
    """ Images object
    
    Variables
    ---------
    self.images_fft  : fft of images. 
    self.images_var  : image parameters
            r : orientations
            a : Euler angles
            organgst : shift
            ctf : ctf parameters
    self.image_L_px  : raw image dimensions
    self.image_sstep : Image pixel size. In principal, can be different for each image, but we usually assume the same for all images batched together.

    self.image_fstep : Image fourier pixel size
    self.image_points : Describes the coordinates of points in the image
        
    self.n : number of images
    self.indices : indices read from file
    self.misc
    
    self.raw_image_f_radius_u : the distance from the 0 frequency to the nearest end of the image, in the Fourier domain. For the raw images.
    self.max_radius_u : Radius after crop
        
    self.proc   : log of processing
    self.source : what kind of processing do we see currently

    Methods
    -------
    starloader : load data
    
    """
    #
    # Todo: check what is a jnp and what is an onp. Consider changing variables to jnp.
    #
    def __init__(self):
        self.images_fft = None
        self.images_var = None
        self.image_L_px = None
        # Image pixel size. In principal, can be different for each image, but we usually assume the same for all images batched together.
        self.image_sstep = None

        self.image_fstep = None
        self.image_points = None
        
        self.n = None
        self.indices = None
        self.misc = {}

        self.raw_image_f_radius_u = None # the distance from the 0 frequency to the nearest end of the image, in the Fourier domain.
        
        self.max_radius_u = None
        
        self.proc = []
        self.source = None
        self.masks = []
        
        return
    
    
    def starloader(self, files_base, files_star_in, threads= 2):
        images_fft, index_array, grp_array, orient_array, angles_array, organgst_array, orgpx_array, apiximg_array, ctf_array, df, gb = emfiles3.read_star_mrcs(files_base,files_star_in, loadimages = True, returnungroup = True, threads=4)
        self.images_fft = images_fft

        #orient_array = jax.device_put(orient_array)
        #organgst_array = jax.device_put(organgst_array)
        #apiximg_array = jax.device_put(apiximg_array) 
        #ctf_array = jax.device_put(ctf_array)

        self.indices = {
            "index_array": index_array,
            "grp_array": grp_array
            }
        self.images_var = {
            "r":orient_array,
            "a":angles_array,
            "organgst":organgst_array,
            #orgpx_array,
            "ctf":ctf_array
            }
        self.n = images_fft.shape[0]
        self.misc["apiximg_array"] = apiximg_array

        self.image_L_px = images_fft.shape[-2]
        # Image pixel size. In principal, can be different for each image, but we usually assume the same for all images batched together.
        # TODO: check that it's constant
        self.image_sstep = apiximg_array[0]

        self.image_fstep = jem3coorutils.get_image_fstep(self.image_L_px, self.image_sstep)

        full_pts3_u, full_pts2_u, full_pts_s_u, full_pts_rad, image_fstep =  jem3coorutils.get_image_points(self.image_L_px, self.image_sstep)
        self.image_points = {
            "pts3_u": full_pts3_u, # 3d coordinates of each grid point
            "pts2_u": full_pts2_u, # 2d coordinates of each grid point
            "pts_s_u": full_pts_s_u, # radius of each grid point (polar)
            "pts_rad": full_pts_rad, # angle of each grid point (polar)
            "pts_id": onp.arange(images_fft.shape[-2]*images_fft.shape[-1]).reshape(images_fft.shape[-2],images_fft.shape[-1]),
            "format" : "Grid"
            }

        self.raw_image_f_radius_u = jem3filterutils.get_u_radius(self.image_L_px, image_fstep)
        print("raw_images_class: starloader: full frequency radius (a.k.a. Nyquist):",self.raw_image_f_radius_u)

        # self.max_radius_u =  #### TODO: Add this
        
        self.source = "starloader"
        self.status = "raw"
        self.proc = self.proc + ["starloader",]

    def data_cropper_radial_(self, radius_u):    
        #
        # Downsample. Choose only the pixels that are in a desired frequency radius
        #

        # slightly smaller ball to avoid numerical artifacts

        print("raw_images_class: data_cropper_radial: radius:",radius_u,radius_u/self.image_fstep)
        
        self.max_radius_u = radius_u
        
        run_image_mask = jem3filterutils.get_rad_mask(self.image_points["pts2_u"], radius_u)
        #print(type(run_image_mask))
        run_image_mask = onp.asarray(run_image_mask) # make mask into numpy mask
        #print(type(run_image_mask))

        self.masks.append(run_image_mask)
        #print(self.image_points["pts3_u"].shape,self.image_points["pts2_u"].shape,self.image_points["pts_s_u"].shape,self.image_points["pts_rad"].shape,self.image_points["pts_id"].shape,self.images_fft.shape,run_image_mask.shape  )
        #print(self.image_points["pts3_u"][0,0],self.image_points["pts2_u"][0,0],self.image_points["pts_s_u"][0,0],self.image_points["pts_rad"][0,0].shape,self.image_points["pts_id"][0,0],self.images_fft[0,0,0],run_image_mask[0,0]  )
        
        self.image_points["pts3_u"] = jem3filterutils.get_array_subset(self.image_points["pts3_u"], run_image_mask)
        self.image_points["pts2_u"] = jem3filterutils.get_array_subset(self.image_points["pts2_u"], run_image_mask)
        self.image_points["pts_s_u"]  = jem3filterutils.get_array_subset(self.image_points["pts_s_u"], run_image_mask)
        self.image_points["pts_rad"]  = jem3filterutils.get_array_subset(self.image_points["pts_rad"], run_image_mask)
        #pts3_fpx = jem3filterutils.get_array_subset(full_pts3_fpx, run_image_mask)
        self.image_points["pts_id"]  = jem3filterutils.get_array_subset(self.image_points["pts_id"], run_image_mask)

        
        self.source = "data_cropper_radial"
        self.status = "vec"
        self.proc = self.proc + [["data_cropper_radial",radius_u,radius_u/self.image_fstep],]
        
        t32=time.time()
        self.images_fft = jem3filterutils.get_array_subset_npvectorize(self.images_fft,run_image_mask)
        print("WARNING: api01.raw_images_class.data_cropper_radial_ subset selection needs more testing")
        t33=time.time()

        #print(self.image_points["pts3_u"].shape,self.image_points["pts2_u"].shape,self.image_points["pts_s_u"].shape,self.image_points["pts_rad"].shape,self.image_points["pts_id"].shape,self.images_fft.shape,run_image_mask.shape  )
        #print(self.image_points["pts3_u"][0],self.image_points["pts2_u"][0],self.image_points["pts_s_u"][0],self.image_points["pts_rad"][0].shape,self.image_points["pts_id"][0],self.images_fft[0,0],run_image_mask[0,0]  )
        
        print("raw_images_class: data_cropper_radial: pixels considered:", self.images_fft.shape, type(self.images_fft))


        
class raw_volume():
    """ Volume object
    """
    def __init__(self):
        self.fv = None   # grid of volume in Fourier domain
        self.vol_sstep = None
        self.vol_fstep = None
        self.raw_pfac = None
        self.raw_s_dim = None

    def put_fvol_(self, fv0, vol_sstep):
        assert( fv0.shape[-2]==fv0.shape[-1] )
        assert( fv0.shape[-3]==fv0.shape[-1] )        
        self.vol_sstep = vol_sstep
        self.vol_fstep =  1.0/(vol_sstep * fv0.shape[-2])   #vol_fstep =  1.0/(apix2*L*pfac), but the effective pfac can be inaccurate, so we'll use th
        self.fv=fv0 #jnp.array(fv0)

    def put_fvol_explicit_steps_(self, fv0, fvol_fstep):
        assert( fv0.shape[-2]==fv0.shape[-1] )
        assert( fv0.shape[-3]==fv0.shape[-1] )        
        self.vol_fstep =  fvol_fstep
        self.fv=fv0
        
    def proc_vol_(self, vol, vol_sstep, pfac):
        # TODO: replace with more generic processing
        self.raw_s_dim = vol.shape
        self.raw_pfac = pfac
        fv0 = (jvop2.ft3d_withpyemfilters_np(vol, pfac))
        self.put_fvol_(fv0, vol_sstep)
        return

    def get_fvol_grid_coordinates(self):
        # returns numpy
        full_vol_pts3_u,_ = jem3coorutils.get_vol_s_points_mod(self.fv.shape[-2], self.vol_fstep)
        return full_vol_pts3_u

    def get_fvol_ball_mask(self, run_radius_u):
        # returns numpy
        return jem3filterutils.get_rad_mask(self.get_fvol_grid_coordinates(), run_radius_u )

    def get_subcube(self, radius_u ):
        # you should probably pad this radius externally.
        rad_px = int(onp.ceil(radius_u/self.vol_fstep))
        new_L = 2*rad_px+1
        if new_L < self.fv.shape[-2]:
            new_fv = jem3filterutils.get_subgrid3d(self.fv, new_L)
        else:
            new_fv = copy.copy(self.fv)
        return new_fv

    def put_subcube_(self, sub_fv):
        self.fv = jem3filterutils.put_subgrid3d(self.fv,sub_fv)
        

class ops_generator_class():
    """ Operators object
    """
    def __init__(self,myproject_function,myimgloss_function):
        # myproject( fv,r,organgst,ctfinfo,          _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad )
        # myimgloss( img1, img2, wnsparam , ctfinfo, _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad)
        self.myproject = myproject_function
        self.myimgloss = myimgloss_function
        
        # Core function composition
        self.project_and_loss = (lambda fv,r,organgst,ctfinfo,wnsparam, img2,  _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad:
           self.myimgloss( self.myproject( fv,r,organgst,ctfinfo,_pts3_fpx, _pts2_u, _pts_s_u, _pts_rad ), img2, wnsparam , ctfinfo, _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad))
           #self.myimgloss( self.myproject( fv,r,organgst,ctfinfo,_pts3_fpx, _pts2_u, _pts_s_u, _pts_rad ), jax.device_put(img2), wnsparam , ctfinfo, _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad))
        
        # Wrappers
        self.vmap_myproject = (vmap( self.myproject, (None, 0,0,0, None,None,None,None) , 0))
        self.jit_vmap_myproject = jit( self.vmap_myproject )
        self.vmap_project_and_loss = (vmap(self.project_and_loss, (None, 0,0,0,None,0, None,None,None,None) , 0))
        self.jit_vmap_project_and_loss = jax.jit(self.vmap_project_and_loss) #, static_argnums=(6,7,8,9) )
        
        # compute project and loss for multiple images, then sum
        self.loss_sum = (lambda fv,r,organgst,ctfinfo,wnsparam, img2,  _pts3_fpx, _pts2_u, _pts_s_u, _pts_rad:
            vmap(self.project_and_loss, (None, 0,0,0,None,0, None,None,None,None) , 0)(fv,r,organgst,ctfinfo,wnsparam, img2,_pts3_fpx, _pts2_u, _pts_s_u, _pts_rad).sum() )
        self.jit_loss_sum = jax.jit(self.loss_sum)
        
        # volume and weights gradient
        self.grad_vol_sum     = jax.grad(self.loss_sum, argnums=0)
        self.jit_grad_vol_sum = jax.jit(self.grad_vol_sum)
        
        self.grad_vol_and_w_sum     = jax.grad(self.loss_sum, argnums=[0,4])        
        # TODO: for w we can project and reuse. Write operator for this.
        self.jit_grad_vol_and_w_sum = jax.jit(self.grad_vol_and_w_sum)
        
        # images parameters gradients
        self.grad_r_shft      = jax.grad(self.project_and_loss, argnums=[1,2])
        self.vmap_grad_r_shft = vmap(self.grad_r_shft, (None, 0,0,0,None,0, None,None,None,None), (0,0))
        self.jit_vmap_grad_r_shft = jax.jit(self.vmap_grad_r_shft)

        self.jit_grad_r_shft  = jax.jit(self.grad_r_shft)
        self.vmap_jit_grad_r_shft = vmap(self.jit_grad_r_shft, (None, 0,0,0,None,0, None,None,None,None), (0,0))
        
        self.grad_r_shft_ctf      = jax.grad(self.project_and_loss, argnums=[1,2,3])
        self.vmap_grad_r_shft_ctf = vmap(self.grad_r_shft_ctf, (None, 0,0,0,None,0, None,None,None,None), (0,0,0))
        self.jit_vmap_grad_r_shft_ctf = jax.jit(self.vmap_grad_r_shft_ctf)

    def calc_cgb(self,fic_x,r,organgst,ctf, wns, images_fft, pts3_fpx, pts2_u, pts_s_u, pts_rad):
        cgb = -0.5 * jnp.conj(self.jit_grad_vol_sum( 0*fic_x, r,organgst,ctf, wns, images_fft, pts3_fpx, pts2_u, pts_s_u, pts_rad) )
        return cgb

    def get_cgAAopp(self, orient_array,organgst_array,ctf_array, wns, tmpimages, pts3_fpx, pts2_u, pts_s_u, pts_rad ):
        #print(tmpzeroimages.shape)
        #print(wns.shape)
        tmpzeroimages = 0*tmpimages
        return (lambda f :  0.5 * jnp.conj(self.jit_grad_vol_sum(f, orient_array,organgst_array,ctf_array, wns, tmpzeroimages, pts3_fpx, pts2_u, pts_s_u, pts_rad)))


        
