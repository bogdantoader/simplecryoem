from __future__ import print_function
import numpy as onp
import sys
import time
import os

import jax.numpy as jnp

from pyfftw.builders import fft2
from numpy.fft import fftshift
#from pyfftw.builders import fft2


from pyem import mrc
from pyem import star


#
# Internal services
#

def read_mrc_one_by_one(files_base,particles):
    # todo: check that these are all in the same file!
    zreader = mrc.ZSliceReader(files_base+particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0])
    imgs = [zreader.read(ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX]) for i,ptcl in particles.iterrows()]
    #for i, ptcl in particles.iterrows():
    #    p1r = zreader.read(ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX])
    #    #print(i)
    zreader.close()
    return imgs


def read_mrc_idxs(files_base,particles):
    # todo: are all in same file?
    # todo: which compat parameter???
    #zreader = mrc.ZSliceReader(files_base+particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0])
    #imgs = [zreader.read(ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX]) for i,ptcl in particles.iterrows()]
    
    tmp_imgs = mrc.read(files_base+particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0])#,compat="relion")
    #print(particles[star.UCSF.IMAGE_ORIGINAL_INDEX].iloc[0])
    #print("read_mrc_idxs:",len(tmp_imgs),tmp_imgs.shape)
    #print("read_mrc_idxs:",mrc.read_header(files_base+particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0]))
    #for i, ptcl in particles.iterrows():
    #    print (i,ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX])
    #tmp_imgs2 = mrc.read_imgs(files_base+particles[star.UCSF.IMAGE_ORIGINAL_PATH].iloc[0], idx=0, num=-1)
    #print( ((npo.array(tmp_imgs)-npo.array(tmp_imgs2))**2).sum() )
    ###print("read_mrc_idxs:",len(tmp_imgs),tmp_imgs.shape)
    #print( particles.shape, tmp_imgs.shape , len(tmp_imgs.shape))
    #print( [ (i,ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX]) for i,ptcl in particles.iterrows()] )
    if len(tmp_imgs.shape) < 3:  # hack for case of one image
        tmp_imgs = onp.expand_dims( tmp_imgs , axis=-1) #.expand_dims(-1)
        #print("===",tmp_imgs.shape)
    imgs = [onp.transpose(tmp_imgs[:,:,ptcl[star.UCSF.IMAGE_ORIGINAL_INDEX]]) for i,ptcl in particles.iterrows()] ### NOTE THE TRANSPOSE
    #imgs = tmp_imgs
    return imgs



#def calculate_apix_array(df):
#    try:
#        if df.ndim == 2:
#            if star.Relion.IMAGEPIXELSIZE in df:
#                return df.iloc[:][star.Relion.IMAGEPIXELSIZE]
#            if star.Relion.MICROGRAPHPIXELSIZE in df:
#                return df.iloc[:][star.Relion.MICROGRAPHPIXELSIZE]
#            return 10000.0 * df.iloc[:][star.Relion.DETECTORPIXELSIZE] / df.iloc[:][star.Relion.MAGNIFICATION]
#        elif df.ndim == 1:
#            if star.Relion.IMAGEPIXELSIZE in df:
#                return df[star.Relion.IMAGEPIXELSIZE]
#            if star.Relion.MICROGRAPHPIXELSIZE in df:
#                return df[star.Relion.MICROGRAPHPIXELSIZE]
#            return 10000.0 * df[star.Relion.DETECTORPIXELSIZE] / df[star.Relion.MAGNIFICATION]
#        else:
#            raise ValueError
#    except KeyError:
#        return None



def df2apiximg( df ):
    """ Extracts image apix (units?)
    
    """
    apiximg = onp.array([ star.calculate_apix(p)
                           for jj,p in df.iterrows()])
    return apiximg






import cryojax.JEM3.em2jax as em2jax3


#
# User accessible
#

def read_star_mrcs(files_base, files_star_in, loadimages = True,  imagesfft = "full", returnungroup = True, threads=1):
    """
    TODO: documentation.
    
    Parameters
    ----------
    files_base, 
    files_star_in, 
    loadimages = True,  
    imagesfft = "full", 
    returnungroup = True, 
    threads=1
    
    Returns
    -------
    images_array, 
    index_array, 
    grpid_array, 
    orient_array, 
    angles_array,
    organgst_array, 
    orgpx_array, 
    apiximg_array, 
    ctf_array, 
    df, 
    gb
            

    
    """
    print("read_star_mrcs: load star")
    df = star.parse_star(files_base+files_star_in , keep_index=False)
    star.augment_star_ucsf(df)
    if 1==1:
        df[star.UCSF.IMAGE_ORIGINAL_PATH] = df[star.UCSF.IMAGE_PATH]
        df[star.UCSF.IMAGE_ORIGINAL_INDEX] = df[star.UCSF.IMAGE_INDEX]
    print("read_star_mrcs: sort star")
    df.sort_values(star.UCSF.IMAGE_ORIGINAL_PATH, inplace=True, kind="mergesort")
    print("read_star_mrcs: group star by mrc file")
    gb = df.groupby(star.UCSF.IMAGE_ORIGINAL_PATH)
    

    #print(df.columns)
    #print(df["rlnGroupNumber"])
    
    #print(df)
    print("read_star_mrcs: number of particles:",len(df))
    
    apix = star.calculate_apix(df)
    print("read_star_mrcs: Computed size is %f A" % apix)
    
    print("read_star_mrcs: grouping particles by output stack")
    gb = df.groupby(star.UCSF.IMAGE_PATH)
    #print(gb)
    
    
    t30=time.time()
    try:
        t31=time.time()
        ctf_array = [em2jax3.df2ctfparam( particles ) for fname, particles in gb]
        
        t32=time.time()
        orient_array = [em2jax3.df2rot( particles )[0] for fname, particles in gb]
        angles_array = [em2jax3.df2rot( particles )[1] for fname, particles in gb]
        
        apiximg_array= [em2jax3.df2apiximg( particles )  for fname, particles in gb]
        
        #print("apiximg_array=",apiximg_array)

        #apiximg_array2= [calculate_apix_array( particles )  for fname, particles in gb]
        #print("apiximg_array2=",apiximg_array2)

        
        orgpx_array  = [em2jax3.df2originpx( particles ) for fname, particles in gb]

        # shift in angstrem, not pixels
        organgst_array  = [orgpx_array[j1]*apiximg_array[j1].reshape(-1,1) for j1 in range(gb.count().shape[0])]

        index_array = [[p["index"] for jj,p in particles.iterrows()]  for fname, particles in gb]
        
        grpid_array = [j1*onp.ones(len(index_array[j1])) for j1 in range(len(index_array))]
        t34=time.time()
    except KeyboardInterrupt:
        print("read_star_mrcs: stopped (keyboard)")
    t35=time.time()
    #print("read_star_mrcs: convert parameters time: ",t32-t31, t34-t31, t35-t30)
    print("read_star_mrcs: convert parameters time: ", t35-t30)

    
    
    if loadimages:
        #print("read_star_mrcs: reading images... ")    
        #t30=time.time()
        #try:
        #    images_array_orig = [onp.array(read_mrc_one_by_one(files_base,particles)) for fname, particles in gb]       
        #except KeyboardInterrupt:
        #    print("read_star_mrcs: stopped (keyboard)")
        #t31=time.time()
        #print("read_star_mrcs: read images time: ",t31-t30)

        print("read_star_mrcs: reading images... ")    
        t30=time.time()
        try:
            images_array = [onp.array(read_mrc_idxs(files_base,particles)) for fname, particles in gb]       
        except KeyboardInterrupt:
            print("read_star_mrcs: stopped (keyboard)")
        t31=time.time()
        print("read_star_mrcs: read images time: ",t31-t30)

        #for j1 in range(len(images_array)):
        #    print( onp.linalg.norm(images_array[j1]-images_array_orig[j1]), onp.linalg.norm(images_array[j1]) )
        # 
        #for j1 in range(len(images_array)):
        #    print( images_array[j1].shape, images_array[j1].dtype )
        
        
        
        if imagesfft == None:
            print("read_star_mrcs: returning original images")
        else:
            if imagesfft == "full":
                #
                # TODO: untested fft!
                #
                print("read_star_mrcs: compute fft2...")
                                
                
                t40=time.time()
                op_fft = fft2(images_array[0][0].copy(),
                              threads=threads,
                              planner_effort="FFTW_ESTIMATE",
                              auto_align_input=True,
                              auto_contiguous=True)
                
                t101=time.time()
                images_array = [onp.array([op_fft( fftshift(f).copy(), onp.zeros(op_fft.output_shape, dtype=op_fft.output_dtype))
                                           for f in images_array[jj]]) for jj in range(len(images_array)) ]
                t102=time.time()
                #print("fft time:", t102-t101)
                
                #t103=time.time()                
                ##images_array = [onp.fft.fft2(onp.fft.fftshift(images_array[jj],axes=(1,2)))  for jj in range(len(images_array)) ]
                #t104=time.time()
                #print("test:", t102-t101,t104-t103, [onp.linalg.norm(images_array[jj]-images_array_T[jj])/ onp.linalg.norm(images_array_T[jj]) for jj in range(len(images_array))], onp.array([onp.linalg.norm(images_array[jj]-images_array_T[jj])/ onp.linalg.norm(images_array_T[jj]) for jj in range(len(images_array))]).sum() )
                
                t41=time.time()
                print("read_star_mrcs: fft time: ",t41-t40)
                #for j1 in range(len(images_array)):
                #    print( images_array[j1].shape, images_array[j1].dtype )
            else:
                raise Exception("read_star_mrcs: images fft","unknown value for imagesfft. None or \"full\" supported.")
        if returnungroup:
            images_array = onp.concatenate(images_array)
            print("read_star_mrcs: images_array.shape:", images_array.shape)

    else:
            images_array = []
    print("read_star_mrcs: len(images_array):", len(images_array))

    if returnungroup:
            orient_array = onp.concatenate(orient_array)
            angles_array = onp.concatenate(angles_array)
            orgpx_array  = onp.concatenate(orgpx_array)
            organgst_array = onp.concatenate(organgst_array)
            apiximg_array= onp.concatenate(apiximg_array)
            ctf_array = onp.concatenate(ctf_array)
            
            #defu_array= onp.concatenate(defu_array)
            #defv_array= onp.concatenate(defv_array)
            #defang_array= onp.concatenate(defang_array)
            #phaseshift_array= onp.concatenate(phaseshift_array)
            #kv_array= onp.concatenate(kv_array)
            #ac_array= onp.concatenate(ac_array)
            #cs_array= onp.concatenate(cs_array)
            #bf_array= onp.concatenate(bf_array)

            grpid_array = onp.concatenate(grpid_array)
            index_array = onp.concatenate(index_array)

            
    return images_array, index_array, grpid_array, orient_array, angles_array, organgst_array, orgpx_array, apiximg_array, ctf_array, df, gb
            


