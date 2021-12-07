from typing import TypeVar
import numpy as onp
from functools import partial

import jax
import jax.numpy as jnp
#from jax import vmap
#from jax import jit

#from cryojax.JEM3 import utils as jem3utils
#from cryojax.JEM3 import lininterp as jem3lininterp
#from cryojax.JEM3 import projutils as jem3projutils
#from cryojax.JEM3 import coorutils as jem3coorutils
#from cryojax.JEM3 import emfiles as emfiles3
#from cryojax.JEM3 import ctf as ctf3


from jax import tree_util

#
# Test functions
#

#import numpy as onp
#import sys
import time
#import os

#from pyem import mrc
#from pyem import star
#from pyem import util

#from numpy.fft import fftshift
#from pyfftw.builders import fft2

#import cryojax.projectPyEM2.jaxvop2 as jvop2
#import cryojax.projectPyEM2.em2jax as em2jax
#import cryojax.projectPyEM2.jaxctf as jctf2
#import cryojax.projectPyEM2.emfiles as emfiles
#import cryojax.projectPyEM2.jaxops as jop
#from cryojax.projectPyEM2.jaxops import tobatches


#import jax.numpy as np
#from jax import grad, jit, vmap, value_and_grad



def getmybatches(nn,bsz):
    #return [list(range(j1,min(j1+bsz,nn))) for j1 in range(0,nn,bsz)]
    #return [tuple(range(j1,min(j1+bsz,nn))) for j1 in range(0,nn,bsz)]
    return [onp.array(range(j1,min(j1+bsz,nn))) for j1 in range(0,nn,bsz)]

###############3


def wrapexec_helper_in2(x,_in, _bt):
    if _in == None:
            return x
    if _in == 0:
            #print(_bt, x.shape)
            return x[_bt]
    print("ERROR")
    return None 

def wrapexec_helper_in(fun,x,in_axes,_bt):
    #return [wrapexec_helper_in2(x[jj],in_axes[jj],_bt) for jj in range(len(in_axes))]#( (wrapexec_helper_in2(x[j],in_axes[j],bt) for j in range(len(in_axes))) )
    return tuple([wrapexec_helper_in2(x[jj],in_axes[jj],_bt) for jj in range(len(in_axes))])#( (wrapexec_helper_in2(x[j],in_axes[j],bt) for j in range(len(in_axes))) )
    #return map( wrapexec_helper_in2(x[jj],in_axes[jj],_bt) , range(len(in_axes)))
    #return map( lambda j:j , range(len(in_axes)))


def wrapexec_helper_out(r0,r1,_out):
    if _out == None:
        return r0+r1
    if _out == 0:
        #print("help out",type(r0),type(r1))
        if isinstance(r0,onp.ndarray):
                return onp.concatenate([r0,r1],axis=0)
        if isinstance(r0,jnp.ndarray):
                return jnp.concatenate([r0,r1],axis=0)
        raise Exception("Wrong Variable Type fro wrapper")
    return None

def wrapexec_helper_tuples(r)-> list:
    if isinstance(r,tuple):
        r=list(r)
    else:
        r=[r,]
    return r

def tonumpy(r):
        if isinstance(r,jnp.ndarray):
                return onp.array(r)
        return r

def wrapexec(fun, myargs: list, in_axes:list, out_axes:list, nn:int,bsz:int , is2numpy = False):
    """ Run fun(*myarg) in batches
    
    Parameters
    ----------
    fun     : function to apply
    myargs  : list. Function argument
    in_axes : a list of the same length as the list of arguments. Only zeros and None are allowed.
              [None, 0, None] mean that the first and third variables are always passed to fun as-is, 
              whereas the second variables is devided to batchs of length up to bsz.
              If the second variable has more than nn elements in dimension 0, the rest of the elements are ignored.
    out_axes: a list of the length of the number of outputs of fun
              [None, 0] means that we should sum over the first output of the func for all the batches, and we should concatenate the second variable.
    nn      : number of elements in batched dimension
    bsz     : maximum number of elements in each batch
    
    is2numpy: convert jax output arrays to numpy arrays. Useful for very large output (e.g., projections)

    Returns
    -------
    If used correctly should return the same as fun(myargs)


    TODO: convert output to numpy? preassign array rather than concat?
    
    TODO: is there a problem with batches of size 1?
    """

    assert( len(myargs) == len(in_axes) )
    assert( set(in_axes) <= {0,None} )
    assert( set(out_axes) <= {0,None} )
    #return lambda x: [x[0:1],x[1:3]] #[wrapexec_helper_in(x,in_axes,[0]) , wrapexec_helper_in(x,in_axes,[1:2]) ]
    #return lambda *args: ([wrapexec_helper_in(fun,args,in_axes,[0,1]) , wrapexec_helper_in(fun,args,in_axes,[1,2]) ])
    #return ([fun(*wrapexec_helper_in(fun,myargs,in_axes,[0,1])) , fun(*wrapexec_helper_in(fun,myargs,in_axes,[1,2])) ])
    b = getmybatches(nn,bsz)
    #print(b)
    r = fun(*(wrapexec_helper_in(fun,myargs,in_axes,b[0])))
    r= wrapexec_helper_tuples(r)
    if is2numpy:
        r = [tonumpy(a) for a in r]
    #print("r",r)
    #print("r0",r[0].shape)
    for j0 in range(1,len(b)):
        rtmp = fun(*(wrapexec_helper_in(fun,myargs,in_axes,b[j0])))
        #print(j0,type(r),len(r),type(rtmp),len(rtmp))
        rtmp = wrapexec_helper_tuples(rtmp)
        if is2numpy:
            rtmp = [tonumpy(a) for a in rtmp]
        #print(j0,type(rtmp),len(rtmp))
        r = [wrapexec_helper_out(r[jj],rtmp[jj],out_axes[jj]) for jj in range(len(out_axes)) ]
    #print(len(r))
    
    if len(r)==1:
        return r[0]
    else:
        return tuple(r)






###########################################################################################





def wrapexec2_helper_out(r0,r1,_out):
    if _out == None:
        return r0+r1
    if _out == 0:
        return r0+[r1,]
    return None

def wrapexec2_helper_out_prep(r0,_out):
    if _out == None:
        return r0
    if _out == 0:
        return [r0,]
    return None

def wrapexec2_helper_out_merge(r0,_out):
    if _out == None:
        return r0
    if _out == 0:
        #print("help out",type(r0),type(r1))
        if isinstance(r0[0],onp.ndarray):
                return onp.concatenate(r0,axis=0)
        if isinstance(r0[0],jnp.ndarray):
                return jnp.concatenate(r0,axis=0)
        raise Exception("Wrong Variable Type fro wrapper")
    return None

def wrapexec2(fun, myargs: list, in_axes:list, out_axes:list, nn:int,bsz:int , is2numpy = False):
    """ Run fun(*myarg) in batches
    
    Parameters
    ----------
    fun     : function to apply
    myargs  : list. Function argument
    in_axes : a list of the same length as the list of arguments. Only zeros and None are allowed.
              [None, 0, None] mean that the first and third variables are always passed to fun as-is, 
              whereas the second variables is devided to batchs of length up to bsz.
              If the second variable has more than nn elements in dimension 0, the rest of the elements are ignored.
    out_axes: a list of the length of the number of outputs of fun
              [None, 0] means that we should sum over the first output of the func for all the batches, and we should concatenate the second variable.
    nn      : number of elements in batched dimension
    bsz     : maximum number of elements in each batch
    
    is2numpy: convert jax output arrays to numpy arrays. Useful for very large output (e.g., projections)

    Returns
    -------
    If used correctly should return the same as fun(myargs)


    TODO: simplify.
    
    TODO: convert output to numpy? preassign array rather than concat?
    
    TODO: is there a problem with batches of size 1?
    """

    assert( len(myargs) == len(in_axes) )
    assert( set(in_axes) <= {0,None} )
    assert( set(out_axes) <= {0,None} )
    #return lambda x: [x[0:1],x[1:3]] #[wrapexec_helper_in(x,in_axes,[0]) , wrapexec_helper_in(x,in_axes,[1:2]) ]
    #return lambda *args: ([wrapexec_helper_in(fun,args,in_axes,[0,1]) , wrapexec_helper_in(fun,args,in_axes,[1,2]) ])
    #return ([fun(*wrapexec_helper_in(fun,myargs,in_axes,[0,1])) , fun(*wrapexec_helper_in(fun,myargs,in_axes,[1,2])) ])
    b = getmybatches(nn,bsz)
    #print(b)
    r = fun(*(wrapexec_helper_in(fun,myargs,in_axes,b[0])))
    r = wrapexec_helper_tuples(r)
    if is2numpy:
        #print("numpy...")
        r = [tonumpy(a) for a in r]
    #print([type(aa) for aa in r])
    r = [wrapexec2_helper_out_prep(r[jj],out_axes[jj]) for jj in range(len(out_axes)) ]
    #print("r",r)
    #print("r0",r[0].shape)
    for j0 in range(1,len(b)):
        rtmp = fun(*(wrapexec_helper_in(fun,myargs,in_axes,b[j0])))
        #print(j0,type(r),len(r),type(rtmp),len(rtmp))
        rtmp = wrapexec_helper_tuples(rtmp)
        if is2numpy:
            rtmp = [tonumpy(a) for a in rtmp]
        #print(j0,type(rtmp),len(rtmp))
        r = [wrapexec2_helper_out(r[jj],rtmp[jj],out_axes[jj]) for jj in range(len(out_axes)) ]
    #print(len(r))

    r = [wrapexec2_helper_out_merge(r[jj],out_axes[jj]) for jj in range(len(out_axes)) ]
    
    if len(r)==1:
        return r[0]
    else:
        return tuple(r)








def test020():

    def f(a,b):
            return a, b

    fvx= wrapexec(f, ( jnp.array([10,20,30]), jnp.array([[100,101],[200,201],[300,301]]) ) , [0,0], [0,0], 3,2 , is2numpy = True)
    print(type(fvx), type(fvx[0]), fvx)


    fvx2= wrapexec2(f, ( jnp.array([10,20,30]), jnp.array([[100,101],[200,201],[300,301]]) ) , [0,0], [0,0], 3,2  , is2numpy = True )
    print(type(fvx2), fvx2)

    print(isinstance(fvx[0],onp.ndarray), type(fvx[0]) )
          #fv = wrapexec(f, x, [0,None], [None,0] )
    #print("fv",fv)

    #print("fv(x)",fv( jnp.array([10,20,30]), jnp.array([100,200,300]) ) )
    return
    

if __name__ == '__main__':
    test020()
