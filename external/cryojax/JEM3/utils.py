from typing import TypeVar
#import numpy as onp
import jax.numpy as jnp


########## General tools ##################

def l2square( x ):
    """ Computes l2 norm square of complex array
    """
    return ((x.conj()*x).real.sum())
    #return jnp.real((x.conj()*x).sum())

def wl2square( x, wgt ):
    """ Computes *weighted* l2 norm square of complex array
    """
    return ((wgt*(x.conj()*x).real).sum())
    #return jnp.real((wgt*(x.conj()*x)).sum())


def rot_coor( rot, coor ):
    """ Rotate 3-d coordinates using a rotation matrix
    
    Parameters
    ----------
    rot  : 3x3 matrix(jax or np)
    coors : array of m-d vectors. 
           Shape: N_1 x ... x N_n x m

    Returns
    -------
    x: array with the shape of vecs.
    """
    return square_mat_vecs_mul( rot, coor )

def square_mat_vecs_mul( mat, vecs ):
    """ Multiply an array of vectors by one matrics. 
    Intended for rotating coordinates using a single rotation matrix (case of m=3).
    
    Parameters
    ----------
    mat  : mxm matrix(jax or np)
    vecs : array of m-d vectors. 
           Shape: N_1 x ... x N_n x m
    
    Returns
    -------
    x: array with the shape of vecs.
    
    Each vector v in the last dimension of coor is hit by mat: mat @ v
    
    """
    x = (mat @ vecs.reshape((-1,vecs.shape[-1],1))).reshape(vecs.shape)    
    return x



#
# debug and testing
#
if __name__ == '__main__':
    import jax
    
    #
    # older functions to compare to
    #
    def rot_coor_old( rot, coor ):
        """ Rotate 3-d coordinates using a rotation matrix

        Parameters
        ----------
        rot  : 3x3 matrix(jax or np)
        coors : array of m-d vectors. 
               Shape: N_1 x ... x N_n x m

        Returns
        -------
        x: array with the shape of vecs.
        """
        # single rotation
        x = (rot @ coor.reshape((-1,3,1))).reshape(coor.shape)    
        return x

    #
    # Tests
    #
    print(" === Testing JEM3 utils ===")
    key = jax.random.PRNGKey(123)
    

    # for the default float type of jax, get machine precision.
    fdtype = jnp.ones(1).dtype
    feps = jnp.finfo(fdtype).eps

    print("\t Default float:", fdtype, "\t eps:", feps)
    
    #
    # Test: rotations
    #
    print("JEM3: utils: Rotations")
    #onp.random.seed(123)
    #x = jnp.array(onp.random.rand(4,5,3))
    #r = jnp.array(onp.random.rand(3,3))
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey,(4,5,3))
    _ , subkey = jax.random.split(subkey)
    r = jax.random.normal(subkey,(3,3))
    #print(x.dtype)
    
    rot_df = (rot_coor_old(r,x) - square_mat_vecs_mul(r,x))
    rot_dfnorm = jnp.linalg.norm(rot_df)/jnp.linalg.norm(rot_coor_old(r,x))
    if (rot_dfnorm>0):
        print("Warning: rotation comparison isn't =0. \n\t It should be identically 0 becuase it is the same operation, but could be handled differently in practice.\n\t Actual relative error: ", rot_dfnorm)
    if (rot_dfnorm > feps*jnp.sqrt(jnp.prod(x.shape))*10):
        print("ERROR: rotations don't match.", rot_dfnorm )
    #print( rot_dfnorm, feps, jnp.sqrt(jnp.prod(x.shape))*10 )

    #
    # Test Norms
    #
    print("JEM3: utils: norm")
    key , subkey1 = jax.random.split(key)
    key , subkey2 = jax.random.split(key)
    key , subkey3 = jax.random.split(key)
    x = jax.random.normal(subkey1,(3,4,5))+1j*jax.random.normal(subkey2,(3,4,5))
    w = jnp.abs(jax.random.normal(subkey,(3,4,5)))
    norm2_df = jnp.abs( jnp.linalg.norm(x)**2 - l2square( x ))
    norm2_dfrel = norm2_df / jnp.abs( jnp.linalg.norm(x)**2 )
    wnorm2_df = jnp.abs( jnp.linalg.norm(jnp.sqrt(w)*x)**2 - wl2square( x,w ))
    wnorm2_dfrel = wnorm2_df / jnp.abs( jnp.linalg.norm(jnp.sqrt(w)*x)**2 )
        
    if norm2_dfrel > feps*jnp.sqrt(jnp.prod(x.shape))*10 :
        print("ERROR: norm2 not consistent", norm2_df, feps*jnp.sqrt(jnp.prod(x.shape))*10 )
    if wnorm2_dfrel > feps*jnp.sqrt(jnp.prod(x.shape))*10 :
        print("ERROR: weighted norm2 not consistent")
