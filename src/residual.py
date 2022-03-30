import time
import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm

from src.projection import rotate_z0
from src.interpolate import find_nearest_one_grid_point_idx


def get_volume_residual(imgs, angles, sigma_noise, nx, N_batches):

    ### First rotate each residual img and return the list of coordinates and coordinate-based residuals
    N_batch_small=10 

    imgs_batches = np.array_split(imgs, N_batch_small)
    angles_batches = np.array_split(angles, N_batch_small)

    print(f"Rotate each image and get list of coords. {imgs.shape[0]} images in {N_batch_small} batches...", end="", flush=True)
    t0 = time.time()
    coords_resid =  [voxel_wise_resid_from2d_fun(img, ang, sigma_noise, nx) 
            for img, ang in zip(imgs_batches, angles_batches)]
    coords, resid = zip(*coords_resid)
    
    coords = np.concatenate(coords, axis=0)
    resid = np.concatenate(resid, axis=0)
    print(f"done in {time.time()-t0} seconds.", flush=True)


    ### Then find the nearest voxel for each coordinate 
    ### and average all the residuals in each voxel
    
    # Make sure the batches are on the CPU.
    coords_batches = np.array_split(coords, N_batches)
    resid_batches = np.array_split(np.array(resid), N_batches)

    print(f"Average residuals in each voxel. {coords.shape[0]} residuals in {N_batches} batches...", end="", flush=True)
    t0 = time.time()

    v_resid_sum = np.zeros([nx,nx,nx])
    v_resid_counts = np.zeros([nx,nx,nx])

    for i in tqdm(range(N_batches)):
        vrs, vrc = get_v_resid(coords_batches[i], resid_batches[i], np.zeros([nx,nx,nx]))

        v_resid_sum += vrs
        v_resid_counts += vrc

    # Add 1e-16 to avoid NaN when counts=0.
    v_resid = v_resid_sum/(v_resid_counts+1e-16)
    
    print(f"done in {time.time()-t0} seconds.", flush=True)

    return v_resid


def voxel_wise_resid_fun(v, angles, shifts, ctf_params, imgs, sigma_noise, x_grid, slice_func_array):

    resid = jnp.abs(slice_func_array(v, angles, shifts, ctf_params) - imgs)/sigma_noise
    resid = resid.reshape(-1)   

    proj_coords = jax.vmap(rotate_z0, in_axes = (None, 0))(x_grid, angles)
    proj_coords = proj_coords.swapaxes(0,1).reshape(3,-1).transpose()

    return proj_coords, resid


def voxel_wise_resid_from2d_fun(imgs, angles, sigma_noise, nx):
    x_grid = jnp.array([1, nx])

    resid = jnp.abs(imgs)/sigma_noise
    resid = resid.reshape(-1)   

    proj_coords = jax.vmap(rotate_z0, in_axes = (None, 0))(x_grid, angles)
    proj_coords = proj_coords.swapaxes(0,1).reshape(3,-1).transpose()

    return proj_coords, resid


@jax.jit
def get_v_resid(coords, resid, zero_vol):
    nx = zero_vol.shape[0]
    x_grid = [1, nx]

    v_idx = jax.vmap(find_nearest_one_grid_point_idx, in_axes=(0,None,None,None))(coords, x_grid, x_grid, x_grid) 

    v_resid_sum = jnp.zeros([nx,nx,nx])
    v_resid_counts = jnp.zeros([nx,nx,nx])

    def body_fun(i, rsc):
        v_resid_sum, v_resid_counts = rsc
        v_resid_sum = v_resid_sum.at[tuple(v_idx[i])].add(resid[i])
        v_resid_counts = v_resid_counts.at[tuple(v_idx[i])].add(1)
        return (v_resid_sum, v_resid_counts) 

    v_resid_sum, v_resid_counts = jax.lax.fori_loop(0, v_idx.shape[0], body_fun, (v_resid_sum, v_resid_counts))
    
    return v_resid_sum, v_resid_counts
