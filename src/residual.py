import numpy as np
import jax.numpy as jnp
import jax
from tqdm import tqdm

from src.projection import rotate_z0
from src.interpolate import find_nearest_one_grid_point_idx


def get_volume_residual(v, angles, shifts, ctf_params, imgs, x_grid, slice_func_array, N_batches):

    coords, resid =  voxel_wise_resid_fun(v, angles, shifts, ctf_params, imgs, x_grid, slice_func_array)
    nn_vol_idx = jax.vmap(find_nearest_one_grid_point_idx, in_axes=(0,None,None,None))(coords, x_grid, x_grid, x_grid)
    nx = x_grid[1].astype(jnp.int32)

    @jax.jit
    def get_v_resid(v_idx, resid):
        v_resid_sum = jnp.zeros([nx,nx,nx])
        v_resid_counts = jnp.zeros([nx,nx,nx])

        for i in jnp.arange(v_idx.shape[0]):
            v_resid_sum = v_resid_sum.at[tuple(v_idx[i])].add(resid[i])
            v_resid_counts = v_resid_counts.at[tuple(v_idx[i])].add(1)

        return v_resid_sum, v_resid_counts

    def get_v_resid_batch(vol_idx, resid, N_batches, get_v_resid):
        # Make sure the batches are on the CPU.
        vol_idx_batches = np.array_split(np.array(vol_idx), N_batches)
        resid_batches = np.array_split(np.array(resid), N_batches)

        v_resid_sum = np.zeros([nx,nx,nx])
        v_resid_counts = np.zeros([nx,nx,nx])

        for i in tqdm(range(len(vol_idx_batches))):
            vrs, vrc = get_v_resid(vol_idx_batches[i], resid_batches[i])

            v_resid_sum += vrs
            v_resid_counts += vrc

        # Add 1e-16 to avoid NaN when counts=0.
        v_resid = v_resid_sum/(v_resid_counts+1e-16)

        return v_resid

    v_resid = get_v_resid_batch(nn_vol_idx, resid, N_batches, get_v_resid)
    return v_resid



def voxel_wise_resid_fun(v, angles, shifts, ctf_params, imgs, x_grid, slice_func_array):

    resid = jnp.abs(slice_func_array(v, angles, shifts, ctf_params) - imgs)
    resid = resid.reshape(-1)   

    proj_coords = jax.vmap(rotate_z0, in_axes = (None, 0))(x_grid, angles)
    proj_coords = proj_coords.swapaxes(0,1).reshape(3,-1).transpose()

    return proj_coords, resid