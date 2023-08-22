import time
import numpy as np
import jax.numpy as jnp
import jax
from simplecryoem.projection import rotate_z0
from simplecryoem.interpolate import find_nearest_one_grid_point_idx


def get_volume_residual(imgs, angles, sigma_noise, x_grid, radius, N_batches):
    nx = int(x_grid[1])

    # Some jitted functions
    vol_coords_from2d_jit = jax.jit(lambda a: vol_coords_from2d_fun(a, radius, x_grid))
    find_nearest_one_grid_point_idx_partial = lambda c: find_nearest_one_grid_point_idx(
        c, x_grid, x_grid, x_grid
    )
    find_nearest_one_grid_point_vmap = jax.jit(
        jax.vmap(find_nearest_one_grid_point_idx_partial)
    )

    # First rotate each img and return the list of coordinates and coordinate-based residuals
    N_batch_small = 10
    angles_batches = np.array_split(angles, N_batch_small)

    print(
        f"Rotate each image and get list of coords. {imgs.shape[0]} images in {N_batch_small} batches...",
        end="",
        flush=True,
    )
    t0 = time.time()
    coords_idx = [vol_coords_from2d_jit(ang) for ang in angles_batches]
    coords, idx = zip(*coords_idx)

    coords = np.concatenate(coords, axis=0)
    idx = np.concatenate(idx, axis=0)

    # Make imgs and sigma have the same shape as coords.
    imgs_arr = imgs.reshape(-1)
    sigma_arr = np.tile(sigma_noise, reps=(imgs.shape[0], 1)).reshape(-1)

    # Filter the coords, imgs and sigma that fall outside the mask radius
    coords = coords[idx]
    imgs_arr = imgs_arr[idx]
    sigma_arr = sigma_arr[idx]

    print(f"done in {time.time()-t0} seconds.", flush=True)

    # Then find the nearest voxel for each coordinate
    # and average all the residuals in each voxel

    # Make sure the batches are on the CPU.
    coords_batches = np.array_split(coords, N_batches)
    imgs_arr_batches = np.array_split(np.array(imgs_arr), N_batches)
    sigma_arr_batches = np.array_split(np.array(sigma_arr), N_batches)

    print(
        f"Average residuals in each voxel. {coords.shape[0]} residuals in {N_batches} batches.",
        flush=True,
    )
    t0 = time.time()

    vol_sum = np.zeros([nx, nx, nx])
    vol_counts = np.zeros([nx, nx, nx])
    vol_sigma = jnp.zeros([nx, nx, nx])
    # for i in tqdm(range(N_batches)):
    for i in range(N_batches):
        t0 = time.time()

        v_idx = find_nearest_one_grid_point_vmap(coords_batches[i])
        v_sum, v_sigma, v_counts = vol_from_coords(
            v_idx, imgs_arr_batches[i], sigma_arr_batches[i], nx
        )

        vol_sum += v_sum
        vol_counts += v_counts

        vol_sigma = merge_sigma_vols(vol_sigma, v_sigma)

        if np.mod(i, 10) == 0:
            print(f"Batch {i}, {time.time()-t0} seconds.", flush=True)

    # Add 1e-16 to avoid NaN when counts=0 (and where vol_sum=0 too).
    vol = vol_sum / (vol_counts + 1e-16)

    print(f"done in {time.time()-t0} seconds.", flush=True)

    return vol, vol_sigma, vol_counts


def vol_coords_from2d_fun(angles, radius, x_grid):
    proj_coords = jax.vmap(rotate_z0, in_axes=(None, 0))(x_grid, angles)
    proj_coords = proj_coords.swapaxes(0, 1).reshape(3, -1).transpose()

    idx = jnp.apply_along_axis(
        lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2 <= radius**2,
        arr=proj_coords,
        axis=1,
    )
    return proj_coords, idx


def vol_from_coords(v_idx, imgs_arr, sigma, nx):
    v_sum = jnp.zeros(nx**3)
    v_counts = jnp.zeros(nx**3)
    v_sigma = jnp.zeros(nx**3)

    x, y, z = v_idx.transpose()
    linear_idx = jnp.ravel_multi_index((x, y, z), dims=[nx, nx, nx])

    resid_arr = jnp.real(jnp.conj(imgs_arr) * imgs_arr)

    v_sum = v_sum.at[linear_idx].add(resid_arr).reshape([nx, nx, nx])
    v_counts = (
        v_counts.at[linear_idx].add(jnp.ones(imgs_arr.shape)).reshape([nx, nx, nx])
    )
    v_sigma = v_sigma.at[linear_idx].set(sigma).reshape([nx, nx, nx])

    return v_sum, v_sigma, v_counts


@jax.jit
def merge_sigma_entries(s1, s2):
    # Under the assumption that at a certain coordinate, sigma can only have
    # one value, we select the first non-zero value.
    return jax.lax.cond(
        s1 != 0, true_fun=lambda _: s1, false_fun=lambda _: s2, operand=None
    )


@jax.jit
def merge_sigma_vols(v1, v2):
    v_shape = v1.shape
    v = jax.vmap(merge_sigma_entries, in_axes=(0, 0))(v1.reshape(-1), v2.reshape(-1))
    return v.reshape(v_shape)
