import jax
from jax import random
import jax.numpy as jnp
from functools import partial

from simplecryoem.jaxops import Slice, Loss, GradV
from simplecryoem.utils import generate_uniform_orientations_jax, generate_perturbed_orientations
from .hmc import proposal_hmc


class CryoProposals:
    """Class containing MCMC proposal functions that can be useful for
    cryo-EM data processing.
    
    Instantiated with:
    - a noise level sigma_noise
    - a given set of HMC parameters (B, B_list, dt_list_hmc, L_hmc, M)
    - Slice, Loss, GradV objects

    Its methods represent MCMC proposal functions for volume, orientations and shifts.
    """

    def __init__(
        self,
        sigma_noise,
        B,
        B_list,
        dt_list_hmc,
        L_hmc,
        M,
        slice: Slice,
        loss: Loss,
        gradv: GradV,
    ):
        self.sigma_noise = sigma_noise
        self.B = B
        self.B_list = B_list
        self.dt_list_hmc = dt_list_hmc
        self.L_hmc = L_hmc
        self.M = M
        self.slice = slice
        self.loss = loss
        self.gradv = gradv

    @partial(jax.jit, static_argnums=(0,))
    def proposal_orientations_uniform(
        self, key, angles0, logPiX0, v, shifts, ctf_params, imgs
    ):
        """Propose new angles sampled uniformly on SO(3)."""

        empty_params = {}
        return self._proposal_orientations(
            key,
            angles0,
            logPiX0,
            v,
            shifts,
            ctf_params,
            imgs,
            generate_uniform_orientations_jax,
            empty_params,
        )

    @partial(jax.jit, static_argnums=(0,))
    def proposal_orientations_perturb(
        self, key, angles0, logPiX0, v, shifts, ctf_params, imgs, sigma_perturb
    ):
        """Propose new angles sampled from a normal distribution 
        around the current angles `angles0`."""

        key, subkey = random.split(key)
        sig_p = random.permutation(subkey, sigma_perturb)[0]
        orient_params = {"sig": sig_p}

        return self._proposal_orientations(
            key,
            angles0,
            logPiX0,
            v,
            shifts,
            ctf_params,
            imgs,
            generate_perturbed_orientations,
            orient_params,
        )

    def _proposal_orientations(
        self,
        key,
        angles0,
        logPiX0,
        v,
        shifts,
        ctf_params,
        imgs,
        generate_orientations_func,
        params_orientations,
    ):
        """Generic orientations proposal function that is called by the
        public orientations proposal functions above, together with a
        function that generates new angles."""

        # logPi = lambda a : -loss_func_batched0_iter(v, a, shifts, ctf_params, imgs, sigma_noise_iter)
        logPi = lambda a: -self.loss.loss_batched0(
            v, a, shifts, ctf_params, imgs, self.sigma_noise
        )

        angles1 = generate_orientations_func(key, angles0, **params_orientations)

        logPiX0 = jax.lax.cond(
            jnp.sum(logPiX0) == jnp.inf,
            true_fun=lambda _: logPi(angles0),
            false_fun=lambda _: logPiX0,
            operand=None,
        )

        logPiX1 = logPi(angles1)
        r = jnp.exp(logPiX1 - logPiX0)

        return angles1, r, logPiX1, logPiX0

    @partial(jax.jit, static_argnums=(0,))
    def proposal_shifts_local(self, key, shifts0, logPiX0, v, proj, ctf_params, imgs):
        """Propose new shifts sampled from a normal distribution 
        around the current shifts `shifts0`."""

        # logPi = lambda sh : -loss_proj_func_batched0_iter(v, proj, sh, ctf_params, imgs, self.sigma_noise)
        logPi = lambda sh: -self.loss.loss_proj_batched(
            v, proj, sh, ctf_params, imgs, self.sigma_noise
        )

        logPiX0 = jax.lax.cond(
            jnp.sum(logPiX0) == jnp.inf,
            true_fun=lambda _: logPi(shifts0),
            false_fun=lambda _: logPiX0,
            operand=None,
        )

        # if jnp.sum(logPiX0) == jnp.inf:
        #    logPiX0 = logPi(shifts0)

        key, subkey = random.split(key)
        B0 = random.permutation(subkey, self.B_list)[0]

        N = shifts0.shape[0]
        shifts1 = random.normal(key, (N, 2)) * B0 + shifts0

        logPiX1 = logPi(shifts1)
        r = jnp.exp(logPiX1 - logPiX0)

        return shifts1, r, logPiX1, logPiX0

    @partial(jax.jit, static_argnums=(0,))
    def proposal_vol(self, key, v0, logPiX0, angles, shifts, ctf_params, imgs):
        """Propose new volume using Hamiltonian Monte Carlo (HMC)."""

        logPi_vol = lambda v: -self.loss.loss_sum(
            v, angles, shifts, ctf_params, imgs, self.sigma_noise
        )
        gradLogPi_vol = lambda v: -jnp.conj(
            self.gradv.grad_loss_volume_sum(
                v, angles, shifts, ctf_params, imgs, self.sigma_noise
            )
        )

        return proposal_hmc(
            key,
            v0,
            logPiX0,
            logPi_vol,
            gradLogPi_vol,
            self.dt_list_hmc,
            self.L_hmc,
            self.M,
        )

    def proposal_vol_batch(self, key, v0, logPiX0, angles, shifts, ctf_params, imgs):
        """Similar to proposal_vol, but it loads the images to GPU in batches to
        compute the gradient, so it is not jit-ed."""

        def logPi_vol(v):
            loss = 0
            N_batch = angles.shape[0]
            for i in range(N_batch):
                loss += -self.loss.loss_sum(
                    v, angles[i], shifts[i], ctf_params[i], imgs[i], self.sigma_noise
                )

            return loss / N_batch

        def gradLogPi_vol(v):
            grad = 0
            N_batch = angles.shape[0]
            for i in range(N_batch):
                grad += -jnp.conj(
                    self.gradv.grad_loss_volume_sum(
                        v,
                        angles[i],
                        shifts[i],
                        ctf_params[i],
                        imgs[i],
                        self.sigma_noise,
                    )
                )

            return grad / N_batch

        return proposal_hmc(
            key,
            v0,
            logPiX0,
            logPi_vol,
            gradLogPi_vol,
            self.dt_list_hmc,
            self.L_hmc,
            self.M,
        )

    @partial(jax.jit, static_argnums=(0, 7))
    def proposal_mtm_orientations_shifts(
        self, key, as0, logPiX0, v, ctf_params, imgs, N_samples_shifts=100
    ):
        """Propose both angles and shifts using Multiple-try Metropolis."""

        key, *keys = random.split(key, 4)

        angles0 = as0[:, :3]
        shifts0 = as0[:, 3:]

        angles1 = generate_uniform_orientations_jax(keys[0], angles0)
        proj = self.slice.rotate_and_interpolate_vmap(v, angles1)

        N = angles0.shape[0]
        shifts1_states = (
            random.uniform(keys[2], (N, N_samples_shifts, 2)) * 2 * self.B - self.B
        )

        # weights has shape [N, N_samples_shifts], w(y_i) = logPi(y_i)
        weights = -jax.vmap(
            self.loss.loss_proj_batched0, in_axes=(None, None, 1, None, None, None)
        )(v, proj, shifts1_states, ctf_params, imgs, self.sigma_noise).transpose()

        # Select the proposed state with probability proportional
        # to weights, batch mode (all images in parallel).
        keys = random.split(key, N)
        sh1idx = jax.vmap(jax.random.categorical, in_axes=(0, 0))(keys, weights)
        shifts1 = jax.vmap(
            lambda s1_states_i, sh1idx_i: s1_states_i[sh1idx_i], in_axes=(0, 0)
        )(shifts1_states, sh1idx)
        # The weights corresponding to proposed state (angles1,shifts1) (i.e. logPiX1)
        weights1 = jax.vmap(
            lambda weights_i, sh1idx_i: weights_i[sh1idx_i], in_axes=(0, 0)
        )(weights, sh1idx)

        weights0 = -self.loss.loss_batched0(
            v, angles0, shifts0, ctf_params, imgs, self.sigma_noise
        )
        weights_reference = jax.vmap(
            lambda weights_i, sh1idx_i, w0_i: weights_i.at[sh1idx_i].set(w0_i),
            in_axes=(0, 0, 0),
        )(weights, sh1idx, weights0)

        r = jax.vmap(self._ratio_sum_exp, in_axes=(0, 0))(weights, weights_reference)
        as1 = jnp.concatenate([angles1, shifts1], axis=1)

        return as1, r, weights1, weights0

    @partial(jax.jit, static_argnums=(0,))
    def _ratio_sum_exp(self, a, b):
        """Given two arrays a=[A1, ..., An], b=[B1,..., Bn],
        compute the ratio sum(exp(a1)) / sum(exp(a2)) in a way
        that doesn't lead to nan's."""

        log_ratio = (
            a[0]
            - b[0]
            + jnp.log(jnp.sum(jnp.exp(a - a[0])))
            - jnp.log(jnp.sum(jnp.exp(b - b[0])))
        )

        return jnp.exp(log_ratio)
