

class CryoProposals:

    def __init__(self, sigma_noise_iter, B, B_list, dt_list_hmc, L_hmc, M_iter):
        self.sigma_noise = sigma_noise
        self.B = B
        self.B_list = B_list
        self.dt_list_hmc = dt_list_hmc
        self.L_hmc = L_hmc
        self.M_iter = M_iter

#def get_jax_proposal_funcs(loss_func_batched0_iter, loss_proj_func_batched0_iter, loss_func_sum_iter, grad_loss_volume_sum_iter, rotate_and_interpolate_iter, 

  
    def proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_orientations_func, params_orientations):
        logPi = lambda a : -loss_func_batched0_iter(v, a, shifts, ctf_params, imgs_iter, sigma_noise_iter)

        angles1 = generate_orientations_func(key, angles0, **params_orientations)

        logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
            true_fun = lambda _ : logPi(angles0),
            false_fun = lambda _ : logPiX0,
            operand = None)

        logPiX1 = logPi(angles1)
        r = jnp.exp(logPiX1 - logPiX0)

        return angles1, r, logPiX1, logPiX0 

    @jax.jit
    def proposal_func_orientations_uniform(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter):
        empty_params = {}
        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_uniform_orientations_jax, empty_params)

    @jax.jit
    def proposal_func_orientations_perturb(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, sigma_perturb):
        key, subkey = random.split(key)
        sig_p = random.permutation(subkey, sigma_perturb)[0]
        orient_params = {'sig' : sig_p}

        return proposal_func_orientations(key, angles0, logPiX0, v, shifts, ctf_params, imgs_iter, generate_perturbed_orientations, orient_params)


    @jax.jit
    def proposal_func_shifts_local(key, shifts0, logPiX0, v, proj, ctf_params, imgs_iter):
        logPi = lambda sh : -loss_proj_func_batched0_iter(v, proj, sh, ctf_params, imgs_iter, sigma_noise_iter)

        logPiX0 = jax.lax.cond(jnp.sum(logPiX0) == jnp.inf,
            true_fun = lambda _ : logPi(shifts0),
            false_fun = lambda _ : logPiX0,
            operand = None)

        #if jnp.sum(logPiX0) == jnp.inf:
        #    logPiX0 = logPi(shifts0)

        key, subkey =  random.split(key)
        B0 = random.permutation(subkey, B_list)[0]

        N = shifts0.shape[0]
        shifts1 = random.normal(key, (N, 2)) * B0 + shifts0

        logPiX1 = logPi(shifts1)
        r = jnp.exp(logPiX1 - logPiX0)

        return shifts1, r, logPiX1, logPiX0


    @jax.jit 
    def proposal_func_mtm_orientations_shifts(key, as0, logPiX0, v, ctf_params, imgs_iter):
        key, *keys = random.split(key, 4)
        
        angles0 = as0[:,:3]
        shifts0 = as0[:,3:]

        angles1 = generate_uniform_orientations_jax(keys[0], angles0)
        proj = rotate_and_interpolate_iter(v, angles1)

        N_samples_shifts = 500
        N = angles0.shape[0]
        #B0 = random.permutation(keys[1], B_list)[0]
        #shifts1_states = random.normal(keys[2], (N,N_samples_shifts,2)) * B0
        shifts1_states = random.uniform(keys[2], (N, N_samples_shifts,2)) * 2 * B - B

        #s1 = np.linspace(-B,B,100)
        #s1x, s1y = jnp.meshgrid(s1,s1)
        #shifts1_states = jnp.array([s1x.ravel(), s1y.ravel()]).transpose()
        #shifts1_states = jnp.repeat(jnp.expand_dims(shifts1_states, 0), N, 0)
       
        # weights has shape [N, N_samples_shifts], w(y_i) = logPi(y_i)
        weights = -jax.vmap(loss_proj_func_batched0_iter, in_axes=(None,None,1,None,None,None))(v, proj, shifts1_states, ctf_params, imgs_iter, sigma_noise_iter).transpose()
        
        # Select the proposed state with probability proportional
        # to weights, batch mode (all images in parallel).
        keys = random.split(key, N) 
        sh1idx = jax.vmap(jax.random.categorical, in_axes=(0,0))(keys, weights) 
        shifts1 = jax.vmap(lambda s1_states_i, sh1idx_i : s1_states_i[sh1idx_i], in_axes=(0,0))(shifts1_states, sh1idx)
        # The weights corresponding to proposed state (angles1,shifts1) (i.e. logPiX1)
        weights1 = jax.vmap(lambda weights_i, sh1idx_i : weights_i[sh1idx_i], in_axes=(0,0))(weights, sh1idx)

        weights0 = -loss_func_batched0_iter(v,angles0,shifts0,ctf_params,imgs_iter,sigma_noise_iter)
        weights_reference = jax.vmap(lambda weights_i, sh1idx_i, w0_i : weights_i.at[sh1idx_i].set(w0_i), in_axes = (0,0,0))(weights, sh1idx, weights0)

        r = jax.vmap(ratio_sum_exp, in_axes=(0,0))(weights, weights_reference)

        as1 = jnp.concatenate([angles1,shifts1], axis=1)

        return as1, r, weights1, weights0 

         
    @jax.jit
    def ratio_sum_exp(a, b):
        """Given two arrays a=[A1, ..., An], b=[B1,..., Bn],
        compute the ratio sum(exp(a1)) / sum(exp(a2)) in a way
        that doesn't lead to nan's."""

        log_ratio = a[0] - b[0] \
            + jnp.log(jnp.sum(jnp.exp(a-a[0]))) \
            - jnp.log(jnp.sum(jnp.exp(b-b[0])))

        return jnp.exp(log_ratio)                

   

    def proposal_func_vol_batch(key, v0, logPiX0, angles, shifts, ctf_params, imgs_iter):
        def logPi_vol(v):
            loss = 0
            N_batch = angles.shape[0]
            for i in range(N_batch):
                loss += -loss_func_sum_iter(v, angles[i], shifts[i], ctf_params[i], imgs_iter[i], sigma_noise_iter) 

            return loss / N_batch

        def gradLogPi_vol(v): 
            grad = 0
            N_batch = angles.shape[0]
            for i in range(N_batch):
                grad += -jnp.conj(grad_loss_volume_sum_iter(v, angles[i], shifts[i], ctf_params[i], imgs_iter[i], sigma_noise_iter)) 

            return grad/N_batch
 
        # Moved this to the proposal_hmc function
        #if logPiX0 == jnp.inf:
        #    logPiX0 = logPi_vol(v0)

        return proposal_hmc(key, v0, logPiX0, logPi_vol, gradLogPi_vol, dt_list_hmc, L_hmc, M_iter)


    @jax.jit
    def proposal_func_vol(key, v0, logPiX0, angles, shifts, ctf_params, imgs_iter):
        logPi_vol = lambda v : -loss_func_sum_iter(v, angles, shifts, ctf_params, imgs_iter, sigma_noise_iter)
        gradLogPi_vol = lambda v : -jnp.conj(grad_loss_volume_sum_iter(v, angles, shifts, ctf_params, imgs_iter, sigma_noise_iter))

        # Moved the below to the proposal_hmc function for generality.
        #logPiX0 = jax.lax.cond(logPiX0 == jnp.inf,
        #    true_fun = lambda _ : logPi_vol(v0),
        #    false_fun = lambda _ : logPiX0,
        #    operand = None)
        
        return proposal_hmc(key, v0, logPiX0, logPi_vol, gradLogPi_vol, dt_list_hmc, L_hmc, M_iter)



    return proposal_func_orientations_uniform, proposal_func_orientations_perturb, proposal_func_shifts_local, proposal_func_vol, proposal_func_vol_batch, proposal_func_mtm_orientations_shifts
