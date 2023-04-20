import jax
import rlax
import optax
import jax.numpy as jnp

"""every batch that comes in needs to be 
    obs.shape (batch_dim, 32+5, 3, 86, 86)
    actions.shape (batch_dim, 32+5,)
    else.shape (batch_dim, 5, *else.shape[1:])
    then:
         intial_obs = obs[:, :32, 3, 86, 86].reshape(-1, 32*3, -1, -1)
         intial_actions = actions[:, :32]
         representation_network_input = np.hstack(initial_obs, intial_actions)
    """

def make_single_device_update(applys, optimizer, config):
    args = config.args
    
    def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
        """Multiplies the gradient of `x` by `scale`."""

        @jax.custom_gradient
        def wrapped(x: jnp.ndarray):
            return x, lambda grad: (grad * scale,)

        return wrapped(x)

    def softmax_cross_entropy(logits, labels):
        labels = jax.lax.stop_gradient(labels)
        return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)

    def loss_fn(params, batch):
        loss = 0
        B = args.batch_size
        K = args.num_unroll_steps

        # initial inference
        obs_stack = batch.observation.reshape((-1, *args.obs_shape))
        action_stack = batch.action[:, :-K]
        h_0, v_0, p_0 = applys.initial_inference(params, obs_stack, action_stack)
        predictions = [(1.0, None, v_0, p_0)]

        # TODO: pretty sure you wont reach the last K
        # unroll model with recurrent inference for K steps
        h_k = h_0
        unroll_actions = batch.action[:, -K:]
        for k in range(K):
            h_kp1, r_k, v_k, p_k  = applys.recurrent_inference(params, h_k, unroll_actions[:, k])
            predictions.append((1/K, r_k, v_k, p_k))
            h_k = h_kp1

        # calulate loss
        for k, (scale, r_k, v_k, p_k) in enumerate(predictions):
            v_loss = jnp.mean(softmax_cross_entropy(v_k, batch.value[:, k]))
            p_loss = jnp.mean(softmax_cross_entropy(p_k, batch.policy[:, k]))
            if k != 0 and r_k is not None:
                r_loss = jnp.mean(softmax_cross_entropy(r_k, batch.reward[:, k]))
                l_k = scale_gradient(v_loss + p_loss + r_loss, scale)
            else:
                l_k = scale_gradient(v_loss + p_loss, scale)
            loss += l_k
            h_k = h_kp1

        return loss / B, (v_loss, p_loss, r_loss)

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def single_device_update(muzero_state, batch):
        (loss, (v_loss, p_loss, r_loss)), grads = value_and_grad(muzero_state.params, batch)
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        updates, new_opt_state = optimizer.update(grads, muzero_state.opt_state, muzero_state.params)
        new_params = optax.apply_updates(muzero_state.params, updates)
        muzero_state = muzero_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            train_step=muzero_state.train_step + 1
        )
        return muzero_state, loss, v_loss, p_loss, r_loss
    return single_device_update

def make_compute_value_target(num_unroll_steps, td_steps, gamma):
    td_steps = jnp.array(td_steps, dtype=jnp.float32)

    def compute_value_target_fn(rewards, values, dones):
        discounts = jnp.zeros_like(rewards, dtype=jnp.float32)
        discounts = jnp.where(dones, 0.0, gamma)

        value_targets = rlax.n_step_bootstrapped_returns(rewards, discounts, values, num_unroll_steps, td_steps)

        return value_targets

    return jax.vmap(compute_value_target_fn)