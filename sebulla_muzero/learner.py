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
    
    def softmax_cross_entropy(logits, labels):
        return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)

    # TODO: add in gradient clipping and stuff
    def loss_fn(params, batch):
        K = args.num_unroll_steps
        v_loss, p_loss, r_loss = 0, 0, 0

        # initial inference
        obs_stack = batch.observation.reshape((-1, *args.obs_shape))
        action_stack = batch.actions[:, :-K]
        embedding, value, policy = applys.initial_inference(params, obs_stack, action_stack)
        v_loss = jnp.add(v_loss, softmax_cross_entropy(value, batch.value[:, -K]))
        p_loss = jnp.add(p_loss, softmax_cross_entropy(policy, batch.policy[:, -K]))

        # unroll model with recurrent inference for K steps
        h_k = embedding
        unroll_actions = batch.actions[:, -K:]
        for k in range(K):
            h_kp1, r_k, v_k, p_k  = applys.recurrent_inference(params, h_k, unroll_actions[:, k])
            v_loss = jnp.add(v_loss, softmax_cross_entropy(v_k, batch.value[:, k]))
            p_loss = jnp.add(p_loss, softmax_cross_entropy(p_k, batch.policy[:, k]))
            r_loss = jnp.add(r_loss, softmax_cross_entropy(r_k, batch.reward[:, k]))
            h_k = h_kp1

        v_loss = v_loss / K
        p_loss = p_loss / K
        r_loss = r_loss / K

        loss = v_loss + p_loss + r_loss
        return loss, (v_loss, p_loss, r_loss)

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def single_device_update(muzero_state, batch):
        (loss, (v_loss, p_loss, r_loss)), grads = value_and_grad(muzero_state.params, batch)
        grads = jax.lax.pmean(grads, axis_name="local_devices")
        updates, new_opt_state = optimizer.update(grads, muzero_state.opt_state)
        new_params = optax.apply_updates(muzero_state.params, updates)
        muzero_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            train_step=muzero_state.train_step + 1
        )
        return muzero_state, loss, v_loss, p_loss, r_loss
    return single_device_update


def make_prepare_data_fn(args, learner_devices, scalar_to_categorical):
    """
    Creates a function that turns data generated by the parallel environment into a batch of data
    i.e. (time_dim, batch_dim, *x.shape) -> (batch_dim, *x.shape) where x is (obs, dones, values, actions, etc.)
    """

    compute_value_target = make_compute_value_target(args.num_unroll_steps, args.td_steps, args.gamma)

    def prepare_data(
        obs: list,
        dones: list,
        values: list,
        actions: list,
        mcts_policies: list,
        rewards: list,
    ):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        mcts_policies = jnp.asarray(mcts_policies, dtype=jnp.float32)
        rewards = jnp.asarray(rewards, dtype=jnp.float32)
        values = jnp.asarray(values, dtype=jnp.float32)
        dones = jnp.asarray(dones)
        value_targets = compute_value_target(rewards, values, dones)
        
        rewards = scalar_to_categorical(rewards)    
        value_targets = scalar_to_categorical(value_targets)
        
        return obs, actions, mcts_policies, rewards, value_targets

    return jax.jit(prepare_data, device=learner_devices[0])

def make_compute_value_target(num_unroll_steps, td_steps, gamma):
    td_steps = jnp.array(td_steps, dtype=jnp.float32)

    def compute_value_target_fn(rewards, values, dones):
        discounts = jnp.zeros_like(rewards, dtype=jnp.float32)
        discounts = jnp.where(dones, 0.0, gamma)

        value_targets = rlax.n_step_bootstrapped_returns(rewards, discounts, values, num_unroll_steps, td_steps)

        return value_targets

    return jax.vmap(compute_value_target_fn)