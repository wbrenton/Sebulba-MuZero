import jax
import random
import numpy as np
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter

from learner import make_single_device_update
from network import make_muzero_network, NetworkApplys
from utils import TrainState
from replay_buffer import Batch
from main import parse_args, MuZeroAtariConfig


if __name__ == "__main__":
    args = parse_args()
    args.obs_shape = (96, 84, 84)
    args.num_actions = 4
    config = MuZeroAtariConfig(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, buffer_key = jax.random.split(key, 3)
    
    network = make_muzero_network(config)
    applys = NetworkApplys(*network.apply)

    init_obs = jnp.zeros((args.local_num_envs,  *args.obs_shape))
    init_action = jnp.zeros((args.local_num_envs, args.num_stacked_frames))
    network_params = network.init(network_key, init_obs, init_action)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(network_params)
    
    muzero_state = TrainState(params=network_params,
                        target_params=network_params,
                        opt_state=opt_state,
                        train_step=0)

    muzero_state = jax.device_put_replicated(muzero_state, devices=jax.devices())
    
    single_device_update = make_single_device_update(applys, optimizer, config)
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
    )
    
    B = args.local_num_envs
    T = args.num_stacked_frames
    C = 3
    R = 84
    
    U = args.num_unroll_steps
    S = 601
    
    observation = jnp.zeros((B, T, C, R, R))
    actions = jnp.zeros((B, T + U))
    
    s_value = jax.nn.one_hot(jnp.zeros(U), S)
    s_policy = jax.nn.one_hot(jnp.zeros(U), args.num_actions)
    # batch of s_policy
    policy = jnp.asarray([s_policy for _ in range(B)])
    value = jnp.asarray([s_value for _ in range(B)])
    reward = jnp.asarray([s_value for _ in range(B)])

    batch = Batch(
        observation=observation,
        actions=actions,
        value=value,
        policy=policy,
        reward=reward,
    )

    shard_fn = lambda x: jax.device_put_sharded(
    jnp.array_split(x, len(jax.devices()), axis=0),
    jax.devices(),
    )

    sharded_batch = jax.tree_map(shard_fn, batch)

    for i in range(1000):
        muzero_state, loss, v_loss, p_loss, r_loss = multi_device_update(muzero_state, sharded_batch)
        step = muzero_state.train_step.mean().item()
        loss = jnp.mean(loss).item()
        v_loss = jnp.mean(v_loss).item()
        p_loss = jnp.mean(p_loss).item()
        r_loss = jnp.mean(r_loss).item()
        print(f"iter {step}, loss {loss:.4f}, v_loss {v_loss:.4f}, p_loss {p_loss:.4f}, r_loss {r_loss:.4f}")