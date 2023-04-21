import jax
import time
import random
import numpy as np
from flax import struct
import jax.numpy as jnp
from jax import flatten_util
from tensorboardX import SummaryWriter

from main import parse_args, MuZeroAtariConfig
import zstandard as zstd

local_num_envs = 100
num_steps = 500
num_actions = 4

@struct.dataclass
class Batch:
    observation: list[jnp.ndarray]
    action: list[jnp.ndarray]
    value: list[jnp.ndarray]
    policy: list[jnp.ndarray]
    reward: list[jnp.ndarray]
    weight: jnp.ndarray
    index: jnp.ndarray

    def extend(self, o, a, v, p, r):
        self.observation.append(o)
        self.action.append(a)
        self.value.append(v)
        self.policy.append(p)
        self.reward.append(r)

@struct.dataclass
class GameHistory:
    observations: jnp.ndarray
    actions: jnp.ndarray
    values: jnp.ndarray
    policies: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

    def add_to_batch(self, batch, b_i, t_i):
        
        # TODO make args dynamic instead of hardcoding
        nsf = 32 # num stacked frames
        nus = 5 # num unroll steps
        tds = 5 # td steps
        
        @jax.vmap
        def _slice(b, t):
            _o = jax.lax.dynamic_slice_in_dim(self.observations[b], t, nsf, axis=0)
            _a = jax.lax.dynamic_slice_in_dim(self.actions[b], t, nsf+nus, axis=0)
            _v = jax.lax.dynamic_slice_in_dim(self.values[b], t, nus+tds, axis=0)
            _p = jax.lax.dynamic_slice_in_dim(self.policies[b], t, nus+tds, axis=0)
            _r = jax.lax.dynamic_slice_in_dim(self.rewards[b], t, nus+tds, axis=0)
            return _o, _a, _v, _p, _r

        o, a, v, p, r = _slice(b_i, t_i)
        batch.extend(o, a, v, p, r)

class ReplayBuffer:
    """
    - priorities is of the shape (compression_dim, batch_dim, time_dim)
    """
    def __init__(self):
        self.max_size = 10 # each element is a compressed game history containing 100 500 step rollouts
        self.batch_size = 1024
        self.timesteps_seen = 0

        self.compressor = zstd.ZstdCompressor()
        self.decompressor = zstd.ZstdDecompressor()

        self.buffer = []
        self.priorities = None

        self.alpha = 1
        self.beta = 1

    def add(self, b_game_history, b_priority):
        if not self._started():
            self._setup(b_game_history)
            self.priorities = jnp.expand_dims(b_priority, axis=0) # add compression dim
        else:
            c_priority = jnp.expand_dims(b_priority, axis=0) # add compression dim
            self.priorities = jnp.concatenate((self.priorities, c_priority), axis=0) # concat along compression dim

        if len(self.buffer) == self.max_size:
            self.buffer.pop(0)
            self.priorities = self.priorities[1:,...] # assert this is correct
        
        b_compressed = self._compress(b_game_history)
        self.buffer.append(b_compressed)
        self.timesteps_seen += b_priorities.size
        del b_game_history

    def sample(self, key):
        index, weights = self._sample(key) # this is surprisingly slow, consider preallocating priorities and using jnp.roll
        buffer_index, batch_index, time_index = index
        decompression_index = jnp.unique(buffer_index)
        batch = self.emtpy_batch(index, weights)
        for i in decompression_index:
            b_game_history = self._decompress(self.buffer[i])
            b_index = jnp.where(buffer_index == i)
            b_batch_index = batch_index[b_index]
            b_time_index = time_index[b_index]
            b_game_history.add_to_batch(batch, b_batch_index, b_time_index)
            del b_game_history

        batch = jax.tree_map(
            lambda x: jnp.concatenate(x) if type(x) == list else x,
            batch,
            is_leaf=lambda x: type(x) in [list, jnp.ndarray]
            )

        return batch

    # cant jit unless priorities shape is static
    def _sample(self, key):
        flat_priorities = self.priorities.ravel()
        probs = flat_priorities**self.alpha / flat_priorities.sum()
        flat_index = jax.random.choice(key, len(flat_priorities), shape=(self.batch_size,), replace=False, p=probs)
        sampled_weights = (1 / len(flat_priorities)) / probs[flat_index]
        sampled_index = jnp.unravel_index(flat_index, self.priorities.shape) 
        return sampled_index, sampled_weights

    def _compress(self, data):
        flat = self.flatten_fn(data)
        bytes = flat.tobytes()
        compressed = self.compressor.compress(bytes)
        del data, flat, bytes
        return compressed

    def _decompress(self, data):
        bytes = self.decompressor.decompress(data)
        flat = jnp.frombuffer(bytes, dtype=jnp.float32)
        d_data = self.unflatten_fn(flat)
        del data, bytes, flat
        return d_data

    def _setup(self, b_game_history):
        self.flatten_fn = lambda x: flatten_util.ravel_pytree(x)[0]
        self.flat_b_game_history_example, self.unflatten_fn = flatten_util.ravel_pytree(b_game_history)

        self.flatten_fn = jax.jit(self.flatten_fn)
        self.unflatten_fn = jax.jit(self.unflatten_fn)

    def emtpy_batch(self, index, weights):
        return Batch(
            observation=[],
            action=[],
            value=[],
            policy=[],
            reward=[],
            weight=weights,
            index=index,
        )

    def _started(self):
        return self.timesteps_seen > 0

    def buffer_memory_usage(self):
        return sum([len(b) for b in self.buffer]) / 1028**3


# ------------------------------------------ begin main -------------------------------------------------------
# ------------------------------------------ begin main -------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    args.obs_shape = (96, 84, 84)
    args.num_actions = 4
    config = MuZeroAtariConfig(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, buffer_key = jax.random.split(key, 3)

    replay_buffer = ReplayBuffer()

    # add N fake game histories to replay buffer
    for _ in range(10):
        b_priorities = jax.random.uniform(key, (args.local_num_envs, args.num_steps))
        b_game_history = GameHistory(
            observations = jax.random.uniform(key, (args.local_num_envs, args.num_steps+32, 3, 84, 84)),
            actions = jax.random.uniform(key, (args.local_num_envs, args.num_steps+32+5)),
            values = jax.random.uniform(key, (args.local_num_envs, args.num_steps+5+5)),
            policies = jax.random.uniform(key, (args.local_num_envs, args.num_steps+5+5, args.num_actions)),
            rewards = jax.random.uniform(key, (args.local_num_envs, args.num_steps+5+5)),
            dones = jax.random.uniform(key, (args.local_num_envs, args.num_steps+5+5))
        )
        replay_buffer.add(b_game_history, b_priorities)
        del b_game_history, b_priorities


    # sample from replay buffer
    batch = replay_buffer.sample(buffer_key)


    # 10 game histories of 100 envs and 500 steps = 6.87 GB
    # 1000 game histories of 500 steps = 6.87 GB
    # x 50
    # 50,000 game histories of 500 steps = 343.5 GB