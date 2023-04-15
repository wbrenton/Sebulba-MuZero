import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple
from jax.random import PRNGKey
from jax.flatten_util import ravel_pytree
from functools import partial

import queue
import threading

#from learner import make_prepare_data_fn

State = Any
Sample = Any

@struct.dataclass
class Trajectory:
    observation: jnp.ndarray
    action: jnp.ndarray
    value_target: jnp.ndarray
    policy_target: jnp.ndarray
    reward_target: jnp.ndarray
    priority: jnp.ndarray

@struct.dataclass
class Batch:
    observation: jnp.ndarray
    actions: jnp.ndarray
    value: jnp.ndarray
    policy: jnp.ndarray
    reward: jnp.ndarray

@struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""
    data: jnp.ndarray
    current_position: jnp.ndarray
    current_size: jnp.ndarray
    added_samples: jnp.ndarray
    key: PRNGKey

class ReplayBuffer:
  """
  Priotized experience replay buffer.

  Modified port of brax.training.replay_buffers
  https://github.com/google/brax/blob/b373f5a45e62189a4a260131c17b10181ccda96a/brax/training/replay_buffers.py

  """

  def __init__(self, max_replay_size: int, dummy_data_sample: Sample,
                sample_batch_size: int) -> State:
    """Init the replay buffer."""
    # takes in (batch, time, ...) outputs (batch, time * ...)
    self._flatten_fn = jax.vmap(lambda x: ravel_pytree(x)[0], axis_name="time")

    # takes in (time, ...) outputs (time * ...)
    dummy_flatten, self._unflatten_fn = ravel_pytree(dummy_data_sample)
    # takes in (batch, time, ...) outputs (batch, time * ...)
    self._unflatten_fn = jax.vmap(self._unflatten_fn, axis_name="time")
    data_size = len(dummy_flatten)

    self._data_shape = (max_replay_size, data_size)
    self._data_dtype = dummy_flatten.dtype
    self._sample_batch_size = sample_batch_size

  def init(self, key: PRNGKey) -> ReplayBufferState:
    return ReplayBufferState(
        data=jnp.zeros(self._data_shape, self._data_dtype),
        current_size=jnp.zeros((), jnp.int32),
        current_position=jnp.zeros((), jnp.int32),
        added_samples=jnp.zeros((), jnp.int32),
        key=key)

  def insert(self, buffer_state: State, samples: Sample) -> State:
    """Insert data in the replay buffer.

    Args:
      buffer_state: Buffer state
      samples: Sample to insert with a leading batch size.

    Returns:
      New buffer state.
    """
    if buffer_state.data.shape != self._data_shape:
      raise ValueError(
          f'buffer_state.data.shape ({buffer_state.data.shape}) '
          f'doesn\'t match the expected value ({self._data_shape})')

    update = self._flatten_fn(samples)
    data = buffer_state.data

    # Make sure update is not larger than the maximum replay size.
    if len(update) > len(data):
      raise ValueError(
          'Trying to insert a batch of samples larger than the maximum replay '
          f'size. num_samples: {len(update)}, max replay size {len(data)}')

    # If needed, roll the buffer to make sure there's enough space to fit
    # `update` after the current position.
    position = buffer_state.current_position
    roll = jnp.minimum(0, len(data) - position - len(update))
    data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0),
                        lambda: data)
    position = position + roll

    # Update the buffer and the control numbers.
    data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
    position = (position + len(update)) % len(data)
    size = jnp.minimum(buffer_state.current_size + len(update), len(data))
    added_samples = buffer_state.added_samples + len(update)

    return buffer_state.replace(
        data=data, current_position=position, current_size=size, added_samples=added_samples)

  def sample(self, buffer_state: State) -> Tuple[State, Sample]:
    """Sample a batch of data according to prioritized experience replay."""
    key, sample_key = jax.random.split(buffer_state.key)
    buffer = self._unflatten_fn(buffer_state.data)
    flat_idx = jnp.argsort(buffer.priority.flatten())[-self._sample_batch_size:]
    batch_idx, time_idx = jnp.unravel_index(flat_idx, buffer.priority.shape)

    def _index(feature, batch_idx, time_idx, size):

      @partial(jax.vmap)
      def batched_slice(sample, idx):
        # NOTE: dynamic_slice_in_dim say's it takes arrays in documentation but throws error "start_index must be a scalar"
        return jax.lax.dynamic_slice_in_dim(sample, idx, size)

      start_index = jnp.maximum(time_idx - size, 0) # don't wrap around
      sample = feature.at[batch_idx].get()
      stack = batched_slice(sample, start_index)

      return stack

    return Batch(
      observation=_index(buffer.observation, batch_idx, time_idx, 2),
      actions=_index(buffer.action, batch_idx, time_idx, 1),
      value=_index(buffer.value_target, batch_idx, time_idx, 1),
      policy=_index(buffer.policy_target, batch_idx, time_idx, 1),
      reward=_index(buffer.reward_target, batch_idx, time_idx, 1),
    ), buffer_state.replace(key=sample_key)

  def size(self, buffer_state: State) -> int:
    """Total amount of elements that are sampleable."""
    
    return buffer_state.current_size
  
def start_replay_buffer_manager(rollout_queue, batch_queue, args, key):
  """
  Conviences function for starting the replay buffer manager.
  Responsible for managing the communication between the actor and learner processes
  """

  prepare_data_fn = make_prepare_data_fn(args, learner_devices, scla)

  def rollout_queue_to_replay_buffer(replay_buffer, buffer_state_queue, rollout_queue):
        """ Thread dedicated to processing rollouts and inserting data in the replay buffer."""
        while True:
          # rollouts are put in the queue every num_steps == rollout_length
          # TODO: plan out logic to get preceding and following observations
          (
              global_step,
              obs,
              values,
              actions,
              mcts_policies,
              rewards,
              dones,
              rewards,
              params_queue_get_time
          ) = rollout_queue.get()

          trajectory = prepare_data_fn(
              obs,
              dones,
              values,
              actions,
              mcts_policies,
              rewards
          )
          buffer_state = buffer_state_queue.get()
          buffer_state = replay_buffer.insert(buffer_state, trajectory)
          buffer_state_queue.put(buffer_state)
    
  def replay_buffer_to_batch_queue(replay_buffer, buffer_state_queue, batch_queue):
      """Thread dedicated to sampling data from the replay buffer and populating the batch queue."""
      while True:
          buffer_state = buffer_state_queue.get()
          buffer_state = replay_buffer.sample(buffer_state)
          buffer_state_queue.put(buffer_state)
          batch_queue.put(buffer_state)

  unstacked_observation = args.obs_sample[-3:, :, :]
  init_trajectory = Trajectory(
    observation=jnp.zeros((args.num_steps, unstacked_observation), dtype=jnp.float32),
    action=jnp.zeros((args.num_steps,), dtype=jnp.float32),
    value_target=jnp.zeros((args.num_steps,), dtype=jnp.float32),
    policy_target=jnp.zeros((args.num_steps, args.num_actions), dtype=jnp.float32),
    reward_target=jnp.zeros((args.num_steps,), dtype=jnp.float32)
  )

  replay_buffer = ReplayBuffer(args.max_replay_size, init_trajectory, args.batch_size)
  buffer_state = replay_buffer.init(key)
  
  buffer_state_queue = queue.Queue(1)
  buffer_state_queue.put(buffer_state)
  
  threading.Thread(target=rollout_queue_to_replay_buffer, args=(replay_buffer, buffer_state_queue, rollout_queue)).start()
  threading.Thread(target=replay_buffer_to_batch_queue, args=(replay_buffer, buffer_state_queue, batch_queue)).start()