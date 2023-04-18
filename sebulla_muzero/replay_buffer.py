import time
import queue
import threading
from typing import Any, Union 

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

import pickle
import zstandard as zstd


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
    
class GameHistory:
  def __init__(self, config=None):
    self.config = config
    self.max_length = config.sequence_length
    
    self.prefix_length = config.num_stacked_frames
    self.suffix_length = config.num_unroll_steps + config.td_steps

    self.observations = []
    self.actions = []
    self.value_targets = []
    self.policy_targets = []
    self.rewards = []
    self.dones = []

  # TODO: need a good method for managing features before and after the scope of this game history
  def init(self, obs, action, value_target, policy_target, reward, done):
    self.observations.extend(obs)
    self.actions.extend(action)
    self.value_targets.extend(value_target)
    self.policy_targets.extend(policy_target)
    self.rewards.extend(reward)
    self.dones.extend(done)

  def make_features(self, step_idx):
    # TODO: need to look at dones and zero out value predictions after done
    """
    Given a step index returns the corresponding features for that game history.
    Real Index:
      [] indicates valid step_idx range
      -prefix_length ...... [0 ....... sequence_length] ........ + suffix_length
    Prefixed features:
      range: -prefix_length ...... [0 ....... sequence_length]
      features: obs, action
    Suffixed features:
      range: [0 ....... sequence_length] ........ + suffix_length
      features: value, policy, reward
      
    Note: prefixed features need to be shifted by prefix_length to align with suffixed features.
          That is the job of prefix_step_idx.
    """
    K = self.config.num_unroll_steps
    # prefixed
    prefix_step_idx = step_idx + self.prefix_length
    obs = np.asarray(self.observations[prefix_step_idx - self.prefix_length : prefix_step_idx])
    action = np.asarray(self.actions[prefix_step_idx - self.prefix_length : prefix_step_idx+K])
    # suffixed
    value = np.asarray(self.value_targets[step_idx:step_idx+K])
    policy = np.asarray(self.policy_targets[step_idx:step_idx+K])
    rewards = np.asarray(self.rewards[step_idx:step_idx+K])
    return obs, action, value, policy, rewards

  def _make_obs_stack(self):
    pass
  
  def _prevent_episode_boundry_crossing(self):
    pass
  
  def assert_match(self, priority):
    return len(self) == len(priority) - self.prefix_length - self.suffix_length
  
  def __len__(self):
    prefixed_vars_len = len(self.observations) - self.prefix_length
    suffixed_vars_len = len(self.dones) - self.suffix_length
    full_vars_len = len(self.actions) - self.prefix_length - self.suffix_length
    assert prefixed_vars_len == suffixed_vars_len == full_vars_len, "GameHistory variables must be the same length."
    return prefixed_vars_len

class ReplayBuffer:
  """
  Replay buffer for storing GameHistory objects and their associated priorities.
  
  Due to large memory requirements, zstd compression is used to compress each GameHistory object in the buffer.
  They are decompressed on the fly when sampled.
  
  Compression Factor: 
    Uncompressed GameHistory
    Compressed GameHistory

  This class is largely a port of Efficient Zero Ye et al. (2020) https://arxiv.org/pdf/2111.00210.pdf
  """

  def __init__(self, max_size: int, batch_size: int, config) -> None:
    self.args = config.args
    self.batch_size = batch_size
    self.max_size = max_size
    self.base_index = 0
    self.collected_count = 0
    self.alpha = 1
    self.beta = 1

    self.buffer: list[GameHistory] = []
    self.priorities: Union[None, np.ndarray] = None
    self.game_lookup: list[list[tuple]] = [] # each tuple is (game_index, step_index)

    self.game_steps_seen = 0
    self.game_steps_to_start = batch_size * 1 # how many batches do you want to collect before starting to sample

    self.zstd_compressor = zstd.ZstdCompressor()
    self.zstd_decompressor = zstd.ZstdDecompressor()
    
    self.scalar_to_categorical = jax.jit(config.scalar_to_categorical, backend="cpu")

  def put_games(self, games: tuple[GameHistory, np.ndarray]) -> None:
    """
    Add a list of games (tuples of game_history and priorities) to the replay buffer.
    """
    for game_history, priority in games:
      self._put_game(game_history, priority)


  def _put_game(self, game: GameHistory, priorities: np.ndarray) -> None:
    """
    Add a game_history and corresponding priorities to the replay buffer.
    """
    #assert game_history.assert_match(priorities), "GameHistory and priorities must be the same length."
    assert len(game) == len(priorities), "GameHistory and priorities must be the same length"

    bytes = pickle.dumps(game)
    compressed_game = self.zstd_compressor.compress(bytes)
    self.buffer.append(compressed_game)
    self.priorities = priorities if not self.game_steps_seen else np.concatenate([self.priorities, priorities])
    self.game_lookup += [(self.base_index + len(self.buffer) - 1, step_index) for step_index in range(len(game))]
    self.game_steps_seen += len(game) * self.args.sequence_length

  def sample(self):
    """
    Sample a batch according to PER and calculate the importance sampling weights to correct for bias in loss calculation.

    i.e. 
    """
    if self.game_steps_to_start > self.game_steps_seen:
      return None 
    universe = len(self.priorities)
    sample_probs = np.power(self.priorities, self.alpha)
    sample_probs = np.divide(sample_probs, np.sum(sample_probs))
    sampled_indices = np.random.choice(universe, size=self.batch_size, p=sample_probs, replace=False)
    weights = (1 / universe) * np.reciprocal(sample_probs[sampled_indices])
    weights = np.power(weights, self.beta)

    # create decompression efficient index 'smart_indicies' {game_index: [step_indices]}
    # to prevent redundant decompression of the same game_history
    smart_indicies = {}
    for index in sampled_indices:
      game_index, step_index = self.game_lookup[index]
      if game_index not in smart_indicies:
        smart_indicies[game_index] = []
      smart_indicies[game_index].append(step_index)

    observations = []
    actions = []
    values = []
    policies = []
    rewards = []
    for game_index in smart_indicies.keys():
      bytes = self.zstd_decompressor.decompress(self.buffer[game_index])
      game_history = pickle.loads(bytes)
      for step_index in smart_indicies[game_index]:
        obs, action, value, policy, reward = game_history.make_features(step_index)
        observations.append(obs)
        actions.append(action)
        values.append(value)
        policies.append(policy)
        rewards.append(reward)

    observations = np.asarray(observations)
    actions = np.asarray(actions)
    values = np.asarray(values)
    policies = np.asarray(policies)
    rewards = np.asarray(rewards)
    
    values = self.scalar_to_categorical(values)
    rewards = self.scalar_to_categorical(rewards)
    
    batch = Batch(
      observation=observations,
      actions=actions,
      value=values,
      policy=policies,
      reward=rewards
    )
    
    return batch, weights, sampled_indices

  def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
    """
    Update the priorities of the replay buffer.
    """
    self.priorities[indices] = priorities


##############################################################################################################
  
def start_replay_buffer_manager(rollout_queue, batch_queue, config):
  """
  Conviences function for starting the replay buffer manager.
  Responsible for managing the communication between the actor and learner processes
  """
  args = config.args

  def preprocess_data(rollout):
    """Convert a batched rollout to a list of GameHistory objects."""
    
    game_histories = []
    for i in range(args.local_num_envs):
      game_history = GameHistory(args)
      game_history.init(
          obs=rollout.obs[i],
          action=rollout.actions[i],
          value_target=rollout.value_targets[i],
          policy_target=rollout.policy_targets[i],
          reward=rollout.rewards[i],
          done=rollout.dones[i],
          )
      game_histories.append((game_history, rollout.priorities[i]))
    return game_histories

  def rollout_queue_to_replay_buffer(replay_buffer, queue, rollout_queue):
        """ Thread dedicated to processing rollouts and inserting data in the replay buffer."""
        while True:
          rollouts = rollout_queue.get()
          game_histories = preprocess_data(rollouts)
          replay_buffer = queue.get()
          replay_buffer.put_games(game_histories)
          queue.put(replay_buffer)
    
  def replay_buffer_to_batch_queue(replay_buffer, queue, batch_queue):
      """Thread dedicated to sampling data from the replay buffer and populating the batch queue."""
      while True:
          replay_buffer = queue.get()
          batch = replay_buffer.sample()
          if batch is None: 
            queue.put(replay_buffer)
            time.sleep(1)
            continue
          queue.put(replay_buffer)
          batch_queue.put(batch)

  replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, config)

  access_queue = queue.Queue(1)
  access_queue.put(replay_buffer)

  threading.Thread(target=rollout_queue_to_replay_buffer, args=(replay_buffer, access_queue, rollout_queue)).start()
  threading.Thread(target=replay_buffer_to_batch_queue, args=(replay_buffer, access_queue, batch_queue)).start()