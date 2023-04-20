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
class Batch:
    observation: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    policy: jnp.ndarray
    reward: jnp.ndarray
    weight: jnp.ndarray
    index: jnp.ndarray


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

@struct.dataclass
class ExperienceBuffer:
  pass

class ReplayBuffer:
  """
  Replay buffer for storing GameHistory objects and their associated priorities.
  
  Due to large memory requirements, zstd compression is used to compress each GameHistory object in the buffer.
  They are decompressed on the fly when sampled.
  
  Compression Factor: # TODO
    Uncompressed GameHistory
    Compressed GameHistory

  This class is largely a port of Efficient Zero Ye et al. (2020) https://arxiv.org/pdf/2111.00210.pdf
  """

  def __init__(self, learner_devices, max_size: int, batch_size: int, config) -> None:
    self.learner_device = learner_devices[0]
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

    self.samples_calls = 0
    self.game_steps_seen = 0
    self.game_steps_to_start = batch_size * 2 # how many batches do you want to collect before starting to sample

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
    self.game_steps_seen += len(game)

  def sample(self, writer, num_batches=10):
    """
    Sample a batch according to PER and calculate the importance sampling weights to correct for bias in loss calculation.

    i.e.
    """
    if self.batch_size * num_batches > self.game_steps_seen:
      return None 
    self.samples_calls += 1
    make_indicies_time_start = time.time()
    N = len(self.priorities)
    sample_probs = self.priorities**self.alpha
    sample_probs /= np.sum(sample_probs)
    sampled_indices = np.random.choice(N, size=self.batch_size*num_batches, p=sample_probs, replace=False)
    weights = ((1 / N) * (1 / sample_probs[sampled_indices])) ** self.beta
    make_indicies_time = time.time() - make_indicies_time_start
    writer.add_scalar("replay_buffer/make_indicies_time", make_indicies_time, self.samples_calls)

    # create decompression efficient index 'smart_indicies' {game_index: [step_indices]}
    # to prevent redundant decompression of the same game_history
    make_smart_index_time_start = time.time()
    smart_indicies = {}
    for index in sampled_indices:
      game_index, step_index = self.game_lookup[index]
      if game_index not in smart_indicies:
        smart_indicies[game_index] = []
      smart_indicies[game_index].append(step_index)
    make_smart_index_time = time.time() - make_smart_index_time_start
    writer.add_scalar("replay_buffer/make_smart_index_time", make_smart_index_time, self.samples_calls)

    # 50,000 rollouts
    # 200 steps per rollout

    # 50 compressed rollouts
    # 1000 batched rollouts

    loop_over_games_time_start = time.time()
    observations = []
    actions = []
    values = []
    policies = []
    rewards = []
    for game_index in smart_indicies.keys():
      # TODO: recompress game_history
      bytes = self.zstd_decompressor.decompress(self.buffer[game_index])
      game_history = pickle.loads(bytes)
      for step_index in smart_indicies[game_index]:
        obs, action, value, policy, reward = game_history.make_features(step_index)
        observations.append(obs)
        actions.append(action)
        values.append(value)
        policies.append(policy)
        rewards.append(reward)
    loop_over_games_time = time.time() - loop_over_games_time_start
    writer.add_scalar("replay_buffer/loop_over_games_time", loop_over_games_time, self.samples_calls)

    stack_time_start = time.time()
    observations = jnp.asarray(observations)
    actions = jnp.asarray(actions)
    values = jnp.asarray(values)
    policies = jnp.asarray(policies)
    rewards = jnp.asarray(rewards)
    stack_time = time.time() - stack_time_start
    writer.add_scalar("replay_buffer/stack_time", stack_time, self.samples_calls)

    to_categorical_time_start = time.time()
    values = self.scalar_to_categorical(values)
    rewards = self.scalar_to_categorical(rewards)
    to_categorical_time = time.time() - to_categorical_time_start
    writer.add_scalar("replay_buffer/to_categorical_time", to_categorical_time, self.samples_calls)

    make_batch_time_start = time.time()
    batch = Batch(
      observation=observations,
      action=actions,
      value=values,
      policy=policies,
      reward=rewards,
      weight=weights,
      index=sampled_indices,
    )
    make_batch_time = time.time() - make_batch_time_start
    writer.add_scalar("replay_buffer/make_batch_time", make_batch_time, self.samples_calls)

    # split batch * num_batches into their own batch objects
    batch = jax.tree_map(lambda x: np.array_split(x, num_batches), batch)

    split_batch_time_start = time.time()
    batches = []
    for idx in range(num_batches):
      b = Batch(
        observation=batch.observation[idx],
        action=batch.action[idx],
        value=batch.value[idx],
        policy=batch.policy[idx],
        reward=batch.reward[idx],
        weight=batch.weight[idx],
        index=batch.index[idx],
      )
      batches.append(b)
    split_batch_time = time.time() - split_batch_time_start
    writer.add_scalar("replay_buffer/split_batch_time", split_batch_time, self.samples_calls)

    return batches

  def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
    """
    Update the priorities of the replay buffer.
    """
    self.priorities[indices] = priorities

##############################################################################################################

def start_replay_buffer_manager(learner_devices, rollout_queue, batch_queue, config, writer):
  """
  Conviences function for starting the replay buffer manager.
  Responsible for managing the communication between the actor and learner processes
  """
  args = config.args

# TODO: this is making a copy, consider making game histories a dataclass
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

  def rollout_queue_to_replay_buffer(replay_buffer, queue, rollout_queue, writer):
        """ Thread dedicated to processing rollouts and inserting data in the replay buffer."""
        counter = 0
        while True:
          rollout_queue_get_time_start = time.time()
          rollouts = rollout_queue.get()
          rollout_queue_get_time = time.time() - rollout_queue_get_time_start

          preprocess_data_time_start = time.time()
          game_histories = preprocess_data(rollouts)
          preprocess_data_time = time.time() - preprocess_data_time_start
          
          buffer_get_time_start = time.time()
          replay_buffer = queue.get()
          buffer_get_time = time.time() - buffer_get_time_start

          put_games_time_start = time.time()
          replay_buffer.put_games(game_histories)
          put_games_time = time.time() - put_games_time_start
          game_steps_seen = replay_buffer.game_steps_seen

          queue.put(replay_buffer)

          counter += 1
          writer.add_scalar('stats/rollout_queue_to_replay_buffer/rollout_queue_get_time', rollout_queue_get_time, counter)
          writer.add_scalar('stats/rollout_queue_to_replay_buffer/preprocess_data_time', preprocess_data_time, counter)
          writer.add_scalar('stats/rollout_queue_to_replay_buffer/buffer_get_time', buffer_get_time, counter)
          writer.add_scalar('stats/rollout_queue_to_replay_buffer/put_games_time', put_games_time, counter)
          writer.add_scalar('stats/rollout_queue_to_replay_buffer/game_steps_seen', game_steps_seen, counter)

  def replay_buffer_to_batch_queue(replay_buffer, queue, batch_queue, writer):
      """Thread dedicated to sampling data from the replay buffer and populating the batch queue."""
      counter = 0
      while True:
          buffer_get_time_start = time.time()
          replay_buffer = queue.get()
          buffer_get_time = time.time() - buffer_get_time_start

          sample_time_start = time.time()
          batches = replay_buffer.sample(writer)
          sample_time = time.time() - sample_time_start

          queue.put(replay_buffer)
          if batches is not None:
            counter += 1
            for batch in batches:
              batch_queue.put(batch)

            writer.add_scalar('stats/replay_buffer_to_batch_queue/buffer_get_time', buffer_get_time, counter)
            writer.add_scalar('stats/replay_buffer_to_batch_queue/sample_time', sample_time, counter)
            continue
          
          # replay buffer is not ready, wait and try again
          time.sleep(1)

  replay_buffer = ReplayBuffer(learner_devices, args.buffer_size, args.batch_size, config)

  access_queue = queue.Queue(1)
  access_queue.put(replay_buffer)

  threading.Thread(target=rollout_queue_to_replay_buffer, args=(replay_buffer, access_queue, rollout_queue, writer)).start()
  threading.Thread(target=replay_buffer_to_batch_queue, args=(replay_buffer, access_queue, batch_queue, writer)).start()