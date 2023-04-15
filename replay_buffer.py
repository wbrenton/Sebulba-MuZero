from flax import struct

import numpy as np
from typing import Union

import zstandard as zstd


class GameHistory:
  def __init__(self, config=None, max_length=100):
    self.config = config
    self.max_length = max_length

    self.observations = []
    self.actions = []
    self.search_values = []
    self.search_policies = []
    self.rewards = []
    self.dones = []

  # TODO: need a good method for managing features before and after the scope of this game history
  def init(self, obs, action, search_value, search_policy, reward, done):
    self.observations.extend(obs)
    self.actions.extend(action)
    self.search_values.extend(search_value)
    self.search_policies.extend(search_policy)
    self.rewards.extend(reward)
    self.dones.extend(done)
    
  def make_features(self):
    pass
  
  def _make_obs_stack(self):
    pass
  
  def _prevent_episode_boundry_crossing(self):
    pass

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
  
  def __init__(self, max_size: int, batch_size: int) -> None:
    self.batch_size = batch_size
    self.max_size = max_size
    self.base_index = 0
    self.collected_count = 0
    self.alpha = 1
    self.beta = 1
    
    self.buffer: list[GameHistory] = []
    self.priorities: Union[None, np.ndarray] = None
    self.game_lookup: list[list[tuple]] = [] # each tuple is (game_index, step_index)
    
    self.zstd_compressor = zstd.ZstdCompressor()
    self.zstd_decompressor = zstd.ZstdDecompressor()
    
  def put_games(self, games: tuple[GameHistory, np.ndarray]) -> None:
    """
    Add a list of games (tuples of game_history and priorities) to the replay buffer.
    """
    for game_histories, priorities in games:
      self._put(game_histories, priorities)


  def _put_game(self, game_history: GameHistory, priorities: np.ndarray) -> None:
    """
    Add a game_history and corresponding priorities to the replay buffer.
    """
    assert len(game_history) == len(priorities), "GameHistory and priorities must be the same length."
    
    compressed_game = self.zstd_compressor.compress(game_history)
    self.buffer.append(compressed_game)
    self.priorities = np.concatenate([self.priorities, priorities])
    self.game_lookup += [(self.base_index + len(self.buffer) - 1, step_index) for step_index in range(len(game_history))]
    
  def sample(self):
    """
    Sample a batch according to PER and calculate the importance sampling weights to correct for bias in loss calculation.
    
    i.e. 
    """
    universe = len(self.priorities)
    sample_probs = np.power(self.priorities, self.alpha)
    sample_probs = np.divide(sample_probs, np.sum(sample_probs))
    sampled_indices = np.random.choice(universe, size=self.batch_size, p=sample_probs, replace=False)
    weights = np.product(np.reciprocal(universe), np.reciprocal(sample_probs[sampled_indices]))
    weights = np.power(weights, self.beta)
    
    # sort the sampled indices by game_index
    sorted_indices = sorted(sampled_indices, key=lambda index: self.game_lookup[index][0])
    
    # iterate through the sorted_indices and create decompression efficient index 'smart_indicies' {game_index: [step_indices]}
    # to prevent redundant decompression of the same game_history
    smart_indicies = {}
    for index in sorted_indices:
      game_index, step_index = self.game_lookup[index]
      if game_index not in smart_indicies:
        smart_indicies[game_index] = []
      smart_indicies[game_index].append(step_index)
    
    
    batch = []
    for game_index in smart_indicies.keys():
      game_history = self.zstd_decompressor.decompress(self.buffer[game_index])
      for step_index in smart_indicies[game_index]:
        features = game_history.make_features(game_history, step_index)
        batch.append(features)
    
    # TODO: create Batch object
    
    
    return batch, weights, sampled_indices

  def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
    """
    Update the priorities of the replay buffer.
    """
    self.priorities[indices] = priorities
