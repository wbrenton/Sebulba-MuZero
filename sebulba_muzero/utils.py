import jax
import chex
import optax
import haiku as hk
from flax import struct
import jax.numpy as jnp
import rlax
from rlax._src import nonlinear_bellman

import gym

@struct.dataclass
class TrainState:
    params: hk.Params
    target_params: hk.Params
    opt_state: optax.OptState
    train_step: chex.Array
  
def min_max_normalize(x):
    min = jnp.min(x)
    max = jnp.max(x)
    return (x - min) / (max - min)

def softmax_temperature_fn(train_step, train_steps):
    """
    Temperature annealing function from Schrittwieser et al., 2020, Appendix F
        0-500k steps: 1
        500k-750k steps: 0.5
        750k-1M steps: 0.25
    """
    percent_complete = train_step / train_steps
    return jax.lax.cond(
        jnp.less_equal(
            percent_complete,
            0.5),
        lambda: 1.,
        lambda: jax.lax.select(
            jnp.less_equal(
                percent_complete,
                0.75),
            0.5,
            0.25)
        )

def make_action_encoding_fn(embeding_resolution, obs_resolution, num_actions):
    """
    recurrent_fn_action_encoder: 
        operates on a batch of actions of shape (batch_dim,)
    initial_fn_action_encoder:
        operates on a batch of actions over time of shape (batch_dim, time_dim)
    """

    def initial_action_encoder(scalar_action):
        return jnp.broadcast_to(scalar_action, (obs_resolution, obs_resolution)) / num_actions

    def recurrent_action_encoder(scalar_action):
        return jnp.broadcast_to(scalar_action, (embeding_resolution, embeding_resolution)) / num_actions

    return jax.vmap(jax.vmap(initial_action_encoder)), jax.vmap(recurrent_action_encoder)

def make_categorical_representation_fns(support_size):
    """
    Creates functions for mapping scalar->categorical and categorical->scalar representations
    Signed hyperbolic pair is h(x) in Schrittwieser et al., 2020, Appendix F
    $/phi$ in Schrittwieser et al., 2020, Appendix F
    """
    support_min_max = support_size - 1
    support_min = -(support_min_max / 2)
    support_max = support_min_max / 2
    
    tx = rlax.muzero_pair(num_bins=support_size,
                        min_value=support_min,
                        max_value=support_max,
                        tx=nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR) 

    def scalar_to_categorical(scalar):
        return tx.apply(scalar)

    def categorical_to_scalar(categorical):
        probability = jax.nn.softmax(categorical, axis=-1)
        return tx.apply_inv(probability)

    return scalar_to_categorical, categorical_to_scalar

def nonlinux_make_env(env_id, seed, num_envs, async_batch_size=1):
    """
    Used to clone make_env behvaior when not on a Linux machine (i.e. devloping locally)
    """

    class FacadeEnvPoolEnvironment:
        """
        Clone of envpools env for testing/devlopment on non-linux machine
        NOTE: Maps to Breakout-v5 only
        """

        def __init__(self, num_envs, async_batch_size):
            self.num_envs = num_envs
            self.async_batch_size = async_batch_size
            self.single_action_space = gym.spaces.Discrete(4)
            self.single_observation_space = gym.spaces.Box(low=0., high=255., shape=(32*3, 84, 84))
            
            self._max_episode_steps = 1000
            self.max_episode_steps = self._max_episode_steps
            self.config = self
            self.spec = self
            
        def config(self):
            return self
        
        def spec(self):
            return self
        
        def async_reset(self):
            return
        
        def recv(self):
            batch_size = self.async_batch_size
            next_obs = jnp.zeros((batch_size, *self.single_observation_space.shape))
            next_reward = jnp.zeros(batch_size)
            next_done = jnp.full(batch_size, False)
            info = {'env_id': jnp.arange(batch_size),
                    'elapsed_step': jnp.ones(batch_size) * 1000,
                    'terminated': next_done,
                    'reward': next_reward}
            return next_obs, next_reward, next_done, None, info

        def send(self, action, env_id):
            return

    return lambda : FacadeEnvPoolEnvironment(num_envs, async_batch_size)
    # lambda is required to mirror envpool api
    
# def tiled_encoding(action):
#     """Turns a scalar action into a ont-hot encoding tiled to (resolution, resolution)"""
#     one_hot = jax.nn.one_hot(action, num_actions)
#     reshape = jnp.reshape(one_hot, (2, 2))
#     return jnp.tile(reshape, (embeding_resolution // 2, embeding_resolution // 2))