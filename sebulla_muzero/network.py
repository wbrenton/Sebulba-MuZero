import jax
import jax.numpy as jnp
import haiku as hk
import chex
from flax import struct

from utils import make_action_encoding_fn, make_categorical_representation_fns

nonlinearity = jax.nn.relu

class ResNetv2Block(hk.Module):

    def __init__(self, channels, name, use_projection=False):
        block_name ='resnetv2_block_linear_' + name
        super().__init__(name=block_name)
        
        self.use_projection = use_projection
        if use_projection:
            self.projection = hk.Conv2D(channels, kernel_shape=3)
        
        conv_0 = hk.Conv2D(channels, kernel_shape=3)
        ln_0 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        conv_1 = hk.Conv2D(channels, kernel_shape=3)
        ln_1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        self.layers = ((conv_0, ln_0), (conv_1, ln_1))

    def __call__(self, input):
        x = shortcut = input
        
        if self.use_projection:
            shortcut = self.projection(shortcut)

        for (conv, layer_norm) in (self.layers):
            x = layer_norm(x)
            x = nonlinearity(x)
            x = conv(x)

        return x + shortcut

def make_tower(num_blocks, channels, use_projection=False, tower_id=None):
    tower = []
    for i in range(num_blocks):
        block_name = tower_id + str(i) if tower_id else str(i)
        
        if use_projection and i == 0:
            block = ResNetv2Block(channels, name=block_name, use_projection=True)
        else:
            block = ResNetv2Block(channels, name=block_name)
        tower.append(block)
        
    return hk.Sequential(tower)

class RepresentationNetwork(hk.Module):
    """
    Maps input observation of shape (batch_dim, 96, 96, 128)
        - 96x96 atari resolution
        - 128 channels composed of (96 + 32)
            - 32 history frames of 3 colors each = 96 channels
            - 32 actions broadcasted to (96, 96) 
    Schrittwieser et al., 2020, Appendix F
    """
    def __init__(self, name='h'):
        super().__init__(name=name)

        self.conv_0 = hk.Conv2D(128, kernel_shape=3, stride=2) # (48, 48)
        self.tower_0 = make_tower(2, 128, tower_id='a')
        
        self.conv_1 = hk.Conv2D(256, kernel_shape=3, stride=2) # (24, 24)
        self.tower_1 = make_tower(3, 256, tower_id='b')

        
        self.pooling_2 = hk.AvgPool(window_shape=(12, 12), strides=2, padding="SAME") # (12, 12)
        self.tower_2 = make_tower(3, 256, tower_id='c')
        
        self.pooling_final = hk.AvgPool(window_shape=(6, 6), strides=2, padding="SAME") # (6, 6)

    def __call__(self, input):
        # assert input.shape == (batch_dim, 96, 96, 128)
        x = jnp.transpose(input, (0, 2, 3, 1))
        x = x / (255.0)
        x = self.conv_0(x)
        x = self.tower_0(x)
        x = self.conv_1(x)
        x = self.tower_1(x)
        x = self.pooling_2(x)
        x = self.tower_2(x)
        embedding = self.pooling_final(x)
        # assert shape = (batch_dim, 6, 6, 256)
        return embedding

class DynamicsNetwork(hk.Module):
    """
    Maps embedding of shape (batch_dim, 6, 6, 256) and action of shape (batch_dim, 6, 6)
    Stacked along channel dimension to (batch_dim, 6, 6, 257)

    """
    def __init__(self, num_blocks, num_hiddens, support_size, action_encoding_fn, name='g'):
        super().__init__(name=name)
        self.torso = make_tower(num_blocks, num_hiddens, use_projection=True)
        self.reward_head = hk.Sequential([hk.Conv2D(16, kernel_shape=1), nonlinearity,
                                          hk.Flatten(-3), hk.Linear(support_size)]) # (AlphaGoZero) Silver et al., 2016 Methods: Neural Network Architecture NOTE: AlphaGoZero uses 2

        self.action_encoding_fn = action_encoding_fn

    def __call__(self, embedding, action):
        action_encoding = self.action_encoding_fn(action)
        action_encoding = jnp.expand_dims(action_encoding, axis=-1)
        x = jnp.concatenate((embedding, action_encoding), axis=-1)

        next_embedding = self.torso(x)
        reward = self.reward_head(next_embedding)

        return next_embedding, reward

class PredictionNetwork(hk.Module):
    """ 
    Maps embedding of shape (batch_dim, 6, 6, 256) to policy and value

    "The prediction function uses the same architecture as AlphaZero:
    one or two convolutional layers that preserve the resolution but reduce the number of planes,
    followed by a fully connected layer to the size of the output." - Schrittwieser et al., 2020, Section 4

    Couldn't find network architecture in AlphaZero paper, but current is per AlphaGoZero Paper.
    """
    def __init__(self, support_size, num_actions, name='f'):
        super().__init__(name=name)
        # NOTE: may need to make this larger
        self.value_head = hk.Sequential([hk.Conv2D(1, kernel_shape=1), nonlinearity,
                                         hk.Flatten(-3), hk.Linear(support_size)])
        self.policy_head = hk.Sequential([hk.Conv2D(2, kernel_shape=1), nonlinearity,
                                         hk.Flatten(-3), hk.Linear(num_actions)])

    def __call__(self, embedding):
        value = self.value_head(embedding)
        policy = self.policy_head(embedding)

        return value, policy

def make_muzero_network(
    support_size,
    num_actions,
    categorical_to_scalar,
    args,
    tiled_action_encoding_fn,
    bias_plane_action_encoding_fn,
    downscale_hiddens=1,
    downscale_blocks=1):
    """
    Creates a MuZero network composed of:
        h(x) Representation Network
            - maps observations to an embedding
        g(x) Dynamics Network
            - maps embedding and action to next embedding and reward
        f(x) Prediction Network
            - maps embedding to policy and value
    """
    asserts = lambda x: type(x) == int and x > 0
    assert asserts(downscale_hiddens) and asserts(downscale_blocks), "downscale_hiddens and downscale_blocks must be positive integers"

    num_blocks = 10 // downscale_blocks
    num_hiddens = 256 // downscale_hiddens

    def fn():
        representation_net = RepresentationNetwork()
        dynamics_net = DynamicsNetwork(num_blocks, num_hiddens, support_size, tiled_action_encoding_fn)
        prediction_net = PredictionNetwork(support_size, num_actions)

        # NOTE: batch size need to be dynamic to accomodate inference and training
        def make_initial_encoding(obs, action):
            B, T, C, H, W = obs.shape
            missing_frames = args.num_stacked_frames - T
            missing_action_frames = args.num_stacked_frames - action.shape[1]
            if missing_frames:
                missing_obs = jnp.zeros((B, missing_frames, C, H, W), dtype=jnp.float32)
                obs = jnp.concatenate([missing_obs, obs], axis=1)

            if missing_action_frames:
                # if len(action.shape) == 2:
                #     missing_action_frames -= action.shape[-1]
                missing_action = jnp.zeros((B, missing_action_frames), dtype=jnp.float32) # obs is availible before action
                action = jnp.concatenate([missing_action, action], axis=1)

            obs = obs[:, :, :3] # TODO: ask costa about 4 channel 
            obs = obs.reshape(B, args.num_stacked_frames*3, H, W) # C
            action = bias_plane_action_encoding_fn(action) # TODO: need to add normalization to this (action index / number of actions)
            encoding = jnp.concatenate([obs, action], axis=1)
            assert encoding.shape[1:] == (128, 84, 84)
            return encoding 

        def initial_inference(observations, actions, scalar=False):
            input = make_initial_encoding(observations, actions)
            embedding = representation_net(input)
            value, policy = prediction_net(embedding)
            value = categorical_to_scalar(value) if scalar else value
            return embedding, value, policy

        def recurrent_inference(embedding, action, scalar=False):
            next_embedding, reward = dynamics_net(embedding, action)
            value, policy = prediction_net(next_embedding)
            reward = categorical_to_scalar(reward) if scalar else reward
            value = categorical_to_scalar(value) if scalar else value
            return embedding, reward, value, policy

        def init(observation, actions):
            """ This is only used to initialize the params. Never for inference. """
            chex.assert_rank([observation, actions], [5, 2]) 
            
            embedding, _, _ = initial_inference(observation, actions)
            dummy_action = actions[:, -1].squeeze()
            next_embedding, reward, value, policy = recurrent_inference(embedding, dummy_action)

            return NetworkOutput(embedding=embedding,
                                next_embedding=next_embedding,
                                reward=reward,
                                value=value,
                                policy=policy)

        return init, (initial_inference, recurrent_inference)

    return hk.without_apply_rng(hk.multi_transform(fn))

@struct.dataclass
class NetworkOutput:
    embedding: jnp.ndarray
    next_embedding: jnp.ndarray
    reward: jnp.ndarray
    value: jnp.ndarray
    policy: jnp.ndarray
    
@struct.dataclass
class NetworkApplys:
    initial_inference: callable
    recurrent_inference: callable
