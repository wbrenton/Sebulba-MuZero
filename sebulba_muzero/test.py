import jax
import jax.numpy as jnp

from network import make_muzero_network, NetworkApplys 

batch_size = 32
support_size = 601
num_actions = 4

muzero_network = make_muzero_network(support_size, num_actions=4)
apply = NetworkApplys(*muzero_network.apply)

key = jax.random.PRNGKey(0)
observation = jnp.ones((32, 4, 96, 96))
actions = jnp.ones(32)
params = muzero_network.init(key, observation, actions)

network_output = apply.network(params, observation, actions)

assert network_output.embedding.shape == (batch_size, 6, 6, 256)
assert network_output.next_embedding.shape == (batch_size, 6, 6, 256)
assert network_output.reward.shape == (batch_size, support_size)
assert network_output.value.shape == (batch_size, support_size)
assert network_output.policy.shape == (batch_size, num_actions)