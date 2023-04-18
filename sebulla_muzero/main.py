import platform
import argparse
import os
import random
import time
import uuid
from collections import deque
from distutils.util import strtobool
from functools import partial
from typing import Sequence
from types import SimpleNamespace

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.6"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
import queue
import threading

system = platform.system()
print(system)
if system == "Linux":
    import envpool
    from jax_smi import initialise_tracking
    initialise_tracking()
else:
    print("envpool is not supported on non Linux systems")
import gym
import jax
import jax.numpy as jnp
import numpy as np
import rlax
import flax
import optax
import haiku as hk
from tensorboardX import SummaryWriter

from utils import TrainState, nonlinux_make_env, make_categorical_representation_fns, make_action_encoding_fn
from network import make_muzero_network, NetworkApplys
from actor import make_rollout_fn, make_mcts_fn
from learner import make_prepare_data_fn, make_single_device_update
from replay_buffer import start_replay_buffer_manager

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--env-id", type=str, default="Breakout-v5",
        help="the id of the environment")
    parser.add_argument("--gray_scale", type=bool, default=False,
        help="the id of the environment")
    parser.add_argument("--total_trainsteps", type=int, default=1_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--local-num-envs", type=int, default=20,
        help="the number of parallel game environments")
    # parser.add_argument("--num-steps", type=int, default=1000,
    #     help="the number of steps to run in each environment per rollout")
    parser.add_argument("--batch-size", type=float, default=1024,
        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=10e-4,
        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--weight-decay", type=float, default=20e-5,
        help="Weight decay for Adam optimizer")
    parser.add_argument("--num-unroll-steps", type=int, default=5,
        help="the number of steps to unroll the model in training")
    parser.add_argument("--td-steps", type=int, default=5,
        help="the discount factor gamma")
    parser.add_argument("--gamma", type=float, default=0.997,
        help="the discount factor gamma")
    parser.add_argument("--num-simulations", type=float, default=50,
        help="number of mcts iterations per action selection")
    parser.add_argument("--support-size", type=int, default=601,
        help="coefficient of the value function")
    parser.add_argument("--num-stacked-frames", type=int, default=32,
        help="number of historical observation frames used to produce initial embedding")
    parser.add_argument("--obs-resolution", type=int, default=84,
        help="atari observation resolution")
    parser.add_argument("--embedding-resolution", type=int, default=6,
        help="learned embedding resolution")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the latent representation clipping coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--buffer-size", type=float, default=10,
        help="maximum number of sequences in the replay buffer")
    parser.add_argument("--sequence-length", type=float, default=500,
        help="maximum length of a sequence in the replay buffer")

    # resource managment
    parser.add_argument("--actor-device-ids", type=int, nargs="+", default=[0, 1],#, 1],
        help="the device ids that actor workers will use (currently only support 1 device)")
    parser.add_argument("--num-actor-threads", type=int, default=3,
        help="the number of actor threads to use per core")
    parser.add_argument("--learner-device-ids", type=int, nargs="+", default=[2, 3],
        help="the device ids that learner workers will use")
    parser.add_argument("--distributed", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to use `jax.distirbuted`")
    parser.add_argument("--profile", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to call block_until_ready() for profiling")
    parser.add_argument("--test-actor-learner-throughput", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to test actor-learner throughput by removing the actor-learner communication")

    args = parser.parse_args()
    args.num_steps = args.sequence_length
    args.channels_per_frame = 1 if args.gray_scale else 3
    return args


ATARI_MAX_FRAMES = int(
    108000 / 4 # from cleanba code
)

# TODO: change this to *args
def make_env_helper(system, env_id, seed, local_num_envs):
    if system == "Linux":
        return make_env(env_id, seed, local_num_envs)
    else:
        return nonlinux_make_env(env_id, seed, local_num_envs)

def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            stack_num=args.num_stacked_frames,
            gray_scale=args.gray_scale,
            episodic_life=True,  # Espeholt et al., 2018, Tab. G.1
            repeat_action_probability=0,  # Hessel et al., 2022 (Muesli) Tab. 10
            noop_max=30,  # Espeholt et al., 2018, Tab. C.1 "Up to 30 no-ops at the beginning of each episode."
            full_action_space=False,  # Espeholt et al., 2018, Appendix G., "Following related work, experts use game-specific action sets."
            max_episode_steps=ATARI_MAX_FRAMES,  # Hessel et al. 2018 (Rainbow DQN), Table 3, Max frames per episode
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class MuZeroAtariConfig:
    def __init__(self, args):
        self.args = args
        self.scalar_to_categorical, self.categorical_to_scalar = make_categorical_representation_fns(args.support_size)
        self.tiled_action_encoding_fn, self.bias_plane_action_encoding_fn = make_action_encoding_fn(
            args.embedding_resolution, args.obs_resolution, args.num_actions
        )


if __name__ == "__main__":
    args = parse_args()
    # TODO: setup jax distributed
    
    args.world_size = jax.process_count()
    args.local_rank = jax.process_index()
    args.num_envs = args.local_num_envs * args.world_size
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in args.actor_device_ids]
    print("learner_devices", learner_devices)
    print("actor_devices", actor_devices)
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    global_actor_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.actor_device_ids
    ]
    print("global_learner_decices", len(global_learner_decices))
    print("global_actor_decices", len(global_actor_decices))
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.actor_devices = [str(item) for item in actor_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{uuid.uuid4()}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, buffer_key = jax.random.split(key, 3)

    # env setup
    envs = make_env_helper(system, args.env_id, args.seed, args.local_num_envs)()
    args.obs_shape = envs.observation_space.shape
    args.num_actions = envs.single_action_space.n
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    config = MuZeroAtariConfig(args)
    ############################### Setup Network, Optimizer, and Update Function #######################################

    def cosine_scheduler(t):
        lr_decay =  0.5 * (1 + jnp.cos(jnp.pi * t / args.num_updates)) 
        return lr_decay * args.lr_init

    network = make_muzero_network(config)
    applys = NetworkApplys(*network.apply)

    init_obs = jnp.zeros((args.local_num_envs,  *args.obs_shape))
    init_action = jnp.zeros((args.local_num_envs, args.num_stacked_frames))
    network_params = network.init(network_key, init_obs, init_action)
    # TODO: resolve optimizer
    # optimizer = optax.scale_by_adam(eps=1e-8)
    # optimizer = optax.chain(
    #     optimizer,
    #     optax.add_decayed_weights(args.weight_decay / args.lr_init))
    # optimizer = optax.chain(optimizer, optax.scale_by_schedule(cosine_scheduler),
    #                         optax.scale(-1.0)) # -1.0 indicates we're minimizing the loss
    optimizer = optax.adam(1e-2)#args.learning_rate)
    opt_state = optimizer.init(network_params)
    muzero_state = TrainState(params=network_params,
                            target_params=network_params,
                            opt_state=opt_state,
                            train_step=0)

    muzero_state = jax.device_put_replicated(muzero_state, devices=learner_devices)

    #prepare_data_fn = make_prepare_data_fn(args, learner_devices, scalar_to_categorical)
    single_device_update = make_single_device_update(applys, optimizer, config)
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_decices,
    )

    ######################## Dispatch Actors ###########################

    import multiprocessing as mp
    num_cpus = mp.cpu_count()
    print(f"num_cpus {num_cpus}")
    fair_num_cpus = num_cpus // len(args.actor_device_ids) 
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x,y,z: None

    rollout_queue = queue.Queue(maxsize=20)#args.num_actor_threads * len(args.actor_device_ids))
    params_queues = []

    for d_idx, d_id in enumerate(args.actor_device_ids):
        device_params = jax.device_put(flax.jax_utils.unreplicate(muzero_state.params), local_devices[d_id])
        rollout_fn = make_rollout_fn(local_devices[d_id], applys, args, make_env_helper)
        for thread_id in range(args.num_actor_threads):
            params_queue = queue.Queue(maxsize=1)
            params_queue.put(device_params)
            threading.Thread(
                target=rollout_fn,
                args=(
                    system,
                    jax.device_put(key, local_devices[d_id]),
                    args,
                    rollout_queue,
                    params_queue,
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    d_idx * args.num_actor_threads + thread_id,
                ),
            ).start()
            params_queues.append(params_queue)

    batch_queue = queue.Queue(maxsize=5) # arbitrary
    start_replay_buffer_manager(rollout_queue, batch_queue, config) # TODO: may want to include some logging inside this thread to monitor the replay buffer

    batch_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_network_version = 0
    while True:
        update_iteration_start = time.time()
        learner_network_version += 1

        batch_queue_get_start = time.time()
        (batch, weights, sampled_indicies) = batch_queue.get()
        batch_queue_get_time.append(time.time() - batch_queue_get_start)
        writer.add_scalar("stats/batch_queue_get_time", np.mean(batch_queue_get_time), learner_network_version)

        device_transfer_start = time.time()
        shard_fn = lambda x: jax.device_put_sharded(
            jnp.array_split(x, len(learner_devices), axis=0),
            learner_devices
        )
        sharded_batch = jax.tree_map(shard_fn, batch)
        writer.add_scalar("stats/learner/data_transfer_time", time.time() - device_transfer_start, learner_network_version)

        training_time_start = time.time()
        muzero_state, loss, v_loss, p_loss, r_loss = multi_device_update(
                muzero_state, sharded_batch
        )
        writer.add_scalar("stats/training_time", time.time() - training_time_start, learner_network_version)
        writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), learner_network_version)
        writer.add_scalar("stats/batch_queue_size", batch_queue.qsize(), learner_network_version)

        # writer.add_scalar("stats/training_time", time.time() - training_time_start, global_step)
        # writer.add_scalar("stats/rollout_queue_size", rollout_queue.qsize(), global_step)
        # writer.add_scalar("stats/params_queue_size", params_queue.qsize(), global_step)
        # print(
        #     global_step,
        #     f"actor_policy_version={actor_network_version}, learner_policy_version={learner_network_version}, training time: {time.time() - training_time_start}s", # learner_network_version
        # )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #writer.add_scalar("charts/learning_rate", muzero.opt_state[1].hyperparams["learning_rate"][0].item(), global_step)
        writer.add_scalar("losses/value_loss", np.mean(v_loss).item(), learner_network_version)
        writer.add_scalar("losses/policy_loss", np.mean(p_loss).item(), learner_network_version)
        writer.add_scalar("losses/reward_loss", np.mean(r_loss).item(), learner_network_version)
        writer.add_scalar("losses/loss", np.mean(loss), learner_network_version)
        
        # update network version
        
        # if update >= args.num_updates:
        #     break
            # for d_idx, d_id in enumerate(args.actor_device_ids):
            #     device_params = jax.device_put(flax.jax_utils.unreplicate(muzero_state.params), local_devices[d_id])
            #     for thread_id in range(args.num_actor_threads):
            #         params_queues[d_idx * args.num_actor_threads + thread_id].put(device_params)

    """ 
    batch contents 
        observation_stack: (1024, 94, 84, 84)
            (batch_size, num_stacked_frames*3, *observation_shape)
        actions: (1024, 32+5)
            (batch_size, num_stacked_frames+num_unroll_steps)
        value_targets: (1024, 6, 601)
            (batch_size, num_unroll_steps+1, support_size)
        policy_targets: (1024, 6, 4)
            (batch_size, num_unroll_steps+1, num_actions)
        reward_targets: (1024, 5, 601)
            (batch_size, num_unroll_steps, support_size)
    """

    # Efficient Zero obviously is an implementation of Atari scale replay buffer, you can see how they did it
    # for now get MuZero completely working on DMControl
    # much smaller memory needs and much faster training
    # Will need to change
        # network arch
        # observation and action stacking 
        # parallel mcts per action dimension
        # replay buffer


    # so priorities are updated 