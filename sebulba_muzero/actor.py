import time
import queue
from collections import deque
from functools import partial

import jax
import mctx
import numpy as np
import jax.numpy as jnp
from flax import struct

from learner import make_compute_value_target
from utils import softmax_temperature_fn


def make_rollout_fn(actor_device, applys, args, make_env):
    """
    Creates a function that performs a rollout of batched environments using a specfic actor device.
    """

    mcts_fn = make_mcts_fn(actor_device, applys, args.total_updates, args.num_simulations, args.gamma)
    compute_value_target = make_compute_value_target(args.num_unroll_steps, args.td_steps, args.gamma)

    def rollout_fn(
        key: jax.random.PRNGKey,
        args,
        rollout_queue: queue.Queue,
        params_queue: queue.Queue,
        writer,
        device_thread_id,
        ):
        print(f"Thread {device_thread_id} started!")

        envs = make_env(args.env_id, args.seed + device_thread_id, args.local_num_envs)()
        num_actor_threads = args.num_actor_threads
        len_actor_device_ids = len(args.actor_device_ids)
        global_step = 0
        # TRY NOT TO MODIFY: start the game
        start_time = time.time()

        # put data in the last index
        episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
        returned_episode_returns = np.zeros((args.local_num_envs,), dtype=np.float32)
        episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
        returned_episode_lengths = np.zeros((args.local_num_envs,), dtype=np.float32)
        envs.async_reset()

        rollout_time = deque(maxlen=10)
        rollout_queue_put_time = deque(maxlen=10)
        actor_policy_version = 0

        params = None
        last_rollout = None
        while True:
            update_time_start = time.time()
            obs = []
            dones = []
            actions = []
            rewards = []
            values = []
            mcts_policies = []
            truncations = []
            terminations = []
            env_recv_time = 0
            h2d_time = 0
            inference_time = 0
            d2h_time = 0
            storage_time = 0
            env_send_time = 0

            # if there are params in the queue, get them
            if params_queue.qsize() != 0:
                params_queue_get_time_start = time.time()
                params, train_step = params_queue.get()
                actor_policy_version += 1
                writer.add_scalar("stats/actor/params_queue_get_time", time.time() - params_queue_get_time_start, actor_policy_version)

            rollout_time_start = time.time()
            for timestep in range(0, args.num_steps):
                env_recv_time_start = time.time()
                next_obs, next_reward, next_done, _, info = envs.recv() # TODO: resolve pitch variable origin
                env_recv_time += time.time() - env_recv_time_start
                global_step += len(next_done) * len_actor_device_ids * num_actor_threads * args.world_size
                env_id = info['env_id']

                if timestep == 0:
                    initial_obs = next_obs
                    action_stack = np.zeros((args.num_stacked_frames, args.local_num_envs))

                elif timestep < args.num_stacked_frames + 1:
                    num_missing = args.num_stacked_frames - timestep
                    missing_actions = np.zeros((num_missing, args.local_num_envs))
                    current_actions = np.stack(actions)
                    action_stack = np.concatenate((missing_actions, current_actions))

                # TODO: fix recompilation on 33rd iteration
                else:
                    action_stack = actions[-args.num_stacked_frames:]
                    action_stack = np.stack(action_stack)

                action_stack = action_stack.transpose(1, 0)
                assert action_stack.shape == (args.local_num_envs, args.num_stacked_frames)

                # move feature to device
                h2d_time_start = time.time()
                next_obs = jax.device_put(next_obs, device=actor_device)
                action_stack = jax.device_put(action_stack, device=actor_device)
                h2d_time += time.time() - h2d_time_start

                inference_time_start = time.time()
                next_obs, action, value, mcts_policy, key = mcts_fn(params, next_obs, action_stack, train_step, key)
                inference_time += time.time() - inference_time_start

                # device to host
                d2h_time_start = time.time()
                action = jax.device_get(action)
                d2h_time += time.time() - d2h_time_start

                env_send_time_start = time.time()
                envs.send(action, env_id)
                env_send_time += time.time() - env_send_time_start
                storage_time_start = time.time()
                obs.append(next_obs[:, -args.channels_per_frame:, :, :]) # TODO: is it safe to assume channels is in chronological order?
                dones.append(next_done)
                actions.append(action)
                values.append(value)
                mcts_policies.append(mcts_policy)
                rewards.append(next_reward)
                
                # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
                # so we use our own truncated flag
                truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
                truncations.append(truncated)
                terminations.append(info["terminated"])

                episode_returns[env_id] += info["reward"]
                returned_episode_returns[env_id] = np.where(
                    info["terminated"] + truncated, episode_returns[env_id], returned_episode_returns[env_id]
                )
                episode_returns[env_id] *= (1 - info["terminated"]) * (1 - truncated)
                episode_lengths[env_id] += 1
                returned_episode_lengths[env_id] = np.where(
                    info["terminated"] + truncated, episode_lengths[env_id], returned_episode_lengths[env_id]
                )
                episode_lengths[env_id] *= (1 - info["terminated"]) * (1 - truncated)
                storage_time += time.time() - storage_time_start

            if args.profile:
                action.block_until_ready()

            # logs
            rollout_time.append(time.time() - rollout_time_start)
            writer.add_scalar("stats/actor/rollout_time", np.mean(rollout_time), global_step)

            avg_episodic_return = np.mean(returned_episode_returns)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            #print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
            #print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            writer.add_scalar("stats/actor/truncations", np.sum(truncations), global_step)
            writer.add_scalar("stats/actor/terminations", np.sum(terminations), global_step)
            writer.add_scalar("stats/actor/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/actor/h2d_time", h2d_time, global_step)
            writer.add_scalar("stats/actor/inference_time", inference_time, global_step)
            writer.add_scalar("stats/actor/d2h_time", d2h_time, global_step)
            writer.add_scalar("stats/actor/storage_time", storage_time, global_step)
            writer.add_scalar("stats/actor/env_send_time", env_send_time, global_step)

            rollout_mgmt_time_start = time.time()
            current_rollout = make_rollout(
                obs,
                actions,
                values,
                mcts_policies,
                rewards,
                dones
            )

            current_rollout = prefix_padding(current_rollout, last_rollout, initial_obs)
            if last_rollout is not None:
                last_rollout = suffix_padding(last_rollout, current_rollout)
                payload_rollout = last_rollout

                # store data
                rollout_queue_put_time_start = time.time()
                rollout_queue.put(payload_rollout)
                rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)
                writer.add_scalar("stats/actor/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)

                print(f"Thread {device_thread_id} finished rollout {global_step} in {time.time() - rollout_time_start} and put it in the queue in {time.time() - rollout_queue_put_time_start}")

            last_rollout = current_rollout
            rollout_mgmt_time = time.time() - rollout_mgmt_time_start
            writer.add_scalar("stats/actor/rollout_mgmt_time", rollout_mgmt_time, global_step)

            writer.add_scalar(
                "charts/SPS",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * num_actor_threads
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )

    @struct.dataclass
    class Rollout:
        """A class for storing batched rollout data with methods for padding"""
        obs: np.ndarray
        actions: np.ndarray
        value_targets: np.ndarray
        policy_targets: np.ndarray
        rewards: np.ndarray
        dones: np.ndarray
        priorities: np.ndarray

    @jax.jit
    def make_rollout(
        obs: list,
        actions: list,
        values: list,
        mcts_policies: list,
        rewards: list,
        dones: list
    ):
        obs = jnp.asarray(obs).swapaxes(0, 1)
        actions = jnp.asarray(actions).swapaxes(0, 1)
        values = jnp.asarray(values).swapaxes(0, 1)
        mcts_policies = jnp.asarray(mcts_policies).swapaxes(0, 1)
        rewards = jnp.asarray(rewards).swapaxes(0, 1)
        dones = jnp.asarray(dones).swapaxes(0, 1)
        value_targets = compute_value_target(rewards, values, dones)
        priorities = jnp.abs(value_targets - values)
        return Rollout(
            obs=obs,
            actions=actions,
            value_targets=value_targets,
            policy_targets=mcts_policies,
            rewards=rewards,
            dones=dones,
            priorities=priorities,
        )

    @jax.jit
    def prefix_padding(current, last, initial_obs):
        if last is None:
            prefix_obs = initial_obs.reshape(
                args.local_num_envs, args.num_stacked_frames, args.channels_per_frame, args.obs_resolution, args.obs_resolution
                )
            prefix_action = jnp.zeros((args.local_num_envs, args.num_stacked_frames), dtype=jnp.float32)

        else:
            prefix_obs = last.obs[:, -args.num_stacked_frames:, :, :, :]
            prefix_action = last.actions[:, -args.num_stacked_frames:]


        return current.replace(
            obs=jnp.concatenate([prefix_obs, current.obs], axis=1),
            actions=jnp.concatenate([prefix_action, current.actions], axis=1),
        )

    @jax.jit
    def suffix_padding(current, next):
        idx = args.num_unroll_steps + args.td_steps
        return current.replace(
            actions=jnp.concatenate([current.actions, next.actions[:, :idx]], axis=1),
            value_targets=jnp.concatenate([current.value_targets, next.value_targets[:, :idx]], axis=1),
            policy_targets=jnp.concatenate([current.policy_targets, next.policy_targets[:, :idx]], axis=1),
            rewards=jnp.concatenate([current.rewards, next.rewards[:, :idx]], axis=1),
            dones=jnp.concatenate([current.dones, next.dones[:, :idx]], axis=1),
        )


    return rollout_fn


def make_mcts_fn(actor_device, applys, train_steps, num_simulations, gamma):

        def recurrent_fn(params, rng, action, prev_embedding):
            embedding, reward, value, policy = applys.recurrent_inference(
                params, prev_embedding, action, scalar=True
            )
            discount = jnp.full_like(reward, gamma)
            output = mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=policy,
                value=value,
            )
            return (output, embedding)

        def mcts(params, observations, actions, train_step, rng):
            observations = jnp.array(observations)
            actions = jnp.array(actions)
            rng, _ = jax.random.split(rng)
            embedding, value, policy = applys.initial_inference(
                params, observations, actions, scalar=True
            )
            root = mctx.RootFnOutput(
                prior_logits=policy,
                value=value,
                embedding=embedding
            )
            output = mctx.muzero_policy(
                params=params,
                rng_key=rng,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                temperature=softmax_temperature_fn(train_step, train_steps)
            )
            
            return observations, output['action'], value, output['action_weights'], rng
 
        return jax.jit(mcts, device=actor_device)
