import time
import queue
from collections import deque

import jax
import mctx
import rlax
import numpy as np
import jax.numpy as jnp

################################################
#################### rollout ###################


def make_rollout_fn(actor_device, applys, args, make_env):
    """
    Currently, loops over the number of training steps to be taken 
    and inside each step loops over the number of steps to taken per train step 
    range(async_update, (num-steps + 1) * async_update)
    when each inner loop finished the data collected is passed to the queue via the payload variable
    """

    mcts_fn = make_mcts_fn(actor_device, applys, args.num_simulations, args.gamma)

    def rollout_fn(
        system: str,
        key: jax.random.PRNGKey,
        args,
        rollout_queue: queue.Queue,
        params_queue: queue.Queue,
        writer,
        device_thread_id,
        ):
        print(f"Thread {device_thread_id} started!")
        
        envs = make_env(system, args.env_id, args.seed + device_thread_id, args.local_num_envs)()
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

        params_queue_get_time = deque(maxlen=10)
        rollout_time = deque(maxlen=10)
        rollout_queue_put_time = deque(maxlen=10)
        actor_policy_version = 0

        while True:
            update_time_start = time.time()
            obs = []
            dones = []
            actions = []
            values = []
            rewards = []
            mcts_policies = []
            truncations = []
            terminations = []
            env_recv_time = 0
            inference_time = 0
            storage_time = 0
            env_send_time = 0

            # get params
            params_queue_get_time_start = time.time()
            params = params_queue.get()
            actor_policy_version += 1
            params_queue_get_time.append(time.time() - params_queue_get_time_start)
            writer.add_scalar("stats/params_queue_get_time", np.mean(params_queue_get_time), global_step)

            rollout_time_start = time.time()
            for timestep in range(0, args.num_steps):
                env_recv_time_start = time.time()
                next_obs, next_reward, next_done, _, info = envs.recv() # TODO: resolve pitch variable origin
                env_recv_time += time.time() - env_recv_time_start
                global_step += len(next_done) * len_actor_device_ids * args.world_size
                env_id = info['env_id']

                if timestep == 0:
                    action_stack = np.zeros((args.num_stacked_frames, args.local_num_envs))

                elif timestep < args.num_stacked_frames + 1:
                    num_missing = args.num_stacked_frames - timestep
                    missing_actions = np.zeros((num_missing, args.local_num_envs))
                    current_actions = np.stack(actions)
                    action_stack = np.concatenate((missing_actions, current_actions))

                # TODO: fix recompilation on 
                else:
                    action_stack = actions[-args.num_stacked_frames:]
                    action_stack = np.stack(action_stack)

                action_stack = action_stack.transpose(1, 0)
                assert action_stack.shape == (args.local_num_envs, args.num_stacked_frames)

                inference_time_start = time.time()
                action, value, mcts_policy, key = mcts_fn(params, next_obs, action_stack, key)
                print(f"Thread {device_thread_id} finished {timestep} in {time.time() - inference_time_start}")
                inference_time += time.time() - inference_time_start
                action = jax.device_get(action)
                value = jax.device_get(value)
                mcts_policies = jax.device_get(mcts_policies)
                key = jax.device_get(key)

                env_send_time_start = time.time()
                envs.send(np.array(action), env_id)
                env_send_time += time.time() - env_send_time_start
                storage_time_start = time.time()
                obs.append(next_obs[:, -3:, :, :])
                dones.append(next_done)
                actions.append(action)
                values.append(value)
                mcts_policies.append(mcts_policy)
                rewards.append(next_reward)
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
            writer.add_scalar("stats/rollout_time", np.mean(rollout_time), global_step)

            avg_episodic_return = np.mean(returned_episode_returns)
            writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.add_scalar("charts/avg_episodic_length", np.mean(returned_episode_lengths), global_step)
            print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            writer.add_scalar("stats/truncations", np.sum(truncations), global_step)
            writer.add_scalar("stats/terminations", np.sum(terminations), global_step)
            writer.add_scalar("stats/env_recv_time", env_recv_time, global_step)
            writer.add_scalar("stats/inference_time", inference_time, global_step)
            writer.add_scalar("stats/storage_time", storage_time, global_step)
            writer.add_scalar("stats/env_send_time", env_send_time, global_step)
            # `make_bulk_array` is actually important. It accumulates the data from the lists
            # into single bulk arrays, which later makes transferring the data to the learner's
            # device slightly faster. See https://wandb.ai/costa-huang/cleanRL/reports/data-transfer-optimization--VmlldzozNjU5MTg1
            if args.learner_device_ids[0] != args.actor_device_ids[0]:
                obs, actions, values, mcts_policies, rewards = make_bulk_array(
                    obs,
                    actions,
                    values,
                    mcts_policies,
                    rewards
                )

            # store data
            payload = (
                global_step,
                obs,
                values,
                actions,
                mcts_policies,
                rewards,
                dones,
                rewards,
                np.mean(params_queue_get_time),
            )
            rollout_queue_put_time_start = time.time()
            rollout_queue.put(payload)
            rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)
            writer.add_scalar("stats/rollout_queue_put_time", np.mean(rollout_queue_put_time), global_step)

            writer.add_scalar(
                "charts/SPS_update",
                int(
                    args.local_num_envs
                    * args.num_steps
                    * len_actor_device_ids
                    * args.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )
    return rollout_fn

@jax.jit
def make_bulk_array(
    obs: list,
    actions: list,
    values: list,
    mcts_policies: list,
    rewards: list,
):
    obs = jnp.asarray(obs)
    actions = jnp.asarray(actions)
    values = jnp.asarray(values)
    mcts_policies = jnp.asarray(mcts_policies)
    rewards = jnp.asarray(rewards)
    return obs, values, actions, mcts_policies, rewards

#################### rollout ###################
################################################

################################################
#################### MCTS ######################

def make_mcts_fn(actor_device, applys, num_simulations, gamma):

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

        def mcts(params, observations, actions, rng):
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
                #temperature= # TODO: add temperature function
            )
            # TODO: you don't want predicted value you want root values of search tree
            return output['action'], value, output['action_weights'], rng
 
        return jax.jit(mcts, device=actor_device)

#################### MCTS ######################
################################################

                # get stacked images and actions
                # obs.append(next_obs)
                # if idx == args.async_update:
                #     stacked_obs = np.asarray(obs).transpose(1, 0, 2, 3, 4)
                #     stacked_actions = np.asarray([[0]*20]).transpose(1,0)
                # elif len(actions) > args.num_stacked_frames:
                #     stacked_obs = np.asarray(obs)[-args.num_stacked_frames].transpose(1, 0, 2, 3, 4)
                #     stacked_actions = np.asarray(actions)[-args.num_stacked_frames].transpose(1,0)
                # else:
                #     stacked_obs = np.asarray(obs).transpose(1, 0, 2, 3, 4)
                #     stacked_actions = np.asarray(actions).transpose(1,0)