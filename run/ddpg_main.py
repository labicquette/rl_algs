import os
import functools
import time
from typing import Tuple
from omegaconf import DictConfig
import torch
import my_gym
import gym
import bbrl
import copy
import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

from bbrl.workspace import Workspace

from bbrl.agents import Agents, RemoteAgent, TemporalAgent

from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

from algorithms import DDPG

def make_gym_env(env_name):
    return gym.make(env_name)


def get_env_agents(cfg):
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
    get_class(cfg.gym_env),
    get_arguments(cfg.gym_env),
    cfg.algorithm.nb_evals,
    cfg.algorithm.seed,
    )
    return train_env_agent, eval_env_agent




def run_ddpg(cfg):
    # 1)  Build the  logger
    logger = DDPG.Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )

    # 3) Create the DDPG Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic,
        # target_actor,
        target_critic
    ) = DDPG.create_ddpg_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)
    

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = DDPG.setup_optimizers(cfg, actor, critic)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        done, truncated, reward, action = rb_workspace[
            "env/done", "env/truncated", "env/reward", "action"
        ]
        # print(f"done {done}, reward {reward}, action {action}")
        if nb_steps > cfg.algorithm.learning_starts:
            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])


            # Critic update
            # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
            q_agent(rb_workspace, t=0, n_steps=1)
            q_values =rb_workspace["q_value"]
            # print(f"q_values ante : {q_values}")

            with torch.no_grad():
                # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                ag_actor(rb_workspace, t=1, n_steps=1)
                # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                target_q_agent(rb_workspace, t=1, n_steps=1)
                # q_agent(rb_workspace, t=1, n_steps=1)
            # finally q_values contains the above collection at t=0 and t=1
            
            # print(f"q_values post : {post_q_values[1]}")
            post_q_values= rb_workspace["q_value"]

            logger.add_log("q/q_value_1",q_values.mean(), nb_steps)
            logger.add_log("q/q_value_2",q_values.mean(), nb_steps)
            logger.add_log("q/e_q_value_1",post_q_values.mean(), nb_steps)
            logger.add_log("q/e_q_value_2",post_q_values.mean(), nb_steps)
            


            # Compute critic loss
            critic_loss = DDPG.compute_critic_loss(
                cfg, reward, must_bootstrap, q_values[0], post_q_values[1]
            )
            logger.add_log("loss/td_loss_1", critic_loss, nb_steps)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            n = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), cfg.algorithm.max_grad_norm
            )
            logger.add_log("monitor/grad_norm_q_1",n,nb_steps)
            critic_optimizer.step()


            # Actor update
            # Now we determine the actions the current policy would take in the states from the RB
            ag_actor(rb_workspace, t=0, n_steps=1)
            # We determine the Q values resulting from actions of the current policy
            q_agent(rb_workspace, t=0, n_steps=1)
            # and we back-propagate the corresponding loss to maximize the Q values
            q_values = rb_workspace["q_value"]
            
            actor_loss = DDPG.compute_actor_loss(q_values)
            logger.add_log("loss/actor_loss", actor_loss, nb_steps)
            # if -25 < actor_loss < 0 and nb_steps > 2e5:
            actor_optimizer.zero_grad()
            actor_loss.backward()
            n = torch.nn.utils.clip_grad_norm_(
                actor.parameters(), cfg.algorithm.max_grad_norm
            )
            logger.add_log("monitor/grad_norm_action", n, nb_steps)
            logger.add_log("loss/q_loss", actor_loss, nb_steps)
            actor_optimizer.step()
            # Soft update of target q function
            tau = cfg.algorithm.tau_target
            DDPG.soft_update_params(critic, target_critic, tau)
            # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./ddpg_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "ddpg_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                if cfg.plot_agents:
                    plot_policy(
                        actor,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent.agent,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )



@hydra.main(config_path=os.path.join(os.getcwd(),'configs/'), config_name="ddpg.yaml")
def main (config : DictConfig):
    torch.manual_seed(config.algorithm.seed)
    run_ddpg(config)

if __name__=='__main__':
    main()