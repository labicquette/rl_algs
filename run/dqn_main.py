import os
import functools
import time
from omegaconf import OmegaConf
import my_gym as gym  #O.Sigaud Gym (0.21.0)
import bbrl
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, RemoteAgent, TemporalAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger

from algorithms import DQN

def make_env(env_name):
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


def run_dqn(cfg, max_grad_norm=0.5):
    # 1)  Build the  logger
    logger = DQN.Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the DQN-like Agent
    train_agent, eval_agent, q_agent = DQN.create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training

    # 6) Configure the optimizer over the a2c agent
    optimizer = DQN.setup_optimizers(cfg, q_agent)
    nb_steps = 0
    tmp_steps = 0

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1, stochastic=True
            )
        else:
            train_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        transition_workspace = train_workspace.get_transitions()

        # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
        

        q_values, done, truncated, reward, action = transition_workspace[
            "q_values", "env/done", "env/truncated", "env/reward", "action"
        ]

        nb_steps += len(action[0]) * cfg.algorithm.n_envs

        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj
        must_bootstrap = torch.logical_or(~done[1], truncated[1])

        # Compute critic loss
        critic_loss, td = DQN.compute_critic_loss(
            cfg, reward, must_bootstrap, q_values, action
        )

        # Store the loss for tensorboard display
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_agent.parameters(), max_grad_norm)
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"epoch: {epoch}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./dqn0_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn0_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)