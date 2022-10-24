
from algorithms.TD3 import create_td3_agent
from algorithms.cem import create_cem_agent
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy 
import math

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

from bbrl.workspace import Workspace

from bbrl.agents import Agents, RemoteAgent, TemporalAgent

from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer

from torch.distributions import Normal

def create_cem_rl_agents(cfg, train_env_agent, eval_env_agent):
    (
        actors,
        workers,
        eval_agent,
        actor
    ) = create_cem_agent(cfg, train_env_agent, eval_env_agent)
    (
        train_agent,
        eval_agent,
        actor2,
        critic_1,
        critic_2,
        # target_actor,
        target_critic_1,
        target_critic_2,
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    return actors, workers, train_agent, eval_agent, actor, critic_1, critic_2, target_critic_1, target_critic_2  # , target_actor

def reset_actor_optimizer(cfg, actor):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    return actor_optimizer


def reset_critic_optimizer(cfg, critic):
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(parameters, **critic_optimizer_args)
    return critic_optimizer
