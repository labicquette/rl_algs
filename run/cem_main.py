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

from bbrl import get_arguments, get_class, instantiate_class

from bbrl.workspace import Workspace

from bbrl.agents import Agents, RemoteAgent, TemporalAgent

from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

from algorithms import cem
from utils import utils

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




def run_cem(cfg):
    # 1)  Build the  logger
    logger = utils.Logger(cfg)
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

    # 3) Create the CEM Agent
    (
        actors,
        workers,
        eval_agent,
        actor
    ) = cem.create_cem_agent(cfg, train_env_agent, eval_env_agent)
    
    pop_size = cfg.es_algorithm.pop_size
    n_processes = min(cfg.algorithm.num_processes,cfg.es_algorithm.pop_size)

    cem_pop = cem.Cem(cfg, actor)

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        

            #main cem loop
        acquisition_workspaces = []
        nb_agent_finished = 0
        while(nb_agent_finished < pop_size):
            n_to_launch = min(pop_size-nb_agent_finished, n_processes)
            for idx_agent in range(n_to_launch):        
                idx_weight = idx_agent + nb_agent_finished
                cem_pop.update_acquisition_actor(actors[idx_agent],idx_weight)
                # TODO: add noise args to agents interaction with env ? Alois does not. 
                if epoch > 0 :
                    actors[idx_agent].workspace.zero_grad()
                    actors[idx_agent].workspace.copy_n_last_steps(1)
                    workers[idx_agent](t=1, n_steps=cfg.algorithm.n_steps -1)
                else :
                    workers[idx_agent](t=0, n_steps=cfg.algorithm.n_steps)

            #Wait for agents execution
            running=True
            while running:
                are_running = [a.is_running() for a in workers[:n_to_launch]]
                running = any(are_running)

            nb_agent_finished += n_to_launch
            acquisition_workspaces += [a.get_workspace() for a in workers[:n_to_launch]]        
            
        agents_creward = torch.zeros(len(acquisition_workspaces))
        for i,acquisition_worspace in enumerate(acquisition_workspaces):
            done = acquisition_worspace['env/done']
            cumulated_reward = acquisition_worspace['env/cumulated_reward']
            creward = cumulated_reward[done]
            agents_creward[i] = creward.mean()
            transition_workspace = acquisition_worspace.get_transitions()
            action = transition_workspace["action"]
            nb_steps += action[0].shape[0]
            rb.put(transition_workspace)
            
        rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

        cem_pop.train(agents_creward)
        
        done, truncated, reward, action = rb_workspace[
            "env/done", "env/truncated", "env/reward", "action"
        ]

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



@hydra.main(config_path=os.path.join(os.getcwd(),'configs/'), config_name="cem.yaml")
def main (config : DictConfig):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    torch.manual_seed(config.algorithm.seed)
    run_cem(config)

if __name__=='__main__':
    main()