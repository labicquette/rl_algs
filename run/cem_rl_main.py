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
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class

from bbrl.workspace import Workspace

from bbrl.agents import Agents, RemoteAgent, TemporalAgent

from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

from tqdm import tqdm

from algorithms import CEM_RL, TD3, cem

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




def run_cem_rl(cfg):
    # 1)  Build the  logger
    logger = TD3.Logger(cfg)
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

    # 3) Create the CEM_RL Agents
    (
        actors,
        workers,
        train_agent,
        eval_agent,
        actor,
        critic_1,
        critic_2,
        # target_actor,
        target_critic_1,
        target_critic_2,
    ) = CEM_RL.create_cem_rl_agents(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent_1 = TemporalAgent(critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    tau = cfg.algorithm.tau_target


    pop_size = cfg.es_algorithm.pop_size
    n_processes = min(cfg.algorithm.num_processes,cfg.es_algorithm.pop_size)

    cem_pop = cem.Cem(cfg, actor)


    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer_1, critic_optimizer_2 = TD3.setup_optimizers(cfg, actor, critic_1, critic_2)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        # main cem loop
        
        
        # print(f"done {done}, reward {reward}, action {action}")
        if nb_steps > cfg.algorithm.learning_starts:

            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action"
            ]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not done
            # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
            must_bootstrap = torch.logical_or(~done[1], truncated[1])
            selected_actors = random.sample(range(0,cfg.es_algorithm.pop_size),cfg.es_algorithm.pop_size//2)
            for i_pop in tqdm(range(cfg.es_algorithm.pop_size//2)):
                weight = copy.deepcopy(cem_pop.pop_weights[selected_actors[i_pop]])
                vector_to_parameters(weight.detach().clone(), actor.parameters())
                vector_to_parameters(weight.detach().clone(), ag_actor.parameters())
                actor_optimizer = CEM_RL.reset_actor_optimizer(cfg, actor)
                for _ in range(2* cfg.algorithm.n_steps // cfg.es_algorithm.pop_size):
                    q_values = []
                    post_q_values = []
                    q_agents = [q_agent_1, q_agent_2]
                    t_q_agents = [target_q_agent_1, target_q_agent_2]


                    for i in range(0,2):
                        q_agents[i](rb_workspace, detach_actions=True, t=0, n_steps=1)
                        q_values.append(rb_workspace["q_value"])

                    with torch.no_grad():
                        # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                        ag_actor(rb_workspace, t=1, n_steps=1)

                    with torch.no_grad():
                        for i in range(0,2):
                            t_q_agents[i](rb_workspace, t=1, n_steps=1)
                            post_q_values.append(rb_workspace["q_value"])    
                        post_q_value= torch.min(*post_q_values).squeeze(-1)


                    # logger.add_log("q/q_value_1",q_values[0].mean(), nb_steps)
                    # logger.add_log("q/q_value_2",q_values[1].mean(), nb_steps)
                    # logger.add_log("q/e_q_value_1",post_q_values[0].mean(), nb_steps)
                    # logger.add_log("q/e_q_value_2",post_q_values[1].mean(), nb_steps)
            
                    critic_1_loss=  TD3.compute_critic_loss(
                        cfg, reward, must_bootstrap, q_values[0][0], post_q_value[1]
                    ) 
                    critic_2_loss = TD3.compute_critic_loss(
                        cfg, reward, must_bootstrap, q_values[1][0], post_q_value[1]
                    )
                    #logger.add_log("loss/td_loss_1", critic_1_loss, nb_steps)
                    #logger.add_log("loss/td_loss_2", critic_2_loss, nb_steps)

                    critic_loss = critic_1_loss + critic_2_loss
                    critic_optimizer_1.zero_grad()
                    critic_optimizer_2.zero_grad()
                    critic_loss.backward()
                    n = torch.nn.utils.clip_grad_norm_(
                        critic_1.parameters(), cfg.algorithm.max_grad_norm
                    )
                    #logger.add_log("monitor/grad_norm_q_1",n,nb_steps)
                    n = torch.nn.utils.clip_grad_norm_(
                        critic_2.parameters(), cfg.algorithm.max_grad_norm
                    )
                    #logger.add_log("monitor/grad_norm_q_2",n,nb_steps)
                    critic_optimizer_1.step()
                    critic_optimizer_2.step()
                    TD3.soft_update_params(critic_1, target_critic_1, tau)
                    TD3.soft_update_params(critic_2, target_critic_2, tau)
                    
                

                for _ in range(cfg.algorithm.n_steps):
                    # Actor update
                    # Now we determine the actions the current policy would take in the states from the RB
                    ag_actor(rb_workspace, t=0, n_steps=1)
                    # We determine the Q values resulting from actions of the current policy
                    q_agent_1(rb_workspace, t=0, n_steps=1)
                    # and we back-propagate the corresponding loss to maximize the Q values
                    q_values = rb_workspace["q_value"]
                    actor_loss = TD3.compute_actor_loss(q_values)
                    logger.add_log("loss/actor_loss", actor_loss, nb_steps)
                    # if -25 < actor_loss < 0 and nb_steps > 2e5:
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    n = torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), cfg.algorithm.max_grad_norm
                    )
                    logger.add_log("monitor/grad_norm_action",n,nb_steps)
                    logger.add_log("loss/q_loss", actor_loss, nb_steps)
                    actor_optimizer.step()
                    # Soft update of target q function
                    tau = cfg.algorithm.tau_target
                    TD3.soft_update_params(actor, ag_actor, tau)

                # reintroduce weights in population
                vector_param = parameters_to_vector(actor.parameters())
                cem_pop.pop_weights[selected_actors[i_pop]] = vector_param.detach()

        acquisition_workspaces = []
        nb_agent_finished = 0
        while(nb_agent_finished < pop_size):
            n_to_launch = min(pop_size-nb_agent_finished, n_processes)
            for idx_agent in range(n_to_launch):        
                idx_weight = idx_agent + nb_agent_finished
                cem_pop.update_acquisition_actor(actors[idx_agent],idx_weight)
                workers[idx_agent](t=0, n_steps=cfg.algorithm.n_steps)

            # Wait for agents execution
            running=True
            while running:
                are_running = [a.is_running() for a in workers[:n_to_launch]]
                running = any(are_running)

            nb_agent_finished += n_to_launch
            acquisition_workspaces += [a.get_workspace() for a in workers[:n_to_launch]]        
            
        agents_creward = torch.zeros(len(acquisition_workspaces))
        for i,acquisition_workspace in enumerate(acquisition_workspaces):
            done = acquisition_workspace['env/done']
            cumulated_reward = acquisition_workspace['env/cumulated_reward']
            creward = cumulated_reward[done]
            agents_creward[i] = creward.mean()
            transition_workspace = acquisition_workspace.get_transitions()
            action = transition_workspace["action"]
            nb_steps += action[0].shape[0]
            rb.put(transition_workspace)
            
        
        
        cem_pop.train(agents_creward)
        # end CEM execution



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
                        q_agent_1.agent,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )



@hydra.main(config_path=os.path.join(os.getcwd(),'configs/'), config_name="cem_rl.yaml")
def main (config : DictConfig):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    torch.manual_seed(config.algorithm.seed)
    run_cem_rl(config)

if __name__=='__main__':
    main()