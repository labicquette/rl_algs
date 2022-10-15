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




def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)




class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.model(obs)



class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)

    def predict_value(self, obs, action):
        osb_act = torch.cat((obs, action), dim=0)
        q_value = self.model(osb_act)
        return q_value    

