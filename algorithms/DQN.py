import torch
import torch.nn as nn

from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace

from bbrl.agents import Agents, RemoteAgent, TemporalAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.logger import TFLogger



def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class DiscreteQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/env_obs", t))
        q_values = self.model(obs).squeeze(-1)
        self.set(("q_values", t), q_values)
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)

class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t, **kwargs):
        q_values = self.get(("q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)


def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = DiscreteQAgent(obs_size, cfg.algorithm.architecture.hidden_size, act_size)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    q_agent = TemporalAgent(critic)
    tr_agent = Agents(train_env_agent, critic, explorer)
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, q_agent


class Logger():

  def __init__(self, cfg):
    self.logger = instantiate_class(cfg.logger)

  def add_log(self, log_string, loss, epoch):
    self.logger.add_scalar(log_string, loss.item(), epoch)

  # Log losses
  def log_losses(self, cfg, epoch, critic_loss, entropy_loss, a2c_loss):
    self.add_log("critic_loss", critic_loss, epoch)
    self.add_log("entropy_loss", entropy_loss, epoch)
    self.add_log("actor_loss", a2c_loss, epoch)



# Configure the optimizer over the q agent
def setup_optimizers(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer



def compute_critic_loss(cfg, reward, must_bootstrap, q_values, action):
    """_summary_

    Args:
        cfg (_type_): _description_
        reward (torch.Tensor): A (T x B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (T x B) tensor containing 0 if the episode is completed at time $t$
        q_values (torch.Tensor): a (T x B x A) tensor containing Q values
        action (torch.LongTensor): a (T) long tensor containing the chosen action

    Returns:
        torch.Scalar: The DQN loss and the temporal difference
    """
    # We compute the max of Q-values over all actions
    max_q = q_values.max(-1)[0].detach()

    # To get the max of Q(s_{t+1}, a), we take max_q[1:]
    # The same about must_bootstrap. 
    target = (
        reward[:-1] + cfg.algorithm.discount_factor * max_q * must_bootstrap.int()
    )
    # To get Q(s,a), we use torch.gather along the 3rd dimension (the action)
    act = action[0].unsqueeze(-1)
    qvals = torch.gather(q_values[0], dim=1, index=act).squeeze()

    # Compute the temporal difference (use must_boostrap as to mask out finished episodes)
    td = target - qvals 
    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    return critic_loss, td