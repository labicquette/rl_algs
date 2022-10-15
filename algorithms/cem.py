import torch
import bbrl
from bbrl import get_class
from torch.nn.utils.convert_parameters import parameters_to_vector
from bbrl.agents.agent import Agent
from bbrl.agents import  Agents, TemporalAgent
from bbrl.agents.asynchronous import AsynchronousAgent
from bbrl.utils.utils import vector_to_parameters
from models.models import ContinuousDeterministicActor
import copy


def create_cem_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    actor = ContinuousDeterministicActor(obs_size, 
                        cfg.algorithm.architecture.actor_hidden_size, 
                        act_size
                        )
    tr_agent = Agents(train_env_agent, actor)  
    ev_agent = Agents(eval_env_agent, actor)

        # Get an agent that is executed on a complete workspace
    
    acquisition_actors=[]
    acquisition_agents = []
    for i in range(cfg.algorithm.num_processes):
        acquisition_actors.append(ContinuousDeterministicActor(obs_size, 
                        cfg.algorithm.architecture.actor_hidden_size, 
                        act_size
                        ))
        temporal_agent = TemporalAgent(Agents(train_env_agent, acquisition_actors[-1]))
        temporal_agent.seed(cfg.algorithm.seed)
        agent = AsynchronousAgent(temporal_agent)
        acquisition_agents.append(agent)
    eval_agent = TemporalAgent(ev_agent)
    

    return acquisition_actors, acquisition_agents, eval_agent, actor



class Cem:

    def __init__(self, cfg, actor) -> None:

        # hyper-parameters: 
        self.pop_size = cfg.es_algorithm.pop_size
        self.initial_buffer_size = cfg.algorithm.learning_starts

        # CEM objects
        actor_weights = actor.parameters()
        self.centroid = copy.deepcopy(parameters_to_vector(actor_weights).detach())
        code_args = {'num_params': len(self.centroid),'mu_init':self.centroid}
        kwargs = {**cfg.es_algorithm, **code_args}
        self.es_learner = get_class(cfg.es_algorithm)(**kwargs)

        self.pop_weights = self.es_learner.ask(self.pop_size)

        # vector_to_parameters does not seem to work when module are in different processes
        # the transfert agent is used to transfert vector_to_parameters in main thread
        # and then transfert the parameters to another agent in another process.
        self.param_transfert_agent = copy.deepcopy(actor)

    def get_acquisition_actor(self,i) -> Agent:
        actor = self.rl_learner.get_acquisition_actor()
        weight = self.pop_weights[i]

        vector_to_parameters(weight,self.param_transfert_agent.parameters())
        actor.load_state_dict(self.param_transfert_agent.state_dict())
        return actor

    def update_acquisition_actor(self,actor,i) -> None:
        weight = self.pop_weights[i]
        vector_to_parameters(weight,self.param_transfert_agent.parameters())        
        actor.load_state_dict(self.param_transfert_agent.state_dict())

    def train(self,acq_workspaces, fitness, n_total_actor_steps,logger) -> None:

        # Compute fitness of population
        self.es_learner.tell(self.pop_weights,fitness) #  Update CEM
        self.pop_weights = self.es_learner.ask(self.pop_size) # Generate new population

    

    