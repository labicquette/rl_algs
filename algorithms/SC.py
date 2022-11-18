from copy import deepcopy
from bbrl.utils import ReplayBuffer

class surrogate :
    def __init__(self, model, replay_buffer):
        self.model = deepcopy(model)
        self.replay_buffer = replay_buffer

    def forward(self, pop_weights, rb=None, random_sampling=False):
        if rb is None :
            rb = self.replay_buffer
            replay_buffer.
        