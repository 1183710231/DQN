import torch.nn as nn


class BaseQNet(nn.Module):
    # 此处修改  64 -》 100 -》64
    def __init__(self, num_states, dim_state, dim_action, dim_hidden=64):
        super(BaseQNet, self).__init__()
        self.num_states = num_states
        self.dim_state = dim_state
        self.dim_hidden = dim_hidden
        self.dim_action = dim_action

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_action(self, states):
        raise NotImplementedError()
