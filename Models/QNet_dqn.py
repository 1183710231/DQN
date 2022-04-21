import torch
import torch.nn as nn
from math import sqrt
import pickle
import numpy as np

from Models.BaseQNet import BaseQNet


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class QNet_dqn(BaseQNet):
    # 此处修改 64 -》 100 -> 64
    def __init__(self, num_states, dim_state, dim_action, dim_hidden=64, activation=nn.LeakyReLU):
        super().__init__(num_states, dim_state, dim_action, dim_hidden)

        # path = 'param_ent64_rel64_TransE.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        # f = open(path, 'rb')
        # data = pickle.load(f)
        # data2 = data.get('weights').get('entityEmbed')
        # print(data2.shape)

        # self.matrix = nn.Parameter(torch.Tensor(self.num_states, self.dim_state))
        # print(self.num_states, self.dim_state)
        # self.matrix = nn.Parameter(torch.Tensor(data2), requires_grad=False)
        # self.matrix = np.array(data2)


        # 神经网络
        self.qvalue = nn.Sequential(nn.Linear(self.dim_state, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_hidden),
                                    activation(),
                                    nn.Linear(self.dim_hidden, self.dim_action))
        # nn.init.uniform_(self.matrix, -6 / sqrt(self.dim_state), 6 / sqrt(self.dim_state))
        self.apply(init_weight)

    def forward(self, states, **kwargs):
        # embeddings = self.matrix[states]
        # print('states: ', states.shape)
        q_values = self.qvalue(states)
        return q_values

    def get_action(self, states):
        q_values = self.forward(states, )
        # ？？？这里不应该是【0】
        # print(type(q_values))
        max_action = q_values.max(dim=1)[1]  # action index with largest q values
        return max_action
