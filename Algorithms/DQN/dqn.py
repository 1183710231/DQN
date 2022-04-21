import pickle

import numpy as np
import torch
import torch.optim as optim
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import ceil
import time
import collections
import tensorflow as tf
#忽略
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


from dqn_step import dqn_step
from Models.QNet_dqn import QNet_dqn
from Common.replay_memory import Memory
from Utils.env_util import get_env_info
from Utils.file_util import check_path
from Utils.torch_util import device, FLOAT, LONG


class DQN:
    # batch_size:每次只输入一定数量的训练样本对模型进行训练，跑完一次 epoch（全数据集）所需的迭代次数减少



    # 首先初始化Memory D，它的容量为N;
    # 初始化Q网络，随机生成权重ω;
    # 初始化target Q网络，权重为ω−=ω;
    # 循环遍历episode = 1, 2, …, M:
    # 初始化initialstateS1;
    # 循环遍历step = 1, 2,…, T:
    #     用ϵ−greedy策略生成action at：以ϵ概率选择一个随机的action，或选择at = maxaQ(St, a;ω);
    #     执行action at，接收reward rt及新的state St + 1;
    #     将transition样本(St, at, rt, St + 1)存入D中；
    #     从D中随机抽取一个minibatch的transitions(Sj, aj, rj, Sj + 1)；
    #     令yj = rj，如果j + 1步是terminal的话，否则，令yj = rj + γmaxa′Q(St + 1, a′;ω−)；
    #     对(yj−Q(St, aj;ω))2关于ω使用梯度下降法进行更新；
    #     每隔C steps更新target Q网络，ω−=ω。

    def __init__(self,
                 env_id,
                 # 状态池大小
                 memory_size=1000000,
                 # 可调，探索步数小于值时随机选取action，不能太小
                 explore_size=10000,
                 #收敛快慢。每次迭代多少步
                 step_per_iter=3000,
                 #Q网络学习率
                 lr_q=1e-5,
                 #
                 gamma=0.99,
                 #修改 128 -》 64
                 batch_size=128,
                 #池中最小步数
                 min_update_step=1000,
                 #0.9选取贪心action，0.1随机选取
                 epsilon=0.90,
                 # 延迟更新机制
                 update_target_gap=50,
                 #
                 seed=1,
                 #
                 model_path=None
                 ):
        self.env_id = env_id
        self.memory = Memory(size=memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.lr_q = lr_q
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_target_gap = update_target_gap
        self.epsilon = epsilon
        self.seed = seed
        self.model_path = model_path
        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, dim_state, self.num_actions ,self.ent_dict= get_env_info(self.env_id)

        path = 'param_ent64_rel64_TransE.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data2 = data.get('weights').get('entityEmbed')
        self.matrix = np.array(data2)

        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        # initialize networks
        # ？？？这为什么有两个网络
        self.value_net = QNet_dqn(self.env.num_ents, dim_state, self.num_actions).to(device)
        self.value_net_target = QNet_dqn(self.env.num_ents, dim_state, self.num_actions).to(device)

        # load model if necessary
        if self.model_path:
            print("Loading Saved Model {}_dqn.p".format(self.env_id))
            with open('{}/{}_dqn.p'.format(self.model_path, self.env_id), "rb") as f:
                self.value_net = pickle.load(f)

        self.value_net_target.load_state_dict(self.value_net.state_dict())
        # 神经网络优化器
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr_q)

    def choose_action(self, state, flag=False):
        state = FLOAT(self.matrix[[state]]).to(device)
        if np.random.uniform() <= self.epsilon or flag:
            with torch.no_grad():
                action = self.value_net.get_action(state)
                # wkr修改 打印
                # print(action)
                # print(type(action))
            action = action.cpu().numpy()[0]
            # print(action)
            # print(type(action))
        else:  # choose action greedy
            action = np.random.randint(0, self.num_actions)
        return action

    def test(self):
        # self.set_eval()
        test_set = dict()
        # FB15K -> SVKG
        with open('newTest.txt', 'r') as f:
            for line in f.readlines():
                h, r, t = line.rstrip('\n').split('\t')
                try:
                    test_set[self.ent_dict[h]].append(self.ent_dict[t])
                except KeyError:
                    test_set[self.ent_dict[h]] = [self.ent_dict[t]]
        # 记录数据集的基本长度参数
        dataset_len = len(test_set)
        # 初始化性能指标
        mr = 0
        mrr = 0
        map = 0
        # *******
        r_list = []
        # *******
        with torch.no_grad():
            n = 0
            for head in test_set.keys():
                state = FLOAT(self.matrix[[head]]).to(device)
                with torch.no_grad():
                    q_values = self.value_net.forward(state, )
                score_list=q_values.cpu().numpy()
                score_list = tf.argsort(score_list,direction='DESCENDING')
                score_list = list(np.array(score_list).flatten())
                index = 0
                ap = 0
                rank = None
                # for i in range(self.num_actions):
                #     if score_list[i] in test_set[head]:
                #         if rank==None:
                #             # 此处加一防止出0
                #             rank=i+1
                #         index+=1
                #         if index<=2:
                #             ap+=index/(i+1)
                #             count+=1
                for i in test_set[head]:
                    num=score_list.index(i)+1
                    if rank==None:
                        rank=num
                    elif rank>num:
                        rank=num
                    index+=1
                    ap+=index/(num)
                ap /= len(test_set[head])
                mrr += 1 / rank
                mr += rank
                map += ap
                n += 1
            print('<TEST> sample:14952/{}, mr={}|{:.2f}%, mrr={:.4f}, map={:.4f}, n={}'.format
                       (dataset_len, int(mr / n), mr / n / self.num_actions * 100, mrr / n, map / n, n))
            # print(r_list)
        return mr, mrr, map

    def eval(self, i_iter):
        for i in range(10):
            """evaluate model"""
            state = self.env.reset()
            print(f"state num: {state}")
            test_reward = 0
            eval_iter = 0
            while True:
                eval_iter += 1
                # wkr修改，测试时不随机选择
                action = self.choose_action(state, True)
                state, reward, done = self.env.step(action)
                test_reward += reward
                if done:
                    break
            print(f"Iter: {i_iter}, test Reward: {test_reward}, avg_Reward: {test_reward/eval_iter}, eval_iter: {eval_iter}")

    def learn(self, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        while num_steps < self.step_per_iter:
            state = self.env.reset()
            episode_reward = 0

            for t in range(5):
                if global_steps < self.explore_size:  # explore
                    action = self.env.action_sample()
                else:  # choose according to target net
                    action = self.choose_action(state)
                ###wkr修改 有监督学习
                # if (global_steps < 1000000000) and (np.random.uniform() <= 0.3):
                #     action = self.env.supervised_action(state)
                ###
                next_state, reward, done = self.env.step(action)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                # ？？？这个memory作用是干嘛的
                self.memory.push(state, action, reward, next_state, mask, None)
                episode_reward += reward
                global_steps += 1
                num_steps += 1
                if (global_steps >= self.min_update_step) and (len(self.memory) >= self.batch_size):
                    batch = self.memory.sample(self.batch_size)  # random sample batch
                    self.update(batch)

                if global_steps % self.update_target_gap == 0:
                    # ？？？这个state_dict()是干什么的，更新目标表吗
                    self.value_net_target.load_state_dict(
                        self.value_net.state_dict())

                if done or num_steps >= self.step_per_iter:
                    break

                state = next_state

            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)


        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_episode_reward'] = max_episode_reward
        log['min_episode_reward'] = min_episode_reward

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}")

    def update(self, batch):
        # batch_state = LONG(batch.state).to(device)
        batch_action = LONG(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        # batch_next_state = LONG(batch.next_state).to(device)
        # to(device)是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        batch_mask = FLOAT(batch.mask).to(device)

        # print(self.matrix.shape)
        # print(batch.state)
        # input()
        batch_state = FLOAT(self.matrix[list(batch.state)]).to(device)
        batch_next_state = FLOAT(self.matrix[list(batch.next_state)]).to(device)

        # 神经网络更新
        dqn_step(self.value_net, self.optimizer, self.value_net_target, batch_state, batch_action,
                 batch_reward, batch_next_state, batch_mask, self.gamma)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        print("Saving model {}/{}_dqn.p.".format(save_path, self.env_id))
        with open('{}/{}_dqn.p'.format(save_path, self.env_id), 'wb') as f:
            pickle.dump(self.value_net, f)
