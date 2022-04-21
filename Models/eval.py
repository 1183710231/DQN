import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import ceil
import time
import collections

test_batch_size=16


def test(self):
    # self.set_eval()
    test_set=collections.OrderedDict()
    with open('../Algorithms/DQN/test.txt', 'r') as f:
        for line in f.readlines():
            h,r,t=line.split()
            try:
                test_set[h].append(t)
            except KeyError:
                test_set[h]=t
    # 记录数据集的基本长度参数
    dataset_len = len(test_set)
    rank_list_len = len(self.dataset.rank_list)
    steps = int(ceil(rank_list_len / self.test_batch_size))
    # 初始化性能指标
    mr = 0
    mrr = 0
    map = 0
    ts = time.time()
    # *******
    r_list = []
    # *******
    with torch.no_grad():
        n = 0
        for head in test_set.keys():
            score_list = []

            self.clear_text_buffer()
            for j, [text_feature, code_feature, count, flag] in enumerate(
                    DataLoader(self.dataset, batch_size=self.test_batch_size, shuffle=False)):
                score = self.test_step(j, text_feature, code_feature, count)
                for s, f in zip(score, flag):
                    score_list.append([s.item(), f.item()])
                if i == 0:
                    te = time.time()
                    clear_line()
                    print('\r<TEST> sample:{}/{} step:{}/{}, time={}s'.format(i + 1, dataset_len, j + 1, steps,
                                                                              int(te - ts)), end='')
            rank, ap = self.cal_rank(score_list)
            if rank >= 0:
                mrr += 1 / rank
                mr += rank
                map += ap
                n += 1
            else:
                r_list.append(i + 1)
            te = time.time()
            clear_line()
            print('\r<TEST> sample:{}/{}, mr={}|{:.2f}%, mrr={:.4f}, map={:.4f}, n={}, time={}s'.format
                  (i + 1, dataset_len, int(mr / n), mr / n / rank_list_len * 100, mrr / n, map / n, n, int(te - ts)),
                  end='')
        print('')
        # print(r_list)
    self.set_train()
    self.clear_buffer()
    return mr, mrr