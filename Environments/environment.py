import random
import pickle

class Environment:
    # 此处修改 64 -》 100 -》64
    def __init__(self, env_id='FB15k', dim_state=64):
        self.env_id = env_id
        self.dim_state = dim_state
        self.ent_dict = dict()
        self.rel_dict = dict()
        self.num_ents = 0
        self.num_rels = 0
        self.h2r = dict()
        # 存储所有 头实体*10000+关系 -》 尾实体序列
        self.hr2t = dict()
        self.h2rt = dict()
        self.ht2r = dict()
        self.ht_set = set()
        self.tri_set = set()
        # 头实体*10000+关系 的list
        self.pair_set = set()
        self.model = None
        self.matrix = None
        self.cur_state = None
        self.cur_relate= None
        self.p_r = 1
        self.p_e = 1


    def load(self):
        # load knowledge graph file
        triples = list()
        f = open('../../Algorithms/DQN/param_ent64_rel64_TransE.pkl', 'rb')
        data2 = pickle.load(f)
        entlist = data2.get('entlist')
        self.num_ents=len(entlist)
        self.ent_dict = dict(zip(entlist, range(self.num_ents)))
        # FB15K -> SVKG
        with open('../../Environments/KGs/{}/newTrain.txt'.format(self.env_id), 'r',
                  encoding='utf-8') as f:
        # with open('../../Environments/KGs/{}/freebase_mtr100_mte100-train.txt'.format(self.env_id), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # FB15K -> SVKG
                h, r, t = line.rstrip('\n').split('\t')
                try:
                    self.rel_dict[r]
                except KeyError:
                    self.rel_dict[r] = self.num_rels
                    self.num_rels += 1
                h_id, r_id, t_id = self.ent_dict[h], self.rel_dict[r], self.ent_dict[t]
                triples.append([h_id, r_id, t_id])
                # if len(triples) > 2000:
                #     break
        print('Env_id={}, num_state={}, num_action={}'.format(self.env_id, self.num_ents, self.num_rels))
        # adjacency construction
        while self.p_r <= self.num_rels:
            self.p_r *= 10
        while self.p_e <= self.num_ents:
            self.p_e *= 10
        for tri in triples:
            h_id, r_id, t_id = tri[0], tri[1], tri[2]
            try:
                self.h2r[h_id].append(r_id)
            except KeyError:
                self.h2r[h_id] = [r_id]
            try:
                self.hr2t[h_id * self.p_r + r_id].append(t_id)
            except KeyError:
                self.hr2t[h_id * self.p_r + r_id] = [t_id]
            try:
                self.h2rt[h_id].append([r_id, t_id])
            except KeyError:
                self.h2rt[h_id] = [[r_id, t_id]]
            try:
                self.ht2r[h_id * self.p_r + t_id].append(r_id)
            except KeyError:
                self.ht2r[h_id * self.p_r + t_id] = [r_id]
            self.tri_set.add(h_id * self.p_r * self.p_e + r_id * self.p_e + t_id)
            self.pair_set.add(h_id * self.p_r + r_id)
            # KR ADD
            self.ht_set.add(h_id * self.p_r + t_id)


    def seed(self, n):
        random.seed(n)

    def reset(self):
        self.cur_state = random.randint(0, self.num_ents - 1)
        return self.cur_state

    def cal_reward(self, h_id , r_id , t_id):
        # h_id = state
        # r_id = action
        # # 动作正确
        # if (h_id * self.p_r + r_id) in self.pair_set:
        #     reward = 10
        #     done = False
        # #未选中并且到达了终止节点
        # elif (action == self.num_rels) and (h_id * self.p_r + r_id not in (self.hr2t.keys())):
        #     reward = 10
        #     done = True
        # #选择错误
        # else:
        #     reward = -1
        #     done = True
        # return reward, done
        # h_id = state
        # t_id = action
        # # 动作正确
        # # print('h_id={},t_id={},h_id * self.p_r + t_id={}'.format(state,action,h_id * self.p_r + t_id))
        if (h_id * self.p_r *self.p_e + r_id * self.p_e + t_id) in self.tri_set:
            reward = 10
            done = False
        # 未选中并且到达了终止节点
        elif (t_id == self.num_ents):
            reward = 10
            done = True
        # 选择错误
        else:
            reward = -1
            done = True
        return reward, done



    def step(self, action):
        reward, done = self.cal_reward(self.cur_state,self.cur_relate, action)
        self.cur_state = action
        try:
            # 此处因是 头实体 -》 关系 ，所以随机选择一个尾实体
            self.cur_relate = random.sample(self.h2r[self.cur_state], 1)[0]
        except KeyError:
            while True:
                self.cur_state = random.randint(0, self.num_ents - 1)
                if self.cur_state in self.h2r.keys():
                    self.cur_relate = random.sample(self.h2r[self.cur_state], 1)[0]
                    break
        return self.cur_state, reward, done

    def action_sample(self):
        # try:
        #     sam = random.sample(self.h2rt[self.cur_state], 1)[0]
        #     r_id = sam[0]
        #     t_id = sam[1]
        #     action = r_id * self.num_ents + t_id
        #     return action
        # except KeyError:
        #     action = self.num_ents * self.num_rels
        #     return action
        action = random.randint(0, self.num_rels)
        return action

    def supervised_action(self, h):
        try:
            r = random.sample(self.h2r[h], 1)[0]
        except KeyError:
            r = self.num_rels
        return r


