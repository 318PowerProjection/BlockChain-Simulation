from queue import Queue
from MST import Mst
from Divide import DivideBand
from DDPG import DDPG
import numpy as np
import torch
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from params import max_episode, exploration_noise, noise_attenuation, save_interval, state_history, max_action, \
    delta_t, graph_update_interval, numpy_seed, load, load_number

# type = 1 delta_t事件
# type = 2 区块到达事件


class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w


class Chain:
    def __init__(self, tmpsource, tmpdest, node_count):
        self.block_finish_time = []
        self.block_arrive_time = []
        self.block_trans_time = []
        self.block_size = []
        self.block_count = 0        # 到达链路的区块数量
        self.cur_block = -1         # 已经传输完成的区块编号
        self.RTgraph = []
        self.BDgraph = []
        self.source = tmpsource     # int
        self.dest = tmpdest         # list
        self.lastblock_finish_time = 0.0
        self.node_count = node_count

    def upd(self, routingGraph, bandGraph):
        self.RTgraph = routingGraph
        self.BDgraph = bandGraph

    def newBlock(self, task):
        self.block_count += 1
        self.block_arrive_time.append(task.time)
        self.block_size.append(task.data)
        self.block_finish_time.append(-1)
        self.block_trans_time.append(-1)

    def check_remain(self):     # 0 不空闲 1 空闲
        if self.cur_block == -1:          # 没有传输任何区块
            return 1
        now = self.cur_block
        if self.block_finish_time[now] > env.next_delta_task_time:
            return 0
        return 1

    def transmission(self):    # 当前路由方案的传输时间
        tmp_count = 0.0
        tmp_reward = 0.0
        while self.cur_block < self.block_count-1:
            if not self.check_remain():
                break
            i = self.cur_block + 1
            transTime = self.getTransTime(self.block_size[i])
            self.block_trans_time[i] = transTime
            tmp_finish_time = max(self.block_arrive_time[i], self.lastblock_finish_time) + transTime
            if tmp_finish_time > env.next_delta_task_time:      # 如果下一次更改路由之前传不完当前区块的话就不传现在这个
                break
            self.block_finish_time[i] = tmp_finish_time
            self.lastblock_finish_time = self.block_finish_time[i]
            tmp_count += 1.0
            tmp_reward += self.block_finish_time[i] - self.block_arrive_time[i]
            self.cur_block += 1

        return tmp_count, tmp_reward

    def getTransTime(self, data):                 # 获得当前区块的最大传输时间 BFS
        vis = [0 for _ in range(self.node_count)]
        vis[self.source] = 1
        q = Queue(maxsize=0)
        q.put(self.source)
        trans_time_for_node = [0 for _ in range(self.node_count)]
        while not q.empty():
            now = q.get()
            for _next in self.RTgraph[now]:
                if not vis[_next]:
                    trans_time_for_node[_next] = trans_time_for_node[now] + data / self.BDgraph[now][_next]
                    q.put(_next)
                    vis[_next] = 1
        max_trans_time = 0
        for i in self.dest:
            max_trans_time = max(max_trans_time, trans_time_for_node[i])
        return max_trans_time


class Environment:
    def __init__(self):
        self.graph = []                     # init_graph
        self.band = []                      # init_graph
        self.source = []                    # init_graph
        self.dest = []                      # init_graph
        self.block_size = []                # init_graph
        self.edge = []                      # init_graph
        self.edge_count = 0                 # init_graph
        self.node_count = 0                 # init_graph
        self.chain = []                     # init_graph
        self.chain_count = 0                # init_graph
        self.now_transaction = 0            # init_env
        self.count_for_each_chain = []      # init_env
        self.next_delta_task_time = 0       # init_env
        self.transaction_trace = []         # init_trace
        self.event_trace = []               # init_trace
        self.route_graph = []               # RL calc
        self.band_graph = []                # RL calc

        self.state = []                     # init_env
        self.action = []                    # init_env
        self.next_state = []                # init_env
        self.reward = 0.0                   # init_env
        self.done = False                   # init_env
        self.noise = exploration_noise
        self.max_action = max_action

        self.total_block_count = 0
        self.FLAG = False
        self.edge_for_mst = []

        self.seed = int(sys.argv[1])
        # self.seed = 1
        self.result = 0.0
        critic_output = open('./Output/Loss/seed=%d/critic_loss.txt' % self.seed, 'w')
        actor_output = open('./Output/Loss/seed=%d/actor_loss.txt' % self.seed, 'w')
        critic_output.close()
        actor_output.close()

    def init_trace(self):
        trace_file = open('./Input/BlockTrace.txt', 'r')
        self.event_trace = []
        for line in trace_file.readlines():
            event_type, time, chain_id, data = list(map(eval, line.split()))
            self.event_trace.append(Event(event_type, time, chain_id, data))
            # type=2: time时刻在chainID号链到达一个大小为data的block
        trace_file.close()
        trace_file = open('./Input/TransactionTrace.txt', 'r')
        self.transaction_trace = []
        for line in trace_file.readlines():
            event_type, time, chain_id, data = list(map(eval, line.split()))
            self.transaction_trace.append(Event(event_type, time, chain_id, data))
        trace_file.close()
        for it in self.event_trace:
            if it.event_type == 2:
                self.total_block_count += 1

    def init_env(self):
        self.route_graph = []
        self.band_graph = []
        self.now_transaction = -1
        self.next_delta_task_time = 0
        self.count_for_each_chain = [0 for _ in range(self.chain_count)]

        self.state = [0 for _ in range(self.chain_count * state_history + self.chain_count)]
        self.action = [0 for _ in range(self.chain_count * self.edge_count)]
        self.next_state = self.state
        self.reward = 0.0
        self.done = False

        self.chain = []
        for i in range(self.chain_count):
            self.chain.append(Chain(self.source[i], self.dest[i], self.node_count))

        self.noise -= noise_attenuation
        self.noise = max(self.noise, 0)

        # 生成Output文件夹
        if not os.path.exists('./Output/Model/seed=%d/' % self.seed):
            os.mkdir('./Output/Model/seed=%d/' % self.seed)

    def init_graph(self):
        tmp_graph = []
        tmp_band = []
        readfile = open('./Input/graph_info.txt', 'r')
        content = readfile.readlines()
        self.node_count, self.chain_count = map(int, content[0].split())
        for line in content[1:self.node_count+1]:
            tmp_graph.append(list(map(int, line.split())))
        for line in content[self.node_count+1:self.node_count*2+1]:
            tmp_band.append(list(map(float, line.split())))
        node_file = content[self.node_count*2+1:]
        for i in range(self.chain_count):
            self.source.append(list(map(int, node_file[i+i].split()))[0])
            self.dest.append(list(map(int, node_file[i+i+1].split())))
        self.block_size = list(map(float, node_file[len(node_file)-1].split()))
        readfile.close()

        self.graph = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.band = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.edge = []
        for u in range(self.node_count):
            for i in range(len(tmp_graph[u])):
                v = tmp_graph[u][i]
                self.band[u][v] = tmp_band[u][i]
                self.graph[u][v] = 1                # graph 是无向图邻接矩阵
                if u < v:
                    self.edge_count += 1
                    self.edge.append(Edge(u, v, 0))


class Event:
    def __init__(self, event_type, time, chainID, data):
        self.event_type = event_type
        self.data = data
        self.time = time
        self.chainID = chainID

# 所有输出加上噪声限制到0,1
# 加上很小的正数1e-4


def convention_action(action):     # 规约化action 使用正态分布加噪声会导致有负数
    ret = np.array(action)
    np.clip(ret, 0.0, 1.0)
    for i in range(len(ret)):
        ret[i] += 1e-5
    return list(ret)


def convention_state(state):
    p = state_history-1
    for i in range(env.chain_count):
        state[p] /= 50.0
        p += state_history
    for j in range(state_history * env.chain_count, state_history * env.chain_count + env.chain_count):
        state[j] /= 10.0
    return state


def trans(env):         # 把action变成chaincount个邻接表
    edge_action = []
    p = 0
    for i in range(env.chain_count):
        tmp_edge = []
        for j in range(env.edge_count):
            tmp_edge.append(Edge(env.edge[j].u, env.edge[j].v, env.action[j+p]))
        edge_action.append(tmp_edge)
        p += env.edge_count
    return edge_action


def solve_delta(env):
    mst_for_divide = []
    env.action = agent.select_action(env.state)
    env.action = (env.action + np.random.normal(env.noise, env.noise, size=env.edge_count * env.chain_count))
    env.action = convention_action(env.action)
    env.edge_for_mst = trans(env)

    for i in range(env.chain_count):
        mst = Mst(env.node_count, env.edge_for_mst[i])
        tree = mst.kruskal()   # 邻接表形式 无向边
        mst_for_divide.append(tree)

    divide = DivideBand(env.node_count, mst_for_divide, env.band, env.source, env.dest, env.block_size)
    env.route_graph = mst_for_divide         # 最小生成树实际上就是路由方案
    env.band_graph = divide.binary_search()   # 分配带宽
    for i in range(env.chain_count):
        env.chain[i].upd(env.route_graph[i], env.band_graph[i])
    env.next_delta_task_time += delta_t

    for i in range(env.node_count):
        for j in range(env.node_count):
            cost = 0.0
            for p in range(env.chain_count):
                cost += env.band_graph[p][i][j]
            if cost > env.band[i][j]+0.1:
                debug_print()
                break


def update_state(env):            # state: [chain1_history_state, chain2_history_state, chain1_buffer, chain2_buffer]
    count_for_each_chain = [0 for _ in range(env.chain_count)]
    env.now_transaction += 1            # 每次now_transaction会停留在一个type=1的位置
    while env.transaction_trace[env.now_transaction].event_type == 2:
        chainID = env.transaction_trace[env.now_transaction].chainID
        count_for_each_chain[chainID] += 1          # 计算每个链在一个间隔内的交易流量
        env.now_transaction += 1

    env.next_state = env.state
    for i in range(env.chain_count):
        for p in range(state_history-1):
            env.next_state[p+i*state_history] = env.next_state[p+1+i*state_history]
        env.next_state[i*state_history+state_history-1] = count_for_each_chain[i]

    for i in range(env.chain_count):
        env.next_state[env.chain_count*state_history+i] = env.chain[i].block_count - env.chain[i].cur_block

    env.next_state = convention_state(env.next_state)
        
    tmp_count = 0
    for i in range(env.chain_count):
        tmp_count += env.chain[i].cur_block+1       # block_count是区块数量
    if tmp_count == env.total_block_count:          # tmp_count计算已经传输完成的区块数量
        env.done = True


def show(it):
    if it == 0:
        return
    number = env.seed

    x, y = [], []
    c_loss_file = open('./Output/Loss/seed=%d/critic_loss.txt' % number, 'r')
    for line in c_loss_file.readlines():
        data_x, data_y = list(map(eval, line.split()))
        x.append(data_x)
        y.append(data_y)
    plt.title("Critic Loss")
    plt.xlabel("Step")
    plt.ylabel("Critic Loss")
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig('./Output/Graph/seed=%d/critic_loss.png' % number)
    plt.close()

    x, y = [], []
    c_loss_file = open('./Output/Loss/seed=%d/actor_loss.txt' % number, 'r')
    for line in c_loss_file.readlines():
        data_x, data_y = list(map(eval, line.split()))
        x.append(data_x)
        y.append(data_y)
    plt.title("Actor Loss")
    plt.xlabel("Step")
    plt.ylabel("Actor Loss")
    plt.grid(True)
    plt.plot(x, y)
    plt.savefig('./Output/Graph/seed=%d/actor_loss.png' % number)
    plt.close()

    print("====================================")
    print("Graph is updated...")
    print("====================================")


def debug_print():
    debug_file = open("debug_info.txt", 'w')

    for i in range(env.node_count):
        for j in range(env.node_count):
            cost = 0.0
            for p in range(env.chain_count):
                cost += env.band_graph[p][i][j]
            print('%.2f' % cost, end='\t', file=debug_file)
        print(file=debug_file)
    print(file=debug_file)
    for i in range(env.node_count):
        for j in range(env.node_count):
            print('%.2f' % env.band[i][j], end='\t', file=debug_file)
        print(file=debug_file)
    print(file=debug_file)

    for it in range(env.chain_count):
        g1 = env.chain[it].RTgraph
        g2 = env.chain[it].BDgraph
        for i in range(len(g1)):
            print(i, ': ', end=' ', file=debug_file)
            for j in range(len(g1[i])):
                print(g1[i][j], end=' ', file=debug_file)
            print('', file=debug_file)
        print('', file=debug_file)
        for i in range(len(g2)):
            for j in range(len(g2[i])):
                print('\t%.2lf' % g2[i][j], end=' ', file=debug_file)
            print('', file=debug_file)

    cnt = 0
    for it in range(env.chain_count):
        for i in range(env.chain[it].block_count):
            print('%d\t%.2f\t%.2f' % (cnt, env.chain[it].block_arrive_time[i], env.chain[it].block_finish_time[i]), file=debug_file)
            cnt += 1

    debug_file.close()


def check_done():
    cnt = 0
    for i in range(env.chain_count):
        cnt += env.chain[i].cur_block+1
    if cnt == env.total_block_count:
        return 1
    return 0


if __name__ == '__main__':
    env = Environment()
    env.init_trace()
    env.init_graph()

    np.random.seed(numpy_seed)
    torch.manual_seed(env.seed)

    state_dim = state_history * env.chain_count + env.chain_count
    action_dim = env.edge_count * env.chain_count
    agent = DDPG(state_dim, action_dim, max_action)

    if load:
        agent.load_model(env.seed, load_number)
        env.noise = 0.0

    for it in range(max_episode+1):
        env.init_env()
        block_in_interval = 0.0
        reward_in_interval = 0.0

        for cnt in range(len(env.event_trace)):
            task = env.event_trace[cnt]
            if task.event_type == 1:                # deltaT事件
                if cnt == 0:                  # 如果是开始状态则只更新路由
                    solve_delta(env)
                    continue

                env.reward = -reward_in_interval / block_in_interval    # reward是平均时延 优化目标是越小越好，但是reward越大越好
                block_in_interval = 0.0
                reward_in_interval = 0.0

                if env.chain[2].cur_block == 214:
                    xx = 0

                update_state(env)               # calc next_state 只有下一个δt的时候才能知道这段内的reward和next state
                agent.replay_buffer.put([env.state, env.action, env.reward, env.next_state, np.float(env.done)])
                if env.done:
                    break

                env.state = env.next_state
                env.reward = 0.0
                solve_delta(env)
                # debug_print()
                for i in range(env.chain_count):     # 在堵塞的情况下，更新路由后应该接着传输之前没传完的区块
                    tmp_count, tmp_reward = env.chain[i].transmission()
                    block_in_interval += tmp_count
                    reward_in_interval += tmp_reward

            if task.event_type == 2:              # 新区块达到事件
                env.chain[task.chainID].newBlock(task)
                tmp_count, tmp_reward = env.chain[task.chainID].transmission()
                block_in_interval += tmp_count
                reward_in_interval += tmp_reward

        if not load and it > 50 or load:
            agent.update(env.seed)

        if it % save_interval == 0:
            agent.save(env.seed, it)
        if it % graph_update_interval == 0:
            show(it)
        print("Iteration:\t%d done" % it)
        # if it == 100:
        #     debug_print()

    agent.save_model(env.seed, max_episode)
    show(max_episode)

