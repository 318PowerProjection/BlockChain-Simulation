from queue import Queue
from MST import Mst
from Divide import DivideBand
from DDPG import DDPG
import numpy as np
import sys
import matplotlib.pyplot as plt
from params import max_episode, exploration_noise, noise_attenuation, save_interval, state_history, max_action, delta_t, graph_update

# type = 1 delta_t事件
# type = 2 区块到达事件


class Edge:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w


class Chain:
    def __init__(self, tmpsource, tmpdest, node_count):
        self.blockFinishTime = list()
        self.blockArriveTime = list()
        self.blockSize = list()
        self.blockCount = 0
        self.curBlock = 0       # 已经传输完成的区块数量
        self.RTgraph = list()
        self.BDgraph = list()
        self.source = tmpsource  # int
        self.dest = tmpdest         # list
        self.lastBlockFinishTime = 0.0
        self.node_count = node_count
        self.FLAG = False

    def upd(self, routingGraph, bandGraph):
        self.RTgraph = routingGraph
        self.BDgraph = bandGraph

    def newBlock(self, task):
        self.blockCount += 1
        self.blockArriveTime.append(task.time)
        self.blockSize.append(task.data)
        self.blockFinishTime.append(-1)

    def check_remain(self, lastDeltaTaskTime):     # 0 不空闲 1 空闲
        if self.curBlock == 0:
            return 1
        now = self.curBlock - 1
        if self.blockFinishTime[now] > lastDeltaTaskTime:
            return 0
        return 1

    def transmission(self, lastDeltaTaskTime):    # 当前路由方案的传输时间上
        tmp_count = 0.0
        tmp_reward = 0.0
        while self.curBlock < self.blockCount:
            if not self.check_remain(lastDeltaTaskTime):
                break
            i = self.curBlock
            transTime = self.getTransTime(self.blockSize[i])
            self.blockFinishTime[i] = max(self.blockArriveTime[i], self.lastBlockFinishTime) + transTime
            self.lastBlockFinishTime = self.blockFinishTime[i]
            tmp_count += 1.0
            tmp_reward += self.blockFinishTime[i] - self.blockArriveTime[i]
            self.curBlock += 1
            if self.blockFinishTime[i] > lastDeltaTaskTime:
                break
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

                    if self.BDgraph[now][_next] == 0:
                        self.FLAG = True
                        return 10000

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
        self.edge = []                      # init_graph
        self.edge_count = 0                 # init_graph
        self.node_count = 0                 # init_graph
        self.chain = []                     # init_graph
        self.chain_count = 0                # init_graph
        self.now_transaction = 0            # init_env
        self.count_for_each_chain = []      # init_env
        self.lastDeltaTaskTime = 0          # init_env
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
        self.result = []

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
        self.lastDeltaTaskTime = delta_t
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
        self.noise = max(self.noise, 1e-4)

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
        readfile.close()

        self.graph = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.band = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.edge = []
        for u in range(self.node_count):
            for i in range(len(tmp_graph[u])):
                v = tmp_graph[u][i]
                self.graph[u][v] = 1                # graph 是无向图邻接矩阵
                self.band[u][v] = tmp_band[u][i]
                if u < v:
                    self.edge_count += 1
                    self.edge.append(Edge(u, v, 0))


class Event:
    def __init__(self, event_type, time, chainID, data):
        self.event_type = event_type
        self.data = data
        self.time = time
        self.chainID = chainID


def convention(action):     # 规约化action
    min_value = 0.0
    for i in range(len(action)): min_value = min(min_value, action[i])
    if min_value < 0:
        for i in range(len(action)): action[i] -= min_value
    max_value = 0.0
    for i in action:
        max_value = max(max_value, i)
    weight = max_action / max_value
    for i in range(len(action)):
        action[i] = action[i] * weight
        # if abs(action[i]-0.0) < 1e-5:
        #     action[i] = max_action
    return action


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
    env.action = convention(env.action)
    env.edge_for_mst = trans(env)

    for i in range(env.chain_count):
        mst = Mst(env.node_count, env.edge_for_mst[i])
        tree = mst.kruskal()   # 邻接表形式 无向边
        mst_for_divide.append(tree)

    divide = DivideBand(env.node_count, mst_for_divide, env.band, env.source, env.dest)
    env.route_graph = mst_for_divide         # 最小生成树实际上就是路由方案
    env.band_graph = divide.binary_search()   # 分配带宽
    for i in range(env.chain_count):
        env.chain[i].upd(env.route_graph[i], env.band_graph[i])


def show(number, result):
    if it == 0:
        return
    x = np.arange(0, max_episode/save_interval-1)
    for i in range(len(x)):
        x[i] *= save_interval
    plt.title("Average Delay")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay")
    plt.plot(x, result)
    plt.savefig('./Output/Graph/seed=%d/average_delay.png' % number)

    print("====================================")
    print("Graph is updated...")
    print("====================================")


if __name__ == '__main__':
    env = Environment()
    env.init_trace()
    env.init_graph()

    state_dim = state_history * env.chain_count + env.chain_count
    action_dim = env.edge_count * env.chain_count
    agent = DDPG(state_dim, action_dim, max_action, env.seed)

    for it in range(1, max_episode):
        if it % save_interval != 0:
            continue
        else:
            agent.load(env.seed, it)

        env.init_env()
        block_in_interval = 0.0
        reward_in_interval = 0.0

        for cnt in range(len(env.event_trace)):
            task = env.event_trace[cnt]
            if task.event_type == 1:                # deltaT事件
                if block_in_interval >= 1.0:
                    env.reward = reward_in_interval / block_in_interval
                else:
                    env.reward = 0.0

                block_in_interval = 0.0
                reward_in_interval = 0.0

                solve_delta(env)
                env.lastDeltaTaskTime += delta_t
                for i in range(env.chain_count):     # 在堵塞的情况下，更新路由后应该接着传输之前没传完的区块
                    env.chain[i].transmission(env.lastDeltaTaskTime)

            if task.event_type == 2:              # 新区块达到事件
                env.chain[task.chainID].newBlock(task)
                tmp_count, tmp_reward = env.chain[task.chainID].transmission(env.lastDeltaTaskTime)
                block_in_interval += tmp_count
                reward_in_interval += tmp_reward

        total_reward = 0.0
        total_count = 0.0
        for i in range(env.chain_count):
            for j in range(env.chain[i].blockCount):
                total_reward += env.chain[i].blockFinishTime[j] - env.chain[i].blockArriveTime[j]
            total_count += env.chain[i].blockCount

        env.result.append(total_reward/total_count)
        print("Iteration:\t%d" % it)

    show(env.seed, env.result)
    result_file = open('./Output/Graph/seed=%d/average_delay.txt' % env.seed, 'w')
    for i in range(len(env.result)):
        print(env.result[i], file=result_file)
    result_file.close()
