from queue import Queue
from shortestpath import Shortest_path
from MST import Mst
from Divide import DivideBand
from DDPG import DDPG
import numpy as np
import sys
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from params import max_episode, exploration_noise, save_interval, state_history, max_action, delta_t, \
    graph_update_interval, numpy_seed


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
        self.block_count = 0  # 到达链路的区块数量
        self.cur_block = -1  # 已经传输完成的区块编号
        self.RTgraph = []
        self.BDgraph = []
        self.source = tmpsource  # int
        self.dest = tmpdest  # list
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

    def check_remain(self):  # 0 不空闲 1 空闲
        if self.cur_block == -1:  # 没有传输任何区块
            return 1
        now = self.cur_block
        if self.block_finish_time[now] > env.next_delta_task_time:
            return 0
        return 1

    def transmission(self):  # 当前路由方案的传输时间
        tmp_count = 0.0
        tmp_reward = 0.0
        while self.cur_block < self.block_count - 1:
            if not self.check_remain():
                break
            i = self.cur_block + 1
            transTime = self.getTransTime(self.block_size[i])
            self.block_trans_time[i] = transTime
            self.block_finish_time[i] = max(self.block_arrive_time[i], self.lastblock_finish_time) + transTime
            self.lastblock_finish_time = self.block_finish_time[i]
            tmp_count += 1.0
            tmp_reward += self.block_finish_time[i] - self.block_arrive_time[i]
            self.cur_block += 1
            if self.block_finish_time[i] > env.next_delta_task_time:
                break
        return tmp_count, tmp_reward

    def getTransTime(self, data):  # 获得当前区块的最大传输时间 BFS
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
        self.graph = []  # init_graph
        self.band = []  # init_graph
        self.source = []  # init_graph
        self.dest = []  # init_graph
        self.block_size = []  # init_graph
        self.edge = []  # init_graph
        self.edge_count = 0  # init_graph
        self.node_count = 0  # init_graph
        self.chain = []  # init_graph
        self.chain_count = 0  # init_graph
        self.now_transaction = 0  # init_env
        self.count_for_each_chain = []  # init_env
        self.next_delta_task_time = 0  # init_env
        self.transaction_trace = []  # init_trace
        self.event_trace = []  # init_trace
        self.route_graph = []  # RL calc
        self.band_graph = []  # RL calc

        self.state = []  # init_env
        self.action = []  # init_env
        self.next_state = []  # init_env
        self.reward = 0.0  # init_env
        self.done = False  # init_env
        self.noise = exploration_noise
        self.max_action = max_action

        self.total_block_count = 0
        self.FLAG = False
        self.edge_for_mst = []

        self.seed = int(sys.argv[1])
        # self.seed = 1
        self.result = []
        self.trans_time_result = []
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

        self.state = [0 for _ in range(state_history + 1)]
        self.action = [0 for _ in range(self.edge_count)]
        self.next_state = self.state
        self.reward = 0.0
        self.done = False

        self.chain = []
        for i in range(self.chain_count):
            self.chain.append(Chain(self.source[i], self.dest[i], self.node_count))

    def init_graph(self):
        tmp_graph = []
        tmp_band = []
        readfile = open('./Input/graph_info.txt', 'r')
        content = readfile.readlines()
        self.node_count, self.chain_count = map(int, content[0].split())
        for line in content[1:self.node_count + 1]:
            tmp_graph.append(list(map(int, line.split())))
        for line in content[self.node_count + 1:self.node_count * 2 + 1]:
            tmp_band.append(list(map(float, line.split())))
        node_file = content[self.node_count * 2 + 1:]
        for i in range(self.chain_count):
            self.source.append(list(map(int, node_file[i + i].split()))[0])
            self.dest.append(list(map(int, node_file[i + i + 1].split())))
        self.block_size = list(map(float, node_file[len(node_file) - 1].split()))
        readfile.close()

        self.graph = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.band = [[0 for _i in range(self.node_count)] for _j in range(self.node_count)]
        self.edge = []
        for u in range(self.node_count):
            for i in range(len(tmp_graph[u])):
                v = tmp_graph[u][i]
                self.band[u][v] = tmp_band[u][i]
                self.graph[u][v] = 1  # graph 是无向图邻接矩阵
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


def convention_action(action):  # 规约化action 使用正态分布加噪声会导致有负数
    ret = np.array(action)
    np.clip(ret, 0.0, 1.0)
    for i in range(len(ret)):
        ret[i] += 1e-5
    return list(ret)


def trans(env):         # 把action变成chaincount个邻接表
    edge_action = []
    tmp_edge = []
    for j in range(env.edge_count):
        tmp_edge.append(Edge(env.edge[j].u, env.edge[j].v, env.action[j]))
    edge_action.append(tmp_edge)
    return edge_action


def trans2(nodeCount, graph):
    result = []
    for i in range(nodeCount):
        for j in range(nodeCount):
            if i < j and graph[i][j] == 1:
               result.append(Edge(i, j, 1))
    return result


def solve_delta(env):
    mst_for_divide = []
    env.action = agent.select_action(env.state)
    env.action = (env.action + np.random.normal(env.noise, env.noise, size=env.edge_count))
    env.action = convention_action(env.action)
    env.edge_for_mst = trans(env)

    mst = Mst(env.node_count, env.edge_for_mst[0])  # 第0条链变成DDPG
    tree = mst.kruskal()   # 邻接表形式 无向边
    mst_for_divide.append(tree)

    for i in range(1, env.chain_count):             # 后两条边变成最短路
        routing = Shortest_path(env.graph, env.chain[i].source, env.chain[i].dest)
        routing.main()
        tmp_graph = routing.result
        tmp_graph = trans2(env.node_count, tmp_graph)
        mst = Mst(env.node_count, tmp_graph)
        tree = mst.kruskal()   # 邻接表形式 无向边
        mst_for_divide.append(tree)

    divide = DivideBand(env.node_count, mst_for_divide, env.band, env.source, env.dest, env.block_size)
    env.route_graph = mst_for_divide         # 最小生成树实际上就是路由方案
    env.band_graph = divide.binary_search()   # 分配带宽
    for i in range(env.chain_count):
        env.chain[i].upd(env.route_graph[i], env.band_graph[i])
    env.next_delta_task_time += delta_t


def show(it):
    if it == 0:
        return
    x = np.arange(0, it/save_interval)
    y = np.array(env.result)
    compare_y = np.array([51.51]*len(x))

    # print(x)
    # print(y)

    for i in range(len(x)):
        x[i] *= save_interval
    plt.title("Average Delay")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay")
    # plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.plot(x, y)
    plt.plot(x, compare_y)
    plt.savefig('./Output/Graph/seed=%d/average_delay.png' % env.seed)
    plt.close()
    print("====================================")
    print("Graph is updated...")
    print("====================================")


def show_tans_time(it):
    if it == 0:
        return
    x = np.arange(0, it/save_interval)
    y = np.array(env.trans_time_result)
    compare_y = np.array([0.9]*len(x))

    # print(x)
    # print(y)

    for i in range(len(x)):
        x[i] *= save_interval
    plt.title("Average Delay")
    plt.xlabel("Episode")
    plt.ylabel("Average Delay")
    # plt.figure(figsize=(12, 6))
    plt.grid(True)
    plt.plot(x, y)
    plt.plot(x, compare_y)
    plt.savefig('./Output/Graph/seed=%d/average_trans_time.png' % env.seed)
    plt.close()
    print("====================================")
    print("Graph is updated...")
    print("====================================")


def debug_print():
    debug_file = open("./Output/debug_info.txt", 'w')

    for it in range(env.chain_count):
        g1 = env.chain[it].RTgraph
        g2 = env.chain[it].BDgraph
        for i in range(len(g1)):
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


def debug_show_trans_time(it):
    file = open('./Output/debug/trans_time%d.txt' % it, 'w')
    for i in range(env.chain_count):
        cnt = 0
        for j in range(env.chain[i].block_count):
            cnt += 1
            print("%d\t%.2f\t%.2f\t%.2f" %
                  (cnt, env.chain[i].block_arrive_time[j],
                   env.chain[i].block_finish_time[j], env.chain[i].block_trans_time[j]),
                  file=file)
    file.close()
    return


if __name__ == '__main__':
    env = Environment()
    env.init_trace()
    env.init_graph()

    np.random.seed(numpy_seed)
    torch.manual_seed(env.seed)

    state_dim = state_history + 1
    action_dim = env.edge_count
    agent = DDPG(state_dim, action_dim, max_action)

    for it in range(max_episode+1):
        if it % save_interval != 0:
            continue
        agent.load(env.seed, it)
        env.init_env()

        for cnt in range(len(env.event_trace)):
            task = env.event_trace[cnt]
            if task.event_type == 1:                # deltaT事件
                solve_delta(env)
                env.next_delta_task_time += delta_t
                for i in range(env.chain_count):     # 在堵塞的情况下，更新路由后应该接着传输之前没传完的区块
                    env.chain[i].transmission()
            if task.event_type == 2:              # 新区块达到事件
                env.chain[task.chainID].newBlock(task)
                env.chain[task.chainID].transmission()

        if it % graph_update_interval == 0:
            show(it)
            show_tans_time(it)

        total_reward = 0.0
        total_trans_time = 0.0
        total_count = 0.0
        for i in range(env.chain_count):
            for j in range(env.chain[i].block_count):
                total_reward += env.chain[i].block_finish_time[j] - env.chain[i].block_arrive_time[j]
                total_trans_time += env.chain[i].block_trans_time[j]
                if env.chain[i].block_finish_time[j] < 0:
                    print("Error")
                    exit()
            total_count += env.chain[i].block_count
        env.result.append(total_reward/total_count)
        env.trans_time_result.append(total_trans_time/total_count)

        print("Iteration:\t%d done" % it)
        # debug_show_trans_time(it)

        # if it == 100:
        #    debug_print()

