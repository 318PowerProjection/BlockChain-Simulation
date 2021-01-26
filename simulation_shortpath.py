from queue import Queue
from shortestpath import Shortest_path
from MST import Mst
from Divide import DivideBand
# from read import readInfo
import math


def readInfo():
    readfile = open('./Input/graph_info.txt', 'r')
    content = readfile.readlines()
    nodeCount, chainCount = map(int, content[0].split())
    band = []
    source = []
    dest = []
    graph = []
    for line in content[1:nodeCount + 1]:
        graph.append(list(map(int, line.split())))
    for line in content[nodeCount + 1:nodeCount * 2 + 1]:
        band.append(list(map(float, line.split())))
    nodeFile = content[nodeCount * 2 + 1:]
    for i in range(chainCount):
        source.append(list(map(int, nodeFile[i + i].split())))
        dest.append(list(map(int, nodeFile[i + i + 1].split())))
    readfile.close()

    # 输入graph和band为邻接表形式，转换成邻接矩阵
    tmpGraph = [[0 for _ in range(nodeCount)] for _ in range(nodeCount)]
    tmpBand = [[0 for _ in range(nodeCount)] for _ in range(nodeCount)]
    for u in range(nodeCount):
        for i in range(len(graph[u])):
            v = graph[u][i]
            tmpGraph[u][v] = 1
            tmpBand[u][v] = band[u][i]
            # print(u, i, band[u][i])
    return tmpGraph, tmpBand, source, dest


_deltaT = 1
graph, band, source, dest= readInfo()  # bandwidth Mbps
nodeCount = len(graph)
routingGraph = list()   # routingGraph[i][j]=list() 记录链i的j节点会向哪些节点发送区块
bandGraph = list()      # bandGraph[i][j][k] 链i的(j,k)边的带宽   有向边
chainCount = len(source)
# buffer = [Queue(maxsize=0) for _ in range(chainCount)]    # 主节点buffer


class Event:
    def __init__(self, type, time, chainID, data):
        self.eventType = type
        self.data = data
        self.time = time
        self.chainID = chainID


class Chain:
    def __init__(self, tmpsource, tmpdest):
        self.blockFinishTime = list()
        self.blockArriveTime = list()
        self.blockSize = list()
        self.blockCount = 0
        self.curBlock = 0       # 已经传输完成的区块数量
        self.RTgraph = list()
        self.BDgraph = list()
        self.source = tmpsource[0]  # int
        self.dest = tmpdest         # list
        self.lastBlockFinishTime = 0.0

    def upd(self, routingGraph, bandGraph):
        self.RTgraph = routingGraph
        self.BDgraph = bandGraph

    def newBlock(self, task):
        self.blockCount += 1
        self.blockArriveTime.append(task.time)
        self.blockSize.append(task.data)
        self.blockFinishTime.append(-1)

    def transmission(self, lastDeltaTaskTime):    # 当前路由方案的传输时间上界
        while self.curBlock < self.blockCount:
            i = self.curBlock
            transTime = self.getTransTime(self.blockSize[i])
            self.blockFinishTime[i] = max(self.blockArriveTime[i], self.lastBlockFinishTime) + transTime
            self.lastBlockFinishTime = self.blockFinishTime[i]
            self.curBlock += 1
            if self.blockFinishTime[i] > lastDeltaTaskTime:
                break

    def getTransTime(self, data):                 # 获得当前区块的最大传输时间 BFS
        vis = [0 for _ in range(nodeCount)]
        vis[self.source] = 1
        q = Queue(maxsize=0)
        q.put(self.source)
        transTimeForNode = [0 for _ in range(nodeCount)]
        while not q.empty():
            now = q.get()
            for next in self.RTgraph[now]:
                if not vis[next]:
                    transTimeForNode[next] = transTimeForNode[now] + data / self.BDgraph[now][next]
                    q.put(next)
                    vis[next] = 1
        MaxTransTime = 0
        for i in self.dest:
            MaxTransTime = max(MaxTransTime, transTimeForNode[i])
        return MaxTransTime


'''
1.deltaT事件
2.收到区块事件
'''


class Edge():
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w


def trans(nodeCount, graph):
    result = []
    for i in range(nodeCount):
        for j in range(nodeCount):
            if i < j and graph[i][j] == 1:
               result.append(Edge(i, j, 1))
    return result


def solve_delta():
    mstForDivide = list()
    for tmpChain in range(chainCount):     # 每次只处理一条链的最短路和MST
        routing = Shortest_path(graph, chain[tmpChain].source, chain[tmpChain].dest)
        routing.main()
        tmp_graph = routing.result
        tmp_graph = trans(nodeCount, tmp_graph)
        mst = Mst(nodeCount, tmp_graph)
        tree = mst.kruskal()   # 邻接表形式 无向边
        mstForDivide.append(tree)

    # source and dest?
    source, dest = [], []
    for i in range(chainCount):
        source.append(chain[i].source)
        dest.append(chain[i].dest)
    data_size = [1.0, 1.0]
    Divide = DivideBand(nodeCount, mstForDivide, band, source, dest, data_size)
    routingGraph = mstForDivide         # 最小生成树实际上就是路由方案
    bandGraph = Divide.binary_search()   # 分配带宽
    return routingGraph, bandGraph


if __name__ == '__main__':
    testOutput = open('./Output/test_output.txt', 'w')
    # 插入区块到达事件
    traceFile = open('./Input/BlockTrace.txt', 'r')
    eventList = list()
    for line in traceFile.readlines():
        event_type, time, chainID, data = list(map(eval, line.split()))
        eventList.append(Event(event_type, time, chainID, data))   # time时刻在chainID号链到达一个大小为data的block

    chain = []
    for i in range(chainCount):
        chain.append(Chain(source[i], dest[i]))

    lastDeltaTaskTime = _deltaT
    for task in eventList:
        if task.eventType == 1:              # deltaT事件
            routingGraph, bandGraph = solve_delta()
            # print(routingGraph, file=testOutput)
            for i in range(chainCount):
                chain[i].upd(routingGraph[i], bandGraph[i])
            lastDeltaTaskTime += _deltaT
            for i in range(chainCount):     # 在堵塞的情况下，更新路由后应该接着传输之前没传完的区块
                chain[i].transmission(lastDeltaTaskTime)

        if task.eventType == 2:              # 新区块达到事件
            chain[task.chainID].newBlock(task)
            chain[task.chainID].transmission(lastDeltaTaskTime)
        # 如果传输和配置新路由冲突了怎么办: 不考虑

    print("blockID\tArriveTime\tFinishTime\tSolveTime", file=testOutput)
    cnt = 0
    result = 0.0
    for _ in range(chainCount):
        for it in range(chain[_].blockCount):
            cnt += 1
            result += chain[_].blockFinishTime[it]-chain[_].blockArriveTime[it]
            print(cnt, '\t', chain[_].blockArriveTime[it], '\t', '%.3lf' % chain[_].blockFinishTime[it],
                  '\t', '%.3f' % (chain[_].blockFinishTime[it]-chain[_].blockArriveTime[it]), file=testOutput)
    print("%.2f" % (result/cnt), file=testOutput)

    for p in range(chainCount):
        for i in range(nodeCount):
            for j in range(len(chain[p].RTgraph[i])):
                print(chain[p].RTgraph[i][j], end=' ', file=testOutput)
            print(file=testOutput)
        print(file=testOutput)
        for i in range(nodeCount):
            for j in range(nodeCount):
                print("\t%.2f" % chain[p].BDgraph[i][j], end=' ', file=testOutput)
            print(file=testOutput)
        print(file=testOutput)
    # 区块传输不会在中间节点缓存，因此同一个区块链中，上游链路的带宽一定不小于下游链路
