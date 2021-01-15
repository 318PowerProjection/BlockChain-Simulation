from queue import Queue


class DivideBand:
    def __init__(self, n, Tree, Bandwidth, source, destination):    # n个多播组
        self.Tree = Tree  # Tree[ [next1,next2,...], ] 三维list T*N*N 每棵树邻接表存储
        self.bandwidth = Bandwidth
        self.n = n
        self.cost = [[0 for _ in range(self.n)] for __ in range(self.n)]
        self.source = source    # list[]
        self.destination = destination      # list[ list[], ... ]

        self.depth = [[0 for _ in range(self.n)] for __ in range(len(self.Tree))]    # 二维list depth[i][j] 链i中的节点j的深度
        for i in range(len(self.Tree)):
            self.calc_depth(i)

        self.tmpBandGraph = []
        for i in range(len(self.Tree)):
            self.tmpBandGraph.append([[0 for _ in range(self.n)] for __ in range(self.n)])

    def calc_depth(self, now):
        q = Queue(maxsize=0)
        q.put(self.source[now])
        self.depth[now][self.source[now]] = 0
        vis = [0 for _ in range(self.n)]
        vis[self.source[now]] = 1

        while not q.empty():  # 这里求depth可以放到init中一遍完成
            u = q.get()
            next = self.Tree[now][u]
            for v in next:
                if vis[v]:
                    continue
                q.put(v)
                vis[v] = 1
                self.depth[now][v] = self.depth[now][u] + 1

    def cut(self, now):     # 剪枝是移除所有不是目的节点的叶节点
        nowTree = self.Tree[now]
        q = Queue(maxsize=0)
        for i in range(len(nowTree)):
            if i != self.source[now] and len(nowTree[i]) == 1:
                q.put(i)
        while not q.empty():
            u = q.get()
            if u not in self.destination[now]:
                v = nowTree[u][0]
                nowTree[v].remove(u)
                if v!=self.source[now] and len(nowTree[v]) == 1:
                    q.put(v)
        self.Tree[now] = nowTree

    def binary_search(self):
        for i in range(len(self.Tree)):
            self.cut(i)      # 剪枝

        blocksize = 1.0           # 默认区块大小为1Mb
        L = 0.0
        R = 16.0      # s  设置传输时延上限
        bandGraph = self.tmpBandGraph.copy()
        while R - L > 1e-3:
            TmpDelay = (R+L)/2.0
            Judge = self.check(blocksize/TmpDelay)
            if Judge:
                R = TmpDelay
                bandGraph = self.tmpBandGraph.copy()
            else:
                L = TmpDelay
        return bandGraph

    def check(self, w):
        for i in range(self.n):      # init
            for j in range(self.n):
                self.cost[i][j] = 0
        for i in range(len(self.Tree)):
            for j in range(self.n):
                for k in range(self.n):
                    self.tmpBandGraph[i][j][k] = 0

        for i in range(len(self.Tree)):
            self.dfs_cost(i, self.source[i], w)     # calculate the cost of each edge

        for i in range(self.n):      # check
            for j in range(self.n):
                if self.cost[i][j] > self.bandwidth[i][j]:
                    return False
        return True

    def dfs_cost(self, now, u, w):
        maxcost = 0
        for v in self.Tree[now][u]:
            if self.depth[now][v] > self.depth[now][u]:
                if len(self.Tree[now][v]) == 1:
                    maxcost = self.tmpBandGraph[now][u][v] = self.depth[now][v] * w
                    continue
                tmp = self.dfs_cost(now, v, w)
                self.tmpBandGraph[now][u][v] = tmp
                self.cost[u][v] += tmp
                maxcost = max(tmp, maxcost)
        return maxcost

    '''
    def dfs_cost(self, now, u, w):                  # 链路上游的带宽要不小于下游
        if u in self.destination[now]:
            return self.depth[now][u] * w
        maxcost = 0             # maxcost含义是u到后继结点里最大的带宽
        for v in self.Tree[now][u]:
            if self.depth[now][v] > self.depth[now][u]:
                tmp = self.dfs_cost(now, v, w)
                self.tmpBandGraph[now][u][v] = max(tmp, self.tmpBandGraph[now][u][v])
                self.cost[u][v] += tmp
                maxcost = max(maxcost, tmp)
        return maxcost
    '''
