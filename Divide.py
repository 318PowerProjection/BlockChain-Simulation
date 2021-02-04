from queue import Queue


class DivideBand:
    def __init__(self, n, Tree, Bandwidth, source, destination, data_size):    # n个多播组
        self.Tree = Tree  # Tree[ [next1,next2,...], ] 三维list T*N*N 每棵树邻接表存储
        self.bandwidth = Bandwidth
        self.n = n
        self.cost = [[0 for _ in range(self.n)] for __ in range(self.n)]
        self.source = source    # list[]
        self.destination = destination      # list[ list[], ... ]
        self.block_size = data_size
        self.tmp_delay = 0.0

        self.depth = [[0 for _ in range(self.n)] for __ in range(len(self.Tree))]    # 二维list depth[i][j] 链i中的节点j的深度
        for i in range(len(self.Tree)):
            self.calc_depth(i)

        self.tmpBandGraph = []
        for i in range(len(self.Tree)):
            self.tmpBandGraph.append([[0 for _i in range(self.n)] for _j in range(self.n)])

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
                if v != self.source[now] and len(nowTree[v]) == 1:
                    q.put(v)
        self.Tree[now] = nowTree

    def binary_search(self):
        for i in range(len(self.Tree)):
            self.cut(i)      # 剪枝
        L = 0.0
        R = 16.0      # s  设置传输时延上限
        bandGraph = self.tmpBandGraph.copy()
        while R - L > 1e-3:
            self.tmp_delay = (R+L)/2.0
            Judge = self.check()
            if Judge:
                R = self.tmp_delay
                bandGraph = self.tmpBandGraph.copy()
            else:
                L = self.tmp_delay
        return bandGraph

    def check(self):
        for i in range(len(self.Tree)):
            for j in range(self.n):
                for k in range(self.n):
                    self.tmpBandGraph[i][j][k] = 0

        for i in range(len(self.Tree)):
            self.calc_band_of_each_tree(i, self.source[i])      # calculate the cost of each edge
            self.recalc_band(i, -1, self.source[i])             # 子树里按照最大的重新分配带宽

        self.cost = [[0 for _i in range(self.n)] for _j in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                for p in range(len(self.Tree)):
                    self.cost[i][j] += self.tmpBandGraph[p][i][j]   # 计算总的带宽消耗
        for i in range(self.n):      # check
            for j in range(self.n):
                if self.cost[i][j] > self.bandwidth[i][j]:
                    return False
        return True

    def calc_band_of_each_tree(self, now, u):
        max_band = 0
        for v in self.Tree[now][u]:
            if self.depth[now][v] > self.depth[now][u]:
                if len(self.Tree[now][v]) == 1:                         # 如果是叶节点的话直接计算所需要的带宽
                    tmp_band = self.depth[now][v] * self.block_size[now] / self.tmp_delay
                else:
                    tmp_band = self.calc_band_of_each_tree(now, v)
                self.tmpBandGraph[now][u][v] = tmp_band
                self.tmpBandGraph[now][v][u] = tmp_band
                max_band = max(max_band, tmp_band)
        return max_band

    def recalc_band(self, now, band, u):
        if u != self.source[now] and len(self.Tree[now][u]) == 1:
            return
        for v in self.Tree[now][u]:
            if self.depth[now][v] < self.depth[now][u]:
                continue
            if band == -1:
                tmp_band = self.tmpBandGraph[now][u][v]     # 当u是主节点的时候，v是这课子树的根节点，整棵树的带宽都应该变成band[u][v]
            else:
                tmp_band = band
            self.tmpBandGraph[now][u][v] = tmp_band
            self.tmpBandGraph[now][v][u] = tmp_band
            self.recalc_band(now, band, v)
        return
