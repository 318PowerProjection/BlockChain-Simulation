# 一次只处理一个多播组的最短路，T个多播组需要生成T个class对象
class Shortest_path:
    def __init__(self, graph, source, dest):
        self.graph = graph      # graph 二维邻接矩阵
        self.source = source    # source int
        self.dest = dest        # dest list
        self.n = len(graph)
        self.result = [[0 for _ in range(self.n)] for _ in range(self.n)]
        self.dist = [[0xfffffff for _ in range(self.n)] for _ in range(self.n)]

    def main(self):
        self.floyd()
        for i in range(len(self.dest)):
            self.find_path(self.dest[i])

    def floyd(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.graph[i][j] != 0:
                    self.dist[i][j] = 1
        for i in range(self.n):
            self.dist[i][i] = 0
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    self.dist[i][j] = min(self.dist[i][j], self.dist[i][k]+self.dist[k][j])

    def find_path(self, t):
        s = self.source
        now = t
        while now != s:
            for i in range(self.n):
                if self.dist[s][now] == self.dist[s][i]+1 and self.graph[i][now] == 1:
                    self.result[i][now] = self.result[now][i] = 1
                    now = i
                    break
