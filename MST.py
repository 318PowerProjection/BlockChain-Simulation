class Mst:
    def __init__(self, n, E):  # G[i][j]为多播组Ti中，e[i][j]的选中概率
        self.n = n
        self.edge = E
        self.m = len(self.edge)
        self.fa = [0 for _ in range(self.n)]
        for i in range(self.n):
            self.fa[i] = i

    def get_fa(self, x):
        if self.fa[x] == x:
            return x
        self.fa[x] = self.get_fa(self.fa[x])
        return self.fa[x]

    def kruskal(self):
        # def cmp(tmp):
        #    return tmp[2]
        # self.edge.sort(key=cmp)

        for i in range(self.m-1):
            for j in range(i+1, self.m):
                if self.edge[i].w > self.edge[j].w:
                    self.edge[i], self.edge[j] = self.edge[j], self.edge[i]

        count = 0
        output = [[] for _ in range(self.n)]        # 输出无向图邻接表
        for i in range(self.m):
            u = self.edge[i].u
            v = self.edge[i].v
            fu = self.get_fa(u)
            fv = self.get_fa(v)
            if fu == fv:
                continue

            self.fa[fu] = fv
            output[u].append(v)
            output[v].append(u)
            count += 1
            if count == self.n-1:
                break
        return output
