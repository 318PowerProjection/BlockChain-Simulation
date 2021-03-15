import numpy as np
import math

# 交易到达率改变时间间隔5s
# 泊松分布lamda = 10 平均区块间隔为10ms 一秒生成期望100个
# 区块包含的交易数量transactionInBlock = 30

timeInterval = 25
countInterval = 4   # 时间间隔的数量
transactionInBlock = 30
_lambda = [[15, 15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15]]
delta_T = 10


class MakeTransaction:
    def __init__(self, possionLambda, startTime, endTime):
        self.possionLambda = possionLambda
        self.possionSize = int(timeInterval*1000/self.possionLambda*2)
        self.startTime = startTime
        self.endTime = endTime
        # 生成2倍于期望的交易数，取前timeInterval内的数

    def makeData(self):
        count = 0
        tmpList = list()
        for _ in range(countInterval):
            transactionList = np.random.poisson(lam=self.possionLambda, size=self.possionSize)
            for trade in transactionList:
                tmpList.append(float(trade) / 1000)
                count += 1

        blockTime = tmpList[0]+self.startTime
        blockList = list()
        for i in range(count):
            blockTime += tmpList[i]
            if i % transactionInBlock == 0:
                blockList.append(blockTime)
            if blockTime > self.endTime:
                break

        tran_list = []
        tran_list.append(tmpList[0] + self.startTime)
        tmp_time = tmpList[0] + self.startTime
        for i in range(1, count):
            tmp_time += tmpList[i]
            if tmp_time >= self.endTime:
                break
            tran_list.append(tmp_time)
        return tran_list, blockList


class TaskData:
    def __init__(self, event_type, time, chainID, data):
        self.time = time
        self.chainID = chainID
        self.data = data
        self.event_type = event_type


TaskTrace = list()
tran_trace = []
for i in range(countInterval):
    chain1 = MakeTransaction(_lambda[0][i], i*timeInterval, (i+1)*timeInterval)
    chain2 = MakeTransaction(_lambda[1][i], i*timeInterval, (i+1)*timeInterval)
    chain3 = MakeTransaction(_lambda[2][i], i*timeInterval, (i+1)*timeInterval)
    tran_trace1, blockTrace1 = chain1.makeData()
    tran_trace2, blockTrace2 = chain2.makeData()
    tran_trace3, blockTrace3 = chain3.makeData()
    for trace in blockTrace1:
        TaskTrace.append(TaskData(2, trace, 0, 1))
    for trace in blockTrace2:
        TaskTrace.append(TaskData(2, trace, 1, 1))
    for trace in blockTrace3:
        TaskTrace.append(TaskData(2, trace, 2, 1))
    for tran in tran_trace1:
        tran_trace.append(TaskData(2, tran, 0, 1))
    for tran in tran_trace2:
        tran_trace.append(TaskData(2, tran, 1, 1))
    for tran in tran_trace3:
        tran_trace.append(TaskData(2, tran, 2, 1))

max_time = 0.0
n = len(TaskTrace)
for i in range(n):
    max_time = max(max_time, TaskTrace[i].time)
max_time = math.ceil(max_time)          # 添加delta_T事件
for i in range(max_time+1000):
    if i % delta_T == 0:
        TaskTrace.append(TaskData(1, i, -1, -1))
for i in range(max_time+1000):
    if i % delta_T == 0:
        tran_trace.append(TaskData(1, i, -1, -1))

n = len(TaskTrace)                      # 排序 block trace
for i in range(0, n-1):
    for j in range(i+1, n):
        if TaskTrace[i].time > TaskTrace[j].time:
            TaskTrace[i], TaskTrace[j] = TaskTrace[j], TaskTrace[i]

output = open('BlockTrace.txt', 'w')    # 输出 block trace
for it in TaskTrace:
    print(it.event_type, '%.3f' % it.time, it.chainID, it.data, file=output)
output.close()


n = len(tran_trace)
for i in range(0, n-1):                 # 排序 transaction trace
    for j in range(i+1, n):
        if tran_trace[i].time > tran_trace[j].time:
            tran_trace[i], tran_trace[j] = tran_trace[j], tran_trace[i]

output = open('TransactionTrace.txt', 'w')
for it in tran_trace:
    print(it.event_type, '%.3f' % it.time, it.chainID, it.data, file=output)
output.close()
