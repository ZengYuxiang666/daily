# 这个是最经典的dijkstra算法，可以得到path，简化板的可以在题解中找到
import heapq
# 该算法用与求带权最短录井问题，但是不适用于带负权值的情况
# 当我们调用 heapq.heappush(pq, (new_time, v)) 时，新的 (new_time, v) 会被插入到 pq 中，
# 同时堆会重新排序，确保堆的根节点始终是当前最短的传播时间和对应的节点。
# 看王道的视频就能理解这个算法，因为最小堆的根节点始终是最小的，所有他每次都能找到最小的距离来更新final数组
class Solution(object):
    def networkDelayTime(self, times, n, k):
        # 使用邻接矩阵表示图，初始化为无穷大
        graph = [[float('inf')] * (n + 1) for _ in range(n + 1)]

        # 通过 times 填充图的边
        for x, y, d in times:
            graph[x][y] = d

        # 初始化 dist, final, path 数组
        dist = [float('inf')] * (n + 1)  # dist[i] 表示从 k 到 i 的最短路径
        final = [False] * (n + 1)  # final[i] 表示节点 i 是否已经确定最短路径
        path = [-1] * (n + 1)  # path[i] 表示最短路径上从哪个节点来到了节点 i
        dist[k] = 0  # 从源节点 k 出发，起始时间为 0

        # 使用优先队列来选择最小的 dist 节点
        pq = [(0, k)]  # (当前传播时间, 当前节点)

        while pq:
            current_dist, u = heapq.heappop(pq)  # 获取当前距离最小的节点

            # 如果 u 已经被最终确定最短路径，则跳过
            if final[u]:
                continue

            # 标记节点 u 已经确定了最短路径
            final[u] = True

            # 更新所有邻接节点的距离
            for v in range(1, n + 1):  # 节点编号从 1 到 n
                if graph[u][v] != float('inf') and not final[v]:  # 如果 u 到 v 有边且 v 未处理
                    new_dist = current_dist + graph[u][v]
                    if new_dist < dist[v]:  # 如果新路径更短，更新 dist
                        dist[v] = new_dist
                        path[v] = u  # 记录路径
                        heapq.heappush(pq, (new_dist, v))

        # 检查是否所有节点都可以到达，如果有不可达的节点，返回 -1
        max_time = max(dist[1:])  # 忽略 dist[0]，节点编号从 1 开始
        return max_time if max_time < float('inf') else -1


